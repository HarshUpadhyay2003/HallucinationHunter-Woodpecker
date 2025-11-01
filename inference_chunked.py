import os, argparse, json, torch, gc, time, warnings, logging, re
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
from types import SimpleNamespace
from vis_corrector import Corrector
from modules.hallucination_detector import calculate_hcs_score
from models.detector import Detector
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# ===========================
# ⚙️ Environment Setup
# ===========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)


def print_once(*args, **kwargs):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    print(*args, **kwargs)


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def simple_entity_extract(text: str):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    stop = {"the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are","this","that","these","those"}
    tokens = [t for t in tokens if len(t) > 2 and t not in stop]
    unique = []
    for t in tokens:
        if t not in unique:
            unique.append(t)
    return [".".join(unique[:8])] if unique else ["none"]


# ===========================
# 📦 Dataset Class
# ===========================
class InferenceDataset(Dataset):
    def __init__(self, data, root):
        self.data = data
        self.root = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = os.path.join(self.root, item["image"])
        img = load_image(path)
        return img, item


# ===========================
# 🚀 Main
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--blip2-path", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--val-model-path", type=str, default="models/val_model.pth")
    parser.add_argument("--qa2c-model-path", default="models/qa2c_model")
    parser.add_argument("--output-jsonl", default="results/out_full.jsonl")
    parser.add_argument("--output-csv", default="results/out_full.csv")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--detector-config")
    parser.add_argument("--detector-model")
    parser.add_argument("--enable-hcs", action="store_true")
    parser.add_argument("--cache-dir", default="./cache_dir")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--api-base", default="none")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--area-threshold", type=float, default=0.001)
    args = parser.parse_args()

    # ---------- Device setup ----------
    if torch.cuda.device_count() > 1:
        print_once(f"🚀 Using {torch.cuda.device_count()} GPUs")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
        print_once(f"🚀 Running on {device}")

    # ---------- Load BLIP-2 ----------
    print_once("🔹 Loading BLIP-2...")
    blip2_proc = Blip2Processor.from_pretrained(args.blip2_path)
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        args.blip2_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if hasattr(torch, "compile") and device == "cuda":
        try:
            blip2_model = torch.compile(blip2_model)
        except Exception:
            pass

    # ---------- Load CLIP ----------
    print_once("🔹 Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(args.clip_path, torch_dtype=torch.float16).to(device)
    clip_proc = CLIPProcessor.from_pretrained(args.clip_path)

    # ---------- Load GroundingDINO ----------
    print_once("🔹 Loading GroundingDINO...")
    if args.detector_model and os.path.exists(args.detector_model):
        from groundingdino.util.inference import load_model
        det_model = load_model(
            model_config_path=args.detector_config,
            model_checkpoint_path=args.detector_model
        ).to(device)
    else:
        det_model = None

    # ---------- Multi-GPU ----------
    if torch.cuda.device_count() > 1:
        blip2_model = torch.nn.DataParallel(blip2_model)
        clip_model = torch.nn.DataParallel(clip_model)
        if det_model:
            det_model = torch.nn.DataParallel(det_model)

    # ---------- Corrector ----------
    model_args = SimpleNamespace(
        api_key=args.api_key,
        api_base=args.api_base,
        detector_config=args.detector_config,
        detector_model_path=args.detector_model,
        cache_dir=args.cache_dir,
        device=device,
        det_model=det_model,
        val_model_path=args.val_model_path,
        qa2c_model_path=args.qa2c_model_path
    )
    corrector = Corrector(model_args)

    # ---------- HCS Detector (Initialized once) ----------
    detector = None
    if args.enable_hcs and args.detector_model and args.detector_config:
        try:
            det_args = SimpleNamespace(
                detector_config=args.detector_config,
                detector_model_path=args.detector_model,
                cache_dir=args.cache_dir,
            )
            detector = Detector(det_args)
            print_once("✅ HCS Detector initialized once successfully.")
        except Exception as e:
            print_once(f"⚠️ Failed to initialize HCS Detector: {e}")

        # ---------- Dataset ----------
    data = json.load(open(args.dataset_json))
    total = len(data)
    chunk_size = args.chunk_size
    num_chunks = (total + chunk_size - 1) // chunk_size
    batch_size = args.batch_size
    print_once(f"🧩 Total {total} samples → {num_chunks} chunks × {chunk_size} each (batch={batch_size})")

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    # ---------- Auto-resume logic ----------
    processed_images = set()
    if os.path.exists(args.output_jsonl) and os.path.getsize(args.output_jsonl) > 0:
        print_once(f"♻️ Resuming from existing file: {args.output_jsonl}")
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "image" in rec:
                        processed_images.add(rec["image"])
                except Exception:
                    continue
        print_once(f"✅ Found {len(processed_images)} already processed samples — skipping those.")
        out_jsonl = open(args.output_jsonl, "a", encoding="utf-8")  # append mode
    else:
        out_jsonl = open(args.output_jsonl, "w", encoding="utf-8")  # new file

    start_time = time.time()

    # ---------- Helper: safe batch handling ----------
    def clean_batch(batch):
        if isinstance(batch, tuple) and len(batch) == 2:
            return list(batch[0]), list(batch[1])
        return batch

    # ---------- Chunk Loop ----------
    for ci in range(num_chunks):
        t0 = time.time()
        start, end = ci * chunk_size, min((ci + 1) * chunk_size, total)
        subset = data[start:end]
        print_once(f"\n🔹 Processing chunk {ci+1}/{num_chunks} ({start}–{end})")

        inf_dataset = InferenceDataset(subset, args.images_root)
        inf_loader = DataLoader(
            inf_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Stable for tokenizer + image ops
            pin_memory=True,
            collate_fn=lambda b: list(zip(*b)),
        )

        chunk_results = []

        for imgs, valid_items in tqdm(inf_loader, desc=f"Chunk {ci+1}/{num_chunks}", ncols=90):
            imgs, valid_items = clean_batch((imgs, valid_items))
            valid = [(img, it) for img, it in zip(imgs, valid_items) if img is not None]
            if not valid:
                continue

            imgs, valid_items = zip(*valid)
            imgs = list(imgs)

            try:
                inputs = blip2_proc(
                    images=imgs,
                    text=[it["query"] for it in valid_items],
                    return_tensors="pt",
                    padding=True
                ).to(device, torch.float16 if device == "cuda" else torch.float32)
            except Exception as e:
                print_once(f"⚠️ Skipping batch: {e}")
                continue

            with torch.no_grad():
                gen = (
                    blip2_model.module.generate(**inputs, max_new_tokens=30)
                    if hasattr(blip2_model, "module")
                    else blip2_model.generate(**inputs, max_new_tokens=30)
                )
                captions = [blip2_proc.decode(g, skip_special_tokens=True) for g in gen]

            for item, caption in zip(valid_items, captions):
                sample = {
                    'img_path': os.path.join(args.images_root, item["image"]),
                    'input_desc': caption,
                    'query': item.get('query', 'Describe this image.')
                }

                # Corrector
                try:
                    corrected = corrector.correct(sample)
                    corrected_output = corrected.get('output', caption)
                except Exception:
                    corrected_output = caption

                # CLIP Similarities
                try:
                    with torch.no_grad():
                        img_inputs = clip_proc(images=Image.open(sample["img_path"]).convert("RGB"), return_tensors="pt").to(device, torch.float16)
                        img_feat = clip_model.get_image_features(**img_inputs)
                        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)

                        texts = [caption, item["text"], corrected_output]
                        txt_inputs = clip_proc(text=texts, return_tensors="pt", padding=True).to(device, torch.float16)
                        txt_feats = clip_model.get_text_features(**txt_inputs)
                        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)

                        sims = (img_feat @ txt_feats.T).squeeze(0).tolist()
                        sim_generated, sim_ground, sim_corrected = map(float, sims)
                except Exception:
                    sim_generated = sim_ground = sim_corrected = 0.0

                # HCS computation (stable, no reinit each time)
                hcs_score = None
                if detector is not None:
                    try:
                        ent_list = simple_entity_extract(caption)
                        det_sample = {
                            'img_path': sample["img_path"],
                            'named_entity': ent_list,
                            'box_threshold': args.box_threshold,
                            'area_threshold': args.area_threshold,
                        }
                        det_sample = detector.detect_objects(det_sample)
                        hcs_score = calculate_hcs_score({
                            'input_desc': caption,
                            'entity_info': det_sample.get('entity_info', {})
                        }, device=device)
                    except Exception as e:
                        print_once(f"⚠️ HCS failed for {item['image']}: {e}")

                chunk_results.append({
                    "image": item["image"],
                    "query": item.get("query"),
                    "generated_text": caption,
                    "ground_truth": item.get("text", ""),
                    "corrected_output": corrected_output,
                    "clip_sim_generated": round(sim_generated, 4),
                    "clip_sim_ground": round(sim_ground, 4),
                    "clip_sim_corrected": round(sim_corrected, 4),
                    "hcs_score": hcs_score
                })

            torch.cuda.empty_cache()
            gc.collect()

        out_jsonl.writelines([json.dumps(r) + "\n" for r in chunk_results])
        out_jsonl.flush()

        dur = (time.time() - t0) / 60
        elapsed = (time.time() - start_time) / 60
        eta = dur * (num_chunks - ci - 1)
        print_once(f"✅ Chunk {ci+1} done in {dur:.2f} min | Elapsed {elapsed:.1f} min | ETA {eta:.1f} min")

        torch.cuda.empty_cache()
        gc.collect()

    out_jsonl.close()
    print_once(f"\n✅ All chunks done → {args.output_jsonl}")

    results = [json.loads(l) for l in open(args.output_jsonl, encoding="utf-8")]
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print_once(f"✅ CSV saved → {args.output_csv}")


if __name__ == "__main__":
    main()
