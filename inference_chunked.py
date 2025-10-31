# chunked.py

import argparse, json, os, torch, gc, time, warnings, logging
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
from types import SimpleNamespace
from vis_corrector import Corrector
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

# ===========================
# ⚙️ Environment Setup
# ===========================
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
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

# ===========================
# 🚀 Main
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--blip2-path", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--val-model-path", type=str, default="models/val_model.pth", help="Path to validation model used by Answerer")
    parser.add_argument("--qa2c-model-path", default="models/qa2c_model")
    parser.add_argument("--output-jsonl", default="results/out_full.jsonl")
    parser.add_argument("--output-csv", default="results/out_full.csv")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--detector-model", required=True)
    parser.add_argument("--cache-dir", default="./cache_dir")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--api-base", default="none")
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

    # Optional compile for PyTorch 2.0+
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
    if args.detector_model.endswith(".pth"):
        from groundingdino.util.inference import load_model
        det_model = load_model(
            model_config_path=args.detector_config,
            model_checkpoint_path=args.detector_model
        ).to(device)
    else:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.detector_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

    # ---------- Multi-GPU ----------
    if torch.cuda.device_count() > 1:
        blip2_model = torch.nn.DataParallel(blip2_model)
        clip_model = torch.nn.DataParallel(clip_model)
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
    model_args.val_model_path = args.val_model_path

    corrector = Corrector(model_args)

    # ---------- Dataset ----------
    data = json.load(open(args.dataset_json))
    total = len(data)
    chunk_size = args.chunk_size
    num_chunks = (total + chunk_size - 1) // chunk_size
    batch_size = args.batch_size
    print_once(f"🧩 Total {total} samples → {num_chunks} chunks × {chunk_size} each (batch={batch_size})")

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    out_jsonl = open(args.output_jsonl, "w", encoding="utf-8")

    start_time = time.time()

    # ---------- Chunk Loop ----------
    for ci in range(num_chunks):
        t0 = time.time()
        start, end = ci * chunk_size, min((ci + 1) * chunk_size, total)
        subset = data[start:end]
        print_once(f"\n🔹 Processing chunk {ci+1}/{num_chunks} ({start}–{end})")

        chunk_results = []

        for bi in range(0, len(subset), batch_size):
            batch = subset[bi:bi + batch_size]

            # Parallel image loading
            with ThreadPoolExecutor(max_workers=8) as ex:
                imgs = list(ex.map(lambda it: load_image(os.path.join(args.images_root, it["image"])), batch))

            valid = [(img, it) for img, it in zip(imgs, batch) if img is not None]
            if not valid:
                continue

            imgs, valid_items = zip(*valid)

            # BLIP-2 batched caption generation
            inputs = blip2_proc(images=imgs, text=[it["query"] for it in valid_items],
                                return_tensors="pt", padding=True).to(device, torch.float16)
            with torch.no_grad():
                gen = blip2_model.module.generate(**inputs, max_new_tokens=30) if hasattr(blip2_model, "module") else blip2_model.generate(**inputs, max_new_tokens=30)
                captions = [blip2_proc.decode(g, skip_special_tokens=True) for g in gen]

            # Process each item
            for item, caption in zip(valid_items, captions):
                sample = {'img_path': os.path.join(args.images_root, item["image"]),
                          'input_desc': caption,
                          'query': item.get('query', 'Describe this image.')}
                try:
                    corrected = corrector.correct(sample)
                    corrected_output = corrected.get('output', caption)
                except Exception:
                    corrected_output = caption

                # CLIP similarity
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

                chunk_results.append({
                    "image": item["image"],
                    "query": item.get("query"),
                    "generated_text": caption,
                    "ground_truth": item["text"],
                    "corrected_output": corrected_output,
                    "clip_sim_generated": round(sim_generated, 4),
                    "clip_sim_ground": round(sim_ground, 4),
                    "clip_sim_corrected": round(sim_corrected, 4)
                })

            torch.cuda.empty_cache()

        # Save chunk
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
