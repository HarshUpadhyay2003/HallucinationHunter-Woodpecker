import argparse, json, os, torch, gc, time
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--blip2-path", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--output-jsonl", default="results/out_full.jsonl")
    parser.add_argument("--output-csv", default="results/out_full.csv")
    parser.add_argument("--chunk-size", type=int, default=2000)   # ✅ larger default
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"🚀 Running on {device}")

    # ---------- Load models ONCE ----------
    print("Loading BLIP-2...")
    blip2_proc = Blip2Processor.from_pretrained(args.blip2_path)
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        args.blip2_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(args.clip_path).to(device)
    clip_proc = CLIPProcessor.from_pretrained(args.clip_path)
    # -------------------------------------

    data = json.load(open(args.dataset_json))
    total = len(data)
    chunk_size = args.chunk_size
    num_chunks = (total + chunk_size - 1) // chunk_size
    print(f"Processing {total} pairs in {num_chunks} chunks of {chunk_size} each")

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    out_jsonl = open(args.output_jsonl, "w", encoding="utf-8")

    start_time = time.time()
    for ci in range(num_chunks):
        t0 = time.time()
        start, end = ci * chunk_size, min((ci + 1) * chunk_size, total)
        subset = data[start:end]
        print(f"\n🧩 Chunk {ci+1}/{num_chunks} ({start}–{end})")

        for item in tqdm(subset):
            img_path = os.path.join(args.images_root, item["image"])
            image = load_image(img_path)
            if image is None:
                continue

            # --- BLIP-2 caption ---
            inputs = blip2_proc(images=image, text=item["query"], return_tensors="pt").to(device)
            with torch.no_grad():
                gen = blip2_model.generate(**inputs, max_new_tokens=30)
                caption = blip2_proc.decode(gen[0], skip_special_tokens=True)

            # --- CLIP similarity ---
            clip_inputs = clip_proc(text=[caption, item["text"]], images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                img_feat = clip_model.get_image_features(**{k: v for k, v in clip_inputs.items() if k.startswith("pixel")})
                txt_feats = clip_model.get_text_features(**{k: v for k, v in clip_inputs.items() if k.startswith("input")})
                sim = torch.nn.functional.cosine_similarity(img_feat, txt_feats).mean().item()

            out_jsonl.write(json.dumps({
                "image": item["image"],
                "generated_text": caption,
                "ground_truth": item["text"],
                "clip_sim": round(float(sim), 4)
            }) + "\n")

        # progress logging
        dur = (time.time() - t0) / 60
        elapsed = (time.time() - start_time) / 60
        eta = dur * (num_chunks - ci - 1)
        print(f"✅ Chunk {ci+1} done in {dur:.2f} min | Elapsed {elapsed:.1f} min | ETA {eta:.1f} min")

        torch.cuda.empty_cache(); gc.collect()

    out_jsonl.close()
    print(f"\n✅ All chunks done → {args.output_jsonl}")

    # summary
    results = [json.loads(l) for l in open(args.output_jsonl)]
    sims = [r["clip_sim"] for r in results]
    print(f"📊 Mean CLIP similarity = {np.mean(sims):.3f} ± {np.std(sims):.3f}")
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"✅ CSV saved → {args.output_csv}")

if __name__ == "__main__":
    main()
