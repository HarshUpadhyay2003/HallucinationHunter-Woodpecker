import argparse, json, os
import torch
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image

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
    parser.add_argument("--output-jsonl", default="results/out.jsonl")
    parser.add_argument("--output-csv", default="results/out.csv")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"🚀 Running on {device}")

    # Load models
    print("Loading BLIP-2...")
    blip2_proc = Blip2Processor.from_pretrained(args.blip2_path)
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(args.blip2_path, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    blip2_model.to(device)

    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(args.clip_path).to(device)
    clip_proc = CLIPProcessor.from_pretrained(args.clip_path)

    # Prepare results folder
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    jsonl_out = open(args.output_jsonl, "w", encoding="utf-8")

    # Load dataset
    data = json.load(open(args.dataset_json))
    print(f"Processing {len(data)} image-caption pairs...")

    for item in tqdm(data):
        img_path = os.path.join(args.images_root, item["image"])
        image = load_image(img_path)
        if image is None: 
            continue

        # 1. Generate BLIP-2 caption (optional, can skip if already provided)
        inputs = blip2_proc(images=image, text=item["query"], return_tensors="pt").to(device)
        with torch.no_grad():
            gen = blip2_model.generate(**inputs, max_new_tokens=30)
            caption = blip2_proc.decode(gen[0], skip_special_tokens=True)

        # 2. Compute CLIP similarity between generated caption & ground truth
        clip_inputs = clip_proc(text=[caption, item["text"]], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_feat = clip_model.get_image_features(**{k: v for k, v in clip_inputs.items() if k.startswith("pixel")})
            txt_feats = clip_model.get_text_features(**{k: v for k, v in clip_inputs.items() if k.startswith("input")})
            sim = torch.nn.functional.cosine_similarity(img_feat, txt_feats).mean().item()

        # Save to JSONL
        out = {
            "image": item["image"],
            "generated_text": caption,
            "ground_truth": item["text"],
            "clip_sim": round(float(sim), 4)
        }
        jsonl_out.write(json.dumps(out) + "\n")

    jsonl_out.close()
    print(f"✅ Saved results to {args.output_jsonl}")

if __name__ == "__main__":
    main()
