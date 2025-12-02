"""
Separate HCS computation script for post-processing inference results.

This script loads JSONL results from inference_chunked.py and adds HCS scores
in a separate pass, allowing the main inference pipeline to run faster.
"""

import json
import os
import argparse
from models.detector import Detector
from modules.hallucination_detector import calculate_hcs_score
from tqdm import tqdm
from types import SimpleNamespace
import re
import torch

# CUDA optimizations for better GPU performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 (faster on A100)
    # Pre-allocate GPU memory for better utilization
    torch.cuda.empty_cache()
    # Reserve memory fraction (0.95 = use 95% of GPU memory)
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.95)
torch.set_grad_enabled(False)  # Disable gradients for inference


def simple_entity_extract(text: str):
    """Extract entities from text for HCS computation."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", 
            "is", "are", "this", "that", "these", "those"}
    tokens = [t for t in tokens if len(t) > 2 and t not in stop]
    unique = []
    for t in tokens:
        if t not in unique:
            unique.append(t)
    return [".".join(unique[:8])] if unique else ["none"]


def main():
    parser = argparse.ArgumentParser(description="Compute HCS scores for pre-generated captions")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL file from inference_chunked.py")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file with HCS scores added")
    parser.add_argument("--detector-config", required=True, help="Path to GroundingDINO config")
    parser.add_argument("--detector-model-path", required=True, help="Path to GroundingDINO model weights")
    parser.add_argument("--cache-dir", default="./cache_dir", help="Cache directory for intermediate files")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Box detection threshold")
    parser.add_argument("--area-threshold", type=float, default=0.001, help="Area threshold for filtering")
    parser.add_argument("--images-root", required=True, help="Root directory for images")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--checkpoint-interval", type=int, default=100, 
                       help="Save checkpoint every N samples (default: 100)")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from existing output file (skip already computed HCS scores)")
    args = parser.parse_args()

    # Initialize detector once
    print(f"🔹 Initializing HCS Detector...")
    det_args = SimpleNamespace(
        detector_config=args.detector_config,
        detector_model_path=args.detector_model_path,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    detector = Detector(det_args)
    print("✅ HCS Detector initialized successfully.")
    
    # Pre-allocate GPU memory for better utilization
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"💾 GPU Memory before: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        # Pre-allocate memory by creating dummy tensors to ensure GPU memory is actively used
        dummy_tensors = []
        try:
            # Allocate memory to maximize GPU utilization
            # Each tensor: 3000x3000x4 bytes = ~36MB
            for _ in range(100):
                dummy = torch.zeros(3000, 3000, dtype=torch.float32, device=args.device)
                dummy_tensors.append(dummy)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            pass  # Silently fail if memory allocation issues
        print(f"💾 GPU Memory after pre-allocation: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"💾 GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        print(f"💾 GPU Memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()*1024**3)/1024**3:.2f} GB")

    # Load input JSONL
    print(f"📖 Loading results from {args.input_jsonl}...")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        all_lines = [json.loads(l.strip()) for l in f if l.strip()]
    
    # Limit to first 100 samples
    lines = all_lines[:100]
    print(f"📊 Limiting to first 100 samples (out of {len(all_lines)} total)...")
    print(f"📊 Processing {len(lines)} samples...")

    # Resume from existing output if requested
    processed_samples = {}
    if args.resume and os.path.exists(args.output_jsonl):
        print(f"🔄 Resuming from existing output: {args.output_jsonl}")
        try:
            with open(args.output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_item = json.loads(line.strip())
                        # Use image path as key to match samples
                        img_key = existing_item.get("image", "")
                        if img_key and existing_item.get("hcs_score") is not None:
                            processed_samples[img_key] = existing_item.get("hcs_score")
            print(f"   Found {len(processed_samples)} already processed samples (will be skipped)")
        except Exception as e:
            print(f"⚠️ Could not load existing output: {e}, starting fresh")

    # Initialize caption cache to avoid recomputing same captions
    caption_cache = {}

    # Process each sample
    updated_results = []
    checkpoint_file = args.output_jsonl + ".checkpoint"
    
    for idx, item in enumerate(tqdm(lines, desc="Computing HCS", ncols=90, mininterval=1.5)):
        # Check if already processed (from resume)
        img_key = item.get("image", "")
        if img_key in processed_samples:
            item["hcs_score"] = processed_samples[img_key]
            updated_results.append(item)
            continue
        
        # Skip if HCS already computed in input
        if item.get("hcs_score") is not None and isinstance(item.get("hcs_score"), dict):
            updated_results.append(item)
            continue

        caption = item.get("generated_text", "")
        img_path = os.path.join(args.images_root, item["image"])
        
        # Check if we've seen this caption before (cache hit)
        if caption in caption_cache:
            item["hcs_score"] = caption_cache[caption]
            updated_results.append(item)
            continue
        
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}, skipping HCS")
            item["hcs_score"] = None
            updated_results.append(item)
            continue

        try:
            # Extract entities
            ent_list = simple_entity_extract(caption)
            det_sample = {
                "img_path": img_path,
                "named_entity": ent_list,
                "box_threshold": args.box_threshold,
                "area_threshold": args.area_threshold,
            }
            
            # Detect objects
            det_result = detector.detect_objects(det_sample)
            
            # Calculate HCS score (function returns sample dict with hcs_scores added)
            hcs_sample = {
                "input_desc": caption,
                "entity_info": det_result.get("entity_info", {})
            }
            hcs_result = calculate_hcs_score(hcs_sample, device=args.device)
            
            # Extract overall HCS score (function modifies dict and returns it)
            if isinstance(hcs_result, dict) and "hcs_scores" in hcs_result:
                item["hcs_score"] = hcs_result["hcs_scores"].get("overall_hcs_score")
            else:
                item["hcs_score"] = hcs_result if not isinstance(hcs_result, dict) else None
            
            # Cache the result for this caption
            if item["hcs_score"] is not None:
                caption_cache[caption] = item["hcs_score"]
            
        except Exception as e:
            print(f"⚠️ HCS failed for {item['image']}: {e}")
            item["hcs_score"] = None

        updated_results.append(item)
        
        # Periodic checkpoint saving
        if (idx + 1) % args.checkpoint_interval == 0:
            # Save checkpoint (all results so far)
            os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                for r in updated_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            # Also update main output file incrementally
            with open(args.output_jsonl, "w", encoding="utf-8") as f:
                for r in updated_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if args.device == "cuda" and torch.cuda.is_available():
                print(f"\n💾 Checkpoint saved at sample {idx+1}/{len(lines)} (GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB)")

    # Final save
    print(f"\n💾 Saving final results to {args.output_jsonl}...")
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in updated_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Remove checkpoint file if exists
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"✅ HCS computation complete → {args.output_jsonl}")
    print(f"📊 Processed {len(updated_results)} samples")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"💾 Final GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")


if __name__ == "__main__":
    main()

