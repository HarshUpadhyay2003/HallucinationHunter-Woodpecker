"""
Extract Images with Truth and Generated Descriptions

This script extracts images from val2017 folder along with their truth descriptions
and generated descriptions from out_full_hcs_1000.jsonl.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import argparse


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    print(f"📖 Loading data from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    print(f"✅ Loaded {len(data)} entries")
    return data


def load_processed_pairs(file_path: str) -> Dict[str, Dict]:
    """Load processed_pairs.json and create a mapping by image filename."""
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: {file_path} not found. Skipping processed_pairs.json mapping.")
        return {}
    
    print(f"📖 Loading processed_pairs.json from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    # Create mapping by image filename
    mapping = {}
    for pair in pairs:
        mapping[pair['image']] = pair
    
    print(f"✅ Loaded {len(mapping)} image mappings from processed_pairs.json")
    return mapping


def extract_images_with_descriptions(
    jsonl_path: str,
    images_root: str,
    output_dir: str,
    processed_pairs_path: str = None,
    copy_images: bool = True
):
    """
    Extract images with their truth and generated descriptions.
    
    Args:
        jsonl_path: Path to out_full_hcs_1000.jsonl
        images_root: Path to val2017 folder containing images
        output_dir: Output directory for extracted images and descriptions
        processed_pairs_path: Optional path to processed_pairs.json for reference
        copy_images: If True, copy images to output directory. If False, just create description files.
    """
    # Load data
    jsonl_data = load_jsonl(jsonl_path)
    processed_pairs = load_processed_pairs(processed_pairs_path) if processed_pairs_path else {}
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, "images")
    descriptions_output_dir = os.path.join(output_dir, "descriptions")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(descriptions_output_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total': len(jsonl_data),
        'images_found': 0,
        'images_missing': 0,
        'images_copied': 0,
        'descriptions_saved': 0
    }
    
    # Process each entry
    print(f"\n📦 Processing {stats['total']} entries...")
    
    all_descriptions = []
    
    for idx, entry in enumerate(jsonl_data):
        image_filename = entry.get('image', '')
        if not image_filename:
            print(f"⚠️  Entry {idx+1}: Missing image filename, skipping...")
            continue
        
        # Find image path
        image_path = os.path.join(images_root, image_filename)
        
        if not os.path.exists(image_path):
            print(f"⚠️  Entry {idx+1}: Image not found: {image_path}")
            stats['images_missing'] += 1
            continue
        
        stats['images_found'] += 1
        
        # Copy image if requested
        if copy_images:
            output_image_path = os.path.join(images_output_dir, image_filename)
            try:
                shutil.copy2(image_path, output_image_path)
                stats['images_copied'] += 1
            except Exception as e:
                print(f"⚠️  Entry {idx+1}: Failed to copy image: {e}")
                continue
        
        # Prepare description data
        description_data = {
            'image': image_filename,
            'truth_description': entry.get('ground_truth', ''),
            'generated_description': entry.get('generated_text', ''),
            'corrected_description': entry.get('corrected_output', ''),
            'query': entry.get('query', ''),
            'clip_sim_generated': entry.get('clip_sim_generated'),
            'clip_sim_ground': entry.get('clip_sim_ground'),
            'clip_sim_corrected': entry.get('clip_sim_corrected'),
            'hcs_score': entry.get('hcs_score')
        }
        
        # Add processed_pairs.json data if available
        if image_filename in processed_pairs:
            description_data['processed_pairs_reference'] = processed_pairs[image_filename]
        
        # Save individual description file
        description_filename = os.path.splitext(image_filename)[0] + '_description.json'
        description_filepath = os.path.join(descriptions_output_dir, description_filename)
        
        try:
            with open(description_filepath, 'w', encoding='utf-8') as f:
                json.dump(description_data, f, ensure_ascii=False, indent=2)
            stats['descriptions_saved'] += 1
        except Exception as e:
            print(f"⚠️  Entry {idx+1}: Failed to save description: {e}")
            continue
        
        # Add to all descriptions list
        all_descriptions.append(description_data)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{stats['total']} entries...")
    
    # Save combined descriptions file
    combined_output_path = os.path.join(output_dir, 'all_descriptions.json')
    try:
        with open(combined_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_descriptions, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved combined descriptions to {combined_output_path}")
    except Exception as e:
        print(f"⚠️  Failed to save combined descriptions: {e}")
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, 'extraction_summary.json')
    summary = {
        'input_jsonl': jsonl_path,
        'images_root': images_root,
        'output_directory': output_dir,
        'statistics': stats,
        'output_files': {
            'images_directory': images_output_dir if copy_images else None,
            'descriptions_directory': descriptions_output_dir,
            'combined_descriptions': combined_output_path
        }
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total entries processed: {stats['total']}")
    print(f"Images found: {stats['images_found']}")
    print(f"Images missing: {stats['images_missing']}")
    if copy_images:
        print(f"Images copied: {stats['images_copied']}")
    print(f"Descriptions saved: {stats['descriptions_saved']}")
    print(f"\n✅ Output directory: {output_dir}")
    print(f"   - Images: {images_output_dir}" if copy_images else "")
    print(f"   - Descriptions: {descriptions_output_dir}")
    print(f"   - Combined: {combined_output_path}")
    print(f"   - Summary: {summary_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract images with truth and generated descriptions from JSONL file"
    )
    parser.add_argument(
        "--jsonl",
        default="results/out_full_hcs_1000.jsonl",
        help="Path to input JSONL file (default: results/out_full_hcs_1000.jsonl)"
    )
    parser.add_argument(
        "--images-root",
        default="datasets/val2017",
        help="Path to val2017 folder containing images (default: datasets/val2017)"
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_images",
        help="Output directory for extracted images and descriptions (default: extracted_images)"
    )
    parser.add_argument(
        "--processed-pairs",
        default="datasets/processed_pairs.json",
        help="Optional path to processed_pairs.json for reference (default: datasets/processed_pairs.json)"
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Don't copy images, only save description files"
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.jsonl):
        print(f"❌ Error: JSONL file not found: {args.jsonl}")
        return
    
    if not os.path.exists(args.images_root):
        print(f"❌ Error: Images root directory not found: {args.images_root}")
        return
    
    # Extract images and descriptions
    extract_images_with_descriptions(
        jsonl_path=args.jsonl,
        images_root=args.images_root,
        output_dir=args.output_dir,
        processed_pairs_path=args.processed_pairs if os.path.exists(args.processed_pairs) else None,
        copy_images=not args.no_copy_images
    )


if __name__ == "__main__":
    main()

