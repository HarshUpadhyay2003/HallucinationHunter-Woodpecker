import json, os

# --- CONFIGURATION ---
# Path to COCO caption annotations file
ANNOT_PATH = "datasets/annotations/captions_val2017.json"   # or captions_train2017.json
# Path to where the generated dataset JSON will be saved
OUT_PATH = "datasets/processed_pairs.json"
# Root folder where images are stored
IMAGES_ROOT = "datasets/val2017"                     # adjust if needed

# --- LOAD COCO ANNOTATIONS ---
print(f"Loading COCO captions from {ANNOT_PATH} ...")
with open(ANNOT_PATH, "r") as f:
    coco_data = json.load(f)

# Create a map from image_id → file_name
id_to_filename = {}
for img in coco_data.get("images", []):
    id_to_filename[img["id"]] = img["file_name"]

# Build list of {image, query, text}
entries = []
for ann in coco_data.get("annotations", []):
    img_id = ann["image_id"]
    caption = ann["caption"].strip()
    if img_id not in id_to_filename:
        continue
    img_file = id_to_filename[img_id]
    img_path = os.path.join(IMAGES_ROOT, img_file)
    if not os.path.exists(img_path):
        # skip missing images (optional)
        continue
    entries.append({
        "image": img_file,
        "query": "Describe this image.",
        "text": caption
    })
print(f"Found {len(entries)} valid image–caption pairs.")
print("Example entries:", entries[:3])

# --- SAVE OUTPUT ---
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(entries)} entries to {OUT_PATH}")
