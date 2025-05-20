import os
import json
from PIL import Image
import torch
import clip

# --- CONFIG ---
IMAGE_FOLDER = "photos"  # Replace with your image folder path
OUTPUT_FILE = "gallery_metadata.json"
TAGS = json.load(open("tags.json")).get("allTags")
CONFIDENCE_THRESHOLD = 0.5  # Add threshold parameter
BATCH_SIZE = 100  # Process tags in batches

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

def process_image(image_path):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        tag_weights = {}
        # Process tags in batches
        for i in range(0, len(TAGS), BATCH_SIZE):
            batch_tags = TAGS[i:i + BATCH_SIZE]
            text_inputs = clip.tokenize(batch_tags).to(device)
            
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Add high-confidence tags from this batch
            for j, score in enumerate(similarity[0]):
                score_float = float(score)
                if score_float >= CONFIDENCE_THRESHOLD:
                    tag_weights[batch_tags[j]] = round(score_float, 4)

        return {"image": os.path.basename(image_path), "tags": tag_weights} if tag_weights else None

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- MAIN LOOP ---
def main():
    images = [
        f
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    print(f"Found {len(images)} image(s). Processing...")

    with open(OUTPUT_FILE, "w") as out_file:
        for img_name in images:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            result = process_image(img_path)
            if result:
                out_file.write(json.dumps(result) + "\n")

    print(f"Done. Metadata saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

