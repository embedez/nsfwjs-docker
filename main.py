import os
import json
from PIL import Image
import torch
import clip
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

# --- CONFIG ---
TAGS = json.load(open("tags.json")).get("allTags")
CONFIDENCE_THRESHOLD = 0.5 
BATCH_SIZE = 100

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

app = FastAPI()

async def process_image(image_data):
    try:
        image = preprocess(Image.open(BytesIO(image_data)).convert("RGB")).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        tag_weights = {}
        
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

        return {"tags": tag_weights} if tag_weights else {"tags": {}}

    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = await process_image(contents)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)