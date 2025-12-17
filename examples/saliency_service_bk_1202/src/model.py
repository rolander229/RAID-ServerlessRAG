import torch
from transformers import CLIPModel, CLIPProcessor

model_checkpoint = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_checkpoint).eval()
processor = CLIPProcessor.from_pretrained(model_checkpoint)

def process_query_and_image(query: str, image_tensor: np.array) -> dict:
    inputs = processor(text=[query], images=[image_tensor], return_tensors="pt")
    with torch.no_grad():
        results = model(**inputs)
    return {
        'text_embeds': results.text_embeds,
        'image_embeds': results.image_embeds
    }
