import requests
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from typing import Optional  # Added import for Optional

# Import local modules
from model import process_query_and_image
from utils import get_random_crop_params, get_cropped_image, update_saliency_map, cosine_similarity

n_iters = 300
min_crop_size = 50

query = "Who developed the Theory of General Relativity?"
image_path = Path("example.jpg")

if not image_path.exists():
    r = requests.get("https://github.com/user-attachments/assets/a5bedef2-e915-4286-bcc9-d599083a99a6")
    with image_path.open("wb") as f:
        f.write(r.content)

image = Image.open(image_path)
im_tensor = np.array(image)
x_dim, y_dim = image.size

# Initial similarity computation
initial_results = process_query_and_image(query, im_tensor)
initial_similarity = cosine_similarity(initial_results['text_embeds'], initial_results['image_embeds']).item()

# Initialize the saliency map
saliency_map = np.zeros((y_dim, x_dim))

for _ in tqdm(range(n_iters)):
    x, y, crop_size = get_random_crop_params(y_dim, x_dim, min_crop_size)
    im_crop = get_cropped_image(im_tensor, x, y, crop_size)

    # Process the cropped image and compute similarity
    crop_results = process_query_and_image(query, im_crop)
    similarity = cosine_similarity(crop_results['text_embeds'], crop_results['image_embeds']).item() - initial_similarity

    # Update the saliency map
    update_saliency_map(saliency_map, similarity, x, y, crop_size)

def plot_saliency_map(image_tensor: np.ndarray, saliency_map: np.ndarray, query: Optional[str]) -> None:
    fig = plt.figure(dpi=150)
    plt.imshow(image_tensor)
    plt.imshow(
        saliency_map,
        norm=colors.TwoSlopeNorm(vcenter=0),
        cmap="jet",
        alpha=0.5,  # make saliency map transparent to see original picture
    )
    if query:
        plt.title(f'Query: "{query}"')
    plt.axis("off")
    return fig

print("Finished")
output_fig = plot_saliency_map(im_tensor, saliency_map, query)
# save fig
output_fig.savefig("saliency_map.png")
