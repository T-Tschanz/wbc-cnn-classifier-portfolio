
import requests
from PIL import Image
from io import BytesIO

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image
