import easyocr
from PIL import Image
import os

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # You can specify more languages if needed

def extract_text_easyocr(image_path):
    # Open the image
    image = Image.open(image_path)

    # Extract text using EasyOCR
    text = reader.readtext(image_path, detail=0)

    # Join the list of text segments into a single string
    extracted_text = ' '.join(text)

    return extracted_text