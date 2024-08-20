from PIL import Image, ImageDraw
from transformers import pipeline
from transformers import pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize the caption generation and summarization pipeline
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
summarizer = pipeline("summarization")

def generate_image_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    return caption

def summarize_combined_attributes(object_texts):
    # Combine all object texts into a single text
    combined_text = " ".join(object_texts.values())

    # Generate a summary for the combined text
    summary = summarizer(combined_text, max_length=100, min_length=50, do_sample=False)

    return summary[0]['summary_text']