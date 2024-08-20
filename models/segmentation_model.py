import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3

# Load the pre-trained Mask R-CNN model
mcrnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the pre-trained Mask R-CNN model
mrcnn.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

def segment_and_extract_objects(image_path, output_dir, masked_output_dir):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = mrcnn(image_tensor)

    # Extract the results
    masks = predictions[0]['masks'].detach().cpu().numpy()
    labels = predictions[0]['labels'].detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy()
    boxes = predictions[0]['boxes'].detach().cpu().numpy()

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masked_output_dir, exist_ok=True)

    master_id = os.path.splitext(os.path.basename(image_path))[0]
    object_ids = []

    # Create a blank image for the masks
    mask_image = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Draw masks and bounding boxes on the image
    for i in range(len(masks)):
        if scores[i] > 0.5:  # Filter out low-confidence predictions
            mask = masks[i, 0]
            mask = np.array(mask > 0.5, dtype=np.uint8)  # Binarize mask

            # Create a translucent red mask
            mask_colored = np.zeros((*mask.shape, 4), dtype=np.uint8)
            mask_colored[mask == 1] = [255, 0, 0, 128]  # Red with 50% transparency

            # Convert to PIL Image
            mask_pil = Image.fromarray(mask_colored, 'RGBA')

            # Composite the mask on the original image
            mask_image = Image.alpha_composite(mask_image, mask_pil)

            # Draw bounding box
            box = boxes[i].astype(int)
            draw = ImageDraw.Draw(image)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=2)

            # Extract and save each segmented object
            object_image = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
            mask_3channel = np.stack([mask] * 3, axis=-1)  # Convert mask to RGB format

            # Apply mask to original image
            object_image[mask_3channel != 0] = np.array(image)[mask_3channel != 0]

            # Convert to PIL Image
            object_image_pil = Image.fromarray(object_image)

            # Crop the object using the bounding box
            cropped_object = object_image_pil.crop((box[0], box[1], box[2], box[3]))

            # Generate unique object ID
            object_id = f"{master_id}_{i}"
            object_ids.append(object_id)

            # Save the cropped object
            object_image_path = os.path.join(output_dir, f"{object_id}.png")
            cropped_object.save(object_image_path)

    # Composite the final image with masks over the original
    final_image = Image.alpha_composite(image.convert('RGBA'), mask_image)

    # Save the masked image
    masked_image_path = os.path.join(masked_output_dir, f"{master_id}_masked.png")
    final_image.save(masked_image_path)

    return master_id, object_ids


def store_metadata(master_id, object_ids, db_path='metadata.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Images (
            master_id TEXT PRIMARY KEY
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Objects (
            object_id TEXT PRIMARY KEY,
            master_id TEXT,
            FOREIGN KEY (master_id) REFERENCES Images (master_id)
        )
    ''')

    # Insert master ID
    cursor.execute('INSERT OR IGNORE INTO Images (master_id) VALUES (?)', (master_id,))

    # Insert object metadata
    for object_id in object_ids:
        cursor.execute('INSERT OR IGNORE INTO Objects (object_id, master_id) VALUES (?, ?)', (object_id, master_id))

    # Commit and close the connection
    conn.commit()
    conn.close()

