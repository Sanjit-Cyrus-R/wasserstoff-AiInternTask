import torch
import yolov5
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load pre-trained YOLOv5 model
yolo_model = yolov5.load("yolov5s.pt")

def identify_objects(image_path):
    results = yolo_model(image_path)

    if results.xyxy[0].shape[0] == 0:
        return "", None
    # Extracting the object names and bounding boxes
    objects = results.names
    boxes = results.xyxy[0].cpu().numpy()
    confidences = results.xyxy[0][:, 4].cpu().numpy()
    labels = [objects[int(x[5])] for x in boxes]  # x[5] is the class label index

    max_conf_index = confidences.argmax()
    best_label = labels[max_conf_index]

    return best_label


def detect_objects_yolov5(image_path, output_dir, confidence_threshold=0.5):
    # Load the image using PIL
    image = Image.open(image_path)

    # Perform object detection
    results = yolo_model(image)

    # Filter results by confidence threshold
    results_df = results.pandas().xyxy[0]  # Get results as pandas dataframe
    filtered_results = results_df[results_df['confidence'] >= confidence_threshold]

    # Create the save path with dynamic filename
    file_name = os.path.basename(image_path)
    file_name_without_ext, file_ext = os.path.splitext(file_name)
    save_path = os.path.join(output_dir, f"yolo_annotated_{file_name_without_ext}.jpg")

    # Display the image using Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # Draw bounding boxes and labels
    for _, row in filtered_results.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence, class_name = row['confidence'], row['name']

        # Draw rectangle around object
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='blue', linewidth=2))
        # Add label
        plt.text(x1, y1, f'{class_name} ({confidence:.2f})', color='blue', fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')  # Hide axes

    # Save the image with annotations
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free up memory

    return filtered_results

