def process_image(image):
    # Save the uploaded image to a file
    image_path = '/content/temp_image.jpg'
    image.save(image_path)

    # Step 1: Segment and extract objects
    master_id, object_ids = segment_and_extract_objects(image_path, object_dir, output_dir)

    # Step 2: Store metadata
    store_metadata(master_id, object_ids, metadata_path)

    # Step 3: Identify objects with YOLO
    labels = {obj_id: identify_objects(os.path.join(object_dir, f"{obj_id}.png")) for obj_id in object_ids}

    # Step 4: Detect objects with YOLOv5 and save annotated image
    annotated_image_path = os.path.join(output_dir, f"yolo_annotated_{os.path.basename(image_path)}")
    detect_objects_yolov5(image_path, annotated_image_path)

    # Load the annotated image for display
    annotated_image = Image.open(annotated_image_path)

    # Step 5: Generate captions
    captions = {master_id: generate_image_caption(image_path)}
    captions.update({obj_id: generate_image_caption(os.path.join(object_dir, f"{obj_id}.png")) for obj_id in object_ids})

    # Step 6: Extract text
    text = {master_id: extract_text_easyocr(image_path)}
    text.update({obj_id: extract_text_easyocr(os.path.join(object_dir, f"{obj_id}.png")) for obj_id in object_ids})

    # Step 7: Summarize combined attributes
    combined_summary = summarize_combined_attributes(captions)

    # Step 8: Generate mapped data
    mapped_data = generate_mapped_data(image_path, object_ids, captions, text)

    # Step 9: Save mapped data to JSON
    with open(mapped_data_path, 'w') as json_file:
        json.dump(mapped_data, json_file, indent=4)

    # Load the mapped data for display
    df = pd.DataFrame(mapped_data)

    # Prepare outputs
    output_images = [Image.open(os.path.join(object_dir, f"{obj_id}.png")) for obj_id in object_ids]
    object_details = [
        f"Object ID: {obj_id}\nLabel: {labels[obj_id]}\nCaption: {captions[obj_id]}\nText: {text[obj_id]}"
        for obj_id in object_ids
    ]

    return annotated_image, combined_summary, output_images, object_details, df

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Annotated Image"),
        gr.Textbox(label="Combined Summary"),
        gr.Gallery(label="Extracted Objects"),
        gr.Textbox(label="Object Details", lines=10),
        gr.Dataframe(label="Mapped Data")
    ],
    title="Image Processing Pipeline",
    description="Upload an image to run the object segmentation, detection, and analysis pipeline."
)

# Launch the Gradio app
iface.launch(share=True)
