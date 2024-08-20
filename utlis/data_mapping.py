import os

def generate_mapped_data(image_path, object_ids, captions, text):
    # Create a dictionary to store the mapped data
    data_mapping = {}

    # Map the master image
    master_image_id = os.path.basename(image_path).split('.')[0]
    data_mapping[master_image_id] = {
        'text': text.get(master_image_id, ''),
        'caption': captions.get(master_image_id, ''),
        'combined_summary': combined_summary
    }

    # Map each object
    for object_id in object_ids:
        data_mapping[object_id] = {
            'text': text.get(object_id, ''),
            'caption': captions.get(object_id, ''),
            'combined_summary': combined_summary

        }

    return data_mapping