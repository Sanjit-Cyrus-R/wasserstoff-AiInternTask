from models.segmentation_model import *
from models.Identification_model import *

img_dir = 'Data\Input images\image1.jpg'
obj_output_dir = 'Data\segmented_objects'

num_segments, master_id, object_ids, final_image = segment_image(img_dir, obj_output_dir)

store_metadata(master_id, object_ids, db_path='Data\output\segment_metadata.db')

best_labels = {}
    
for file_name in object_ids:
        # Construct the full path to the image
    image_path = os.path.join(obj_output_dir, file_name)
        
        # Call the identify_best_objects function
    best_label = identify_best_objects(image_path)
        
        # Add the result to the dictionary
    best_labels[file_name] = best_label
print(best_labels)