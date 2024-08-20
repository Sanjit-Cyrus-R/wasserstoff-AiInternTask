# Building an AI Pipeline for Image Segmentation and Object Analysis
## Objective
Develop a pipeline using transformers or deep learning models that processes an input image to segment, identify, and analyze objects within the image, and outputs a summary table with mapped data for each object.

## Note
For quickstart of the project refer [AI_pipeline_for_image_ipynb](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/AI_pipeline_for_image.ipynb). More details on this file given after project Explanation

## Process flow
![Refer the image](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Process%20Flow.PNG)

### 1. Image Segmentation Using Masked RCNN
With masked rcnn trained on resnet model we do the image segmenation ie identifying the different objects of the image. After identifying the differnt object from the image, they are stored as separate images(objects) with unique IDs. The python code for Image segmentation is found in [segmentation_model.py](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/models/segmentation_model.py). 
After the segmentation of the image each object of the image is output as a separate image. The object images can be found in [Segmented_objects](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/tree/main/Data/segmented_objects) the metadata containing the master image id and the object image ID are stored in a database. The database is found in [metadata.db](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Data/output/metadata.db)
#### Input to segmentation model
![Refer the image](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Data/Input%20images/image1.jpg)
#### Output to segmentation model
![Refer the image](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Data/output/image1_masked.png)


###  2. Object Identification Using Yolo Model
After segmenting the objects each object image is passed to the Pretrained yolov5 model for identification of object. Since yolov5 can detect independent objects in an image, the master image is passed on to the model to get the masked images with annotations and object identifications. The python code is found in [identification_model.py](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/models/Identification_model.py). 
The output of this mapped to the summary table at the last step. As explained earlier the master image is also passed through yolo model for object identification
#### Object Identification using yolo model
![refer the image](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Data/output/yolo_annotated_image1.jpg)


### 3. Text Detection Using Easy OCR
Using easyocr library the text that is present in each object and as well as the entire image is read and displayed. The output of this mapped to the summary table at the last step. The python code is found in [text_extraction_model.py](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/models/text_extraction_model.py)

### 4. Image Caption Generation Using Blip Models
Using Blip models the caption for each object image and as well as the entire image is read and displayed. The output of this mapped to the summary table at the last step. The python code is found in [summarization_model.py](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/models/summarization_model.py)

### 5. Caption Summarisation Using Summarizer
From all the captions that are received from the objects and images, they all are summarised using transformer models summarizer.  The output of this mapped to the summary table at the last step. The python code is found in [summarization_model.py](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/models/summarization_model.py)

### 6. Data Mapping Of Object, Caption And Text
A json file containing the object_id, caption of the object, text present in each object and summary of the entire image is displayed as a json file. The python code is found in [data_mapping.py](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/utlis/data_mapping.py) and the json file is found in [mapped_data.json](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Data/output/mapped_data.json)


### 7. Preparation of Summary Table
Initially the masked image from the masked rcnn and annotated image from yolov5 model is combined. Next the json file got in the previous step is converted to pandas dataframe and the results are downloaded as csv. The csv file is found in [combined_data_summary.csv](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Data/output/combined_data_summary.csv). Now both the combined image and the dataframe are displayed together. The python code is found in [visualisation.py](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/utlis/visualisation.py)

Here is the combined image
![refer the image](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Data/output/combined_output_image.png)

### 8. User Interface generation
A user interface is displayed using gradio library in python

#### User interface
![refer the image](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Image%20Processing%20Pipeline_page-0001.jpg)
![refer the image](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/Image%20Processing%20Pipeline_page-0002.jpg)


## Google colab 
All this process explained above is coded in a jupyter notebook [AI_pipeline_for_image_ipynb](https://github.com/Sanjit-Cyrus-R/wasserstoff-AiInternTask/blob/main/AI_pipeline_for_image.ipynb). The headings in the notebook are ordered just as in the process flow. To start with google colab just replace the image_path with required image under **combining all functions** module/heading and see the output csv and the combined image displayed. All cells have to be run. 
