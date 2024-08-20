import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg


def combine_images(yolo_image_path, mrcnn_image_path, output_path):
    # Load images
    yolo_img = mpimg.imread(yolo_image_path)
    mrcnn_img = mpimg.imread(mrcnn_image_path)

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # Display YOLO annotated image
    axs[0].imshow(yolo_img)
    axs[0].axis('off')
    axs[0].set_title('YOLO Annotated Image')

    # Display Mask R-CNN segmented image
    axs[1].imshow(mrcnn_img)
    axs[1].axis('off')
    axs[1].set_title('Mask R-CNN Segmented Image')

    # Save the combined image
    plt.savefig(output_path, bbox_inches='tight')
    #plt.show()
    plt.close(fig)


def create_combined_data_table(mapped_data, file_path):
    # Convert mapped data to a DataFrame
    df = pd.DataFrame.from_dict(mapped_data, orient='index')
    df.index.name = 'image id'
    # Save the DataFrame to a CSV file
    df.to_csv(file_path)

    return df

def display_combined_output(image_path, table_path):
    # Display the combined image
    img = mpimg.imread(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Combined Output Image')
    plt.show()

    # Display the data table
    df = pd.read_csv(table_path)
    print("Combined Data Table:")
    print(df)

