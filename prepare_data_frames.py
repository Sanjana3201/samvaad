import os
import cv2

# IMPORTANT: Change this path to your "Frames_Word_Level" folder
dataset_path = "C:/Users/Sanjana/OneDrive/Desktop/signdata/ISL_CSLRT_Corpus/Frames_Word_Level"
output_folder = 'prepared_frames'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# We will just copy the images to a new folder for consistency
for word_folder in os.listdir(dataset_path):
    word_path = os.path.join(dataset_path, word_folder)
    if os.path.isdir(word_path):
        output_word_path = os.path.join(output_folder, word_folder)
        if not os.path.exists(output_word_path):
            os.makedirs(output_word_path)
            
        print(f"Preparing data for word: {word_folder}")
        for image_file in os.listdir(word_path):
            image_path = os.path.join(word_path, image_file)
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                # Read and resize image for consistency, then save
                image = cv2.imread(image_path)
                resized_image = cv2.resize(image, (64, 64))
                cv2.imwrite(os.path.join(output_word_path, image_file), resized_image)
                
print("Data preparation complete. Images saved to the 'prepared_frames' folder.")