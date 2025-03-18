import cv2
import os

def resize_images(input_folder, output_folder, target_size=(640, 640)):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error when reading photos: {filename}")
            continue
        
        resized_img = cv2.resize(img, target_size)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_img)

        print(f"Processed: {filename}")

    print("Complete resize photos!")

def main():
    input_folder_no = "./data/no"
    output_folder_no = "./data_resize/no"
    input_folder_yes = "./data/yes"
    output_folder_yes = "./data_resize/yes"
    
    resize_images(input_folder=input_folder_no, output_folder=output_folder_no, target_size=(800, 800))
    resize_images(input_folder=input_folder_yes, output_folder=output_folder_yes, target_size=(800, 800))
    
if __name__ == "__main__":
    main()
