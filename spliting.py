import os
import shutil
import random

def split_dataset(images_src, labels_src, dataset_path, split_ratio=0.8):
    images_train_dst = os.path.join(dataset_path, "images", "train")
    images_val_dst = os.path.join(dataset_path, "images", "val")
    labels_train_dst = os.path.join(dataset_path, "labels", "train")
    labels_val_dst = os.path.join(dataset_path, "labels", "val")
    
    for folder in [images_train_dst, images_val_dst, labels_train_dst, labels_val_dst]:
        os.makedirs(folder, exist_ok=True)
    
    images_files = [f for f in os.listdir(images_src) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    random.seed(1)
    random.shuffle(images_files)
    
    split_index = int(len(images_files) * split_ratio)
    train_images = images_files[:split_index]
    val_images = images_files[split_index:]
    
    for image in train_images:
        shutil.copy(os.path.join(images_src, image), os.path.join(images_train_dst, image))
        labels_file = os.path.splitext(image)[0] + ".txt"
        if os.path.exists(os.path.join(labels_src, labels_file)):
            shutil.copy(os.path.join(labels_src, labels_file), os.path.join(labels_train_dst, labels_file))
    
    for image in val_images:
        shutil.copy(os.path.join(images_src, image), os.path.join(images_val_dst, image))
        labels_file = os.path.splitext(image)[0] + ".txt"
        if os.path.exists(os.path.join(labels_src, labels_file)):
            shutil.copy(os.path.join(labels_src, labels_file), os.path.join(labels_val_dst, labels_file))
    
def main():
    images_src_no = "./data_resize/no"
    labels_src_no = "./data_label/no"
    images_src_yes = "./data_resize/yes"
    labels_src_yes = "./data_label/yes"
    dataset_path = "./dataset"
    
    split_dataset(images_src_no, labels_src_no, dataset_path)
    split_dataset(images_src_yes, labels_src_yes, dataset_path)

if __name__ == "__main__":
    main()