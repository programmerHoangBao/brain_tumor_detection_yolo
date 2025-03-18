import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

def select_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

def detect_image(model_path="./best.pt"):
    image_path = select_image()
    
    if not image_path:
        print("No image selected!")
        return

    print(f"Detecting image: {image_path}")
    model = YOLO(model_path)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_path)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

def main():
    detect_image()

if __name__ == "__main__":
    main()  
