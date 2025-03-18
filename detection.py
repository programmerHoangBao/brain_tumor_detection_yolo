import cv2
import torch
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

def detect_image(model_path="./best.pt"):
    global panel, label_result
    image_path = select_image()
    
    if not image_path:
        print("No image selected!")
        return

    model = YOLO(model_path)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_path)
    
    detected_labels = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{model.names[cls]}: {conf:.2f}"
            detected_labels.append(label)
            cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    image_pil = Image.fromarray(image_rgb)
    image_pil.thumbnail((500, 500))  # Resize for display
    img_tk = ImageTk.PhotoImage(image_pil)
    
    panel.configure(image=img_tk)
    panel.image = img_tk  # Keep reference to prevent garbage collection
    
    label_result.config(text="\n".join(detected_labels) if detected_labels else "No objects detected.")

def create_gui():
    global panel, label_result
    root = tk.Tk()
    root.title("Traffic Sign Detection")
    root.geometry("600x700")
    
    label = Label(root, text="Traffic Sign Detection with YOLO", font=("Arial", 14, "bold"))
    label.pack(pady=10)
    
    panel = Label(root)
    panel.pack()
    
    label_result = Label(root, text="", font=("Arial", 12), fg="blue")
    label_result.pack(pady=10)
    
    btn_detect = Button(root, text="Select and Detect Image", command=detect_image, font=("Arial", 12))
    btn_detect.pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()
