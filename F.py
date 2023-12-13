import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import cv2
from os.path import exists

class ImageClassifier(nn.Module):
    def __init__(self, model_path):
        super(ImageClassifier, self).__init__()
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)

    def detect_objects(self, image_path, sensitivity):
        data = self.preprocess_image(image_path)
        with torch.no_grad():
            prediction = self.model(data)

        objects = self.post_process(prediction, sensitivity)
        return objects

    def post_process(self, prediction, sensitivity):
        # Implement post-processing logic based on your model's output
        # For example, you can use non-maximum suppression to filter out redundant detections
        # You might need to adapt this based on your specific model and its output format.
        # The returned value should be a list of (x, y, width, height) tuples representing detected objects.
        pass

class ObjectDetectionApp(tk.Tk):
    def __init__(self, image_classifier):
        super().__init__()

        self.image_classifier = image_classifier

        self.title("Object Detection App")
        self.geometry("800x600")

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        self.operations_label = tk.Label(self, text="Choose an operation:")
        self.operations_label.pack()

        self.operations_var = tk.StringVar()
        self.operations_var.set("1")  # Default to the Detect Objects operation
        self.operations_menu = tk.OptionMenu(self, self.operations_var, "1")
        self.operations_menu.pack(pady=10)

        self.execute_button = tk.Button(self, text="Execute Operation", command=self.execute_operation)
        self.execute_button.pack(pady=10)

        self.quit_button = tk.Button(self, text="Quit", command=self.destroy)
        self.quit_button.pack(pady=10)

    def execute_operation(self):
        operation_choice = self.operations_var.get()

        if operation_choice == "1":
            self.detect_objects_operation()

    def detect_objects_operation(self):
        image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if not exists(image_path):
            messagebox.showinfo("Error", "Image doesn't exist.")
            return

        sensitivity = simpledialog.askstring("Sensitivity", "Enter Sensitivity:")
        try:
            detected_objects = self.image_classifier.detect_objects(image_path, sensitivity)
            self.display_image_with_objects(image_path, detected_objects)
        except Exception as e:
            messagebox.showinfo("Error", f"Error detecting objects: {str(e)}")

    def display_image_with_objects(self, image_path, objects):
        img = cv2.imread(image_path)
        for (x, y, width, height) in objects:
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

def main():
    model_path = 'your_model.pth'  # Change to the actual path of your PyTorch model file
    if not exists(model_path):
        print("Model file not found.")
        return

    image_classifier = ImageClassifier(model_path)
    app = ObjectDetectionApp(image_classifier)
    app.mainloop()

if __name__ == "__main__":
    main()
