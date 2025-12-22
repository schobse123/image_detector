import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

# Set the appearance mode of the customtkinter library
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Define the class for the application UI
class AppUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Cat and Dog Image Classifier")
        self.geometry("600x500")

        print("Initializing UI components...")
        # Load model + labels
        self.model = self._load_model()
        self.class_names = self._load_labels() or ["cats", "dogs"]
        self._current_ctk_image = None  # Prevent image GC
        
        # UI Layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Title Label
        self.title_label = ctk.CTkLabel(self, text="Cat and Dog Image Classifier", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=1, columnspan=2, pady=20)

        # Image Display Area (Initially empty)
        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.grid(row=1, column=0, padx=20, pady=10)

        # Result Label
        self.result_label = ctk.CTkLabel(self, text="", font=("Roboto", 20, "bold"))
        self.result_label.grid(row=2, column=0, padx=20, pady=10)

        # Upload Button
        self.upload_btn = ctk.CTkButton(self, text="Select Image", command=self.upload_image, height=50, width=200)
        self.upload_btn.grid(row=3, column=0, padx=20, pady=30)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            print(f"Selected file: {file_path}")
            self.display_image(file_path)
            self.predict_image(file_path)
    
    def display_image(self, path):
        img = Image.open(path)
        aspect_ratio = img.width / img.height
        new_height = 250
        new_width = int(aspect_ratio * new_height)
        my_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_width, new_height))
        self._current_ctk_image = my_img
        self.image_label.configure(image=my_img)

    def predict_image(self, path):
        if self.model is None:
            self.result_label.configure(text="Model not loaded", text_color="red")
            return

        self.result_label.configure(text="Predicting...", text_color="gray")
        self.update()

        # Prediction logic
        try:
            # Model already has a Rescaling(1./255) layer.
            img = Image.open(path).convert("RGB").resize((64, 64))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = self.model.predict(img_array, verbose=0)
            pred_idx = int(np.argmax(prediction, axis=1)[0])
            pred_label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
            confidence = float(np.max(prediction))
            self.result_label.configure(text=f"Prediction: {pred_label} ({confidence:.2%})", text_color="green")
        except Exception as e:
            self.result_label.configure(text="Error in prediction", text_color="red")
            print(f"Prediction error: {e}")

    def _load_model(self):
        model_path = Path(__file__).resolve().parent.parent / "trained_cnn_model.keras"
        try:
            # Prefer native Keras file if present.
            if model_path.exists() and model_path.is_file():
                model = tf.keras.models.load_model(str(model_path))
                print(f"Loaded model (Keras): {model_path}")
                return model

            # Backward-compatible: older training used `model.export(...)` which creates a SavedModel directory.
            if model_path.exists() and model_path.is_dir():
                try:
                    model = tf.keras.models.load_model(str(model_path))
                    print(f"Loaded model (SavedModel via load_model): {model_path}")
                    return model
                except Exception:
                    # Keras 3 cannot load SavedModel directly; wrap it for inference.
                    layer = tf.keras.layers.TFSMLayer(str(model_path), call_endpoint="serve")
                    inputs = tf.keras.Input(shape=(64, 64, 3), dtype=tf.float32)
                    outputs = layer(inputs)
                    model = tf.keras.Model(inputs, outputs)
                    print(f"Loaded model (SavedModel via TFSMLayer): {model_path}")
                    return model

            raise FileNotFoundError(f"Model not found at: {model_path}")
        except Exception as e:
            print(f"Model load error: {e}")
            messagebox.showerror(
                "Model Load Error",
                "Could not load 'trained_cnn_model.keras'. Run training first: python model/train_classifier.py",
            )
            return None

    def _load_labels(self):
        labels_path = Path(__file__).resolve().parent.parent / "trained_cnn_model_labels.json"
        try:
            if not labels_path.exists():
                return None
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
                print(f"Loaded labels: {labels}")
                return labels
            return None
        except Exception as e:
            print(f"Labels load error: {e}")
            return None

# To run the UI:
if __name__ == "__main__":
    app = AppUI()
    app.mainloop()