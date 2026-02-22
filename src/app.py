import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = os.path.join("models", "best_model.keras")
CLASS_NAMES = ["benign", "malignant", "normal"]  # mora da se poklapa sa treningom


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first or put the exported model into models/best_model.keras"
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


model = None


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    img = image.convert("RGB").resize((224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}, f"{CLASS_NAMES[idx].upper()} ({probs[idx]*100:.2f}%)"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload ultrasound image"),
    outputs=[gr.Label(label="Probabilities"), gr.Textbox(label="Prediction")],
    title="🩺 BUSI Breast Cancer AI",
    description="EfficientNetB0 + Focal Loss + Fine-Tuning. For learning/demo purposes only.",
)

if __name__ == "__main__":
    demo.launch()