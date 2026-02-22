import os
import glob
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_PATH = os.path.join("models", "best_model.keras")
DATA_PATH = os.path.join("data", "clean")
IMG_SIZE = 224

CLASS_NAMES = ["benign", "malignant", "normal"]  # mora da odgovara treningu

# --------------------------------------------------
# LOAD MODEL (lazy load)
# --------------------------------------------------

model = None


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first (python src/train.py)."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict(image: Image.Image):
    global model

    if model is None:
        model = load_model()

    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))

    result_dict = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    label_text = f"{CLASS_NAMES[predicted_idx].upper()} ({probs[predicted_idx] * 100:.2f}%)"

    return result_dict, label_text


# --------------------------------------------------
# LOAD EXAMPLES (if dataset exists locally)
# --------------------------------------------------

def load_examples():
    examples = []

    if not os.path.exists(DATA_PATH):
        return examples

    for cls in CLASS_NAMES:
        class_dir = os.path.join(DATA_PATH, cls)
        if os.path.exists(class_dir):
            images = glob.glob(os.path.join(class_dir, "*.png"))
            examples += images[:2]  # po 2 slike iz svake klase

    return [[img_path] for img_path in examples]


examples = load_examples()

# --------------------------------------------------
# GRADIO UI
# --------------------------------------------------

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload ultrasound image"),
    outputs=[
        gr.Label(label="Class probabilities"),
        gr.Textbox(label="Final prediction")
    ],
    examples=examples if examples else None,
    title="🩺 BUSI Breast Cancer AI",
    description=(
        "EfficientNetB0 + Focal Loss + Fine-Tuning.\n\n"
        "If dataset exists locally in data/clean/, example images will appear below.\n"
        "Educational demo only — not for medical use."
    ),
)

# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch()