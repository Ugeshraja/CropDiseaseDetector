import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
interpreter = tf.lite.Interpreter(model_path="crop_disease.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image
img = Image.open("test.jpg").convert("RGB")
img = img.resize((128, 128))

# Convert to numpy
img = np.array(img, dtype=np.float32)

# Normalize
img = img / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Set input
interpreter.set_tensor(input_details[0]['index'], img)

# Run inference
interpreter.invoke()

# Get prediction
output = interpreter.get_tensor(output_details[0]['index'])

# Find the predicted class index
predicted_class = np.argmax(output)

# Load class labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f]

# Print disease name
print("Predicted Class Index:", predicted_class)
print("Disease:", labels[predicted_class])