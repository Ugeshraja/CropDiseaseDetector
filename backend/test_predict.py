import tensorflow as tf
import numpy as np
from PIL import Image

# ====================================
# LOAD MODEL
# ====================================

print("Loading Model...")

model = tf.keras.models.load_model(
    "trained_model.keras",
    compile=False
)

print("Model Loaded!")

# ====================================
# CLASS NAMES
# ====================================

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ====================================
# IMAGE PATH
# ====================================

image_path = "test.jpg"

# ====================================
# LOAD IMAGE
# ====================================

image = Image.open(image_path)

image = image.convert("RGB")

image = image.resize((128, 128))

img_array = np.array(image)

img_array = img_array / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)

# ====================================
# PREDICT
# ====================================

prediction = model.predict(img_array)

print("\n========== ALL CLASS PROBABILITIES ==========\n")

for i, p in enumerate(prediction[0]):
    print(
        f"{i}. {class_names[i]} : {float(p)*100:.2f}%"
    )

# ====================================
# FINAL RESULT
# ====================================

index = np.argmax(prediction)

disease = class_names[index]

confidence = float(
    np.max(prediction) * 100
)

print("\n===================================")
print("Predicted Disease :", disease)
print("Confidence        :", round(confidence, 2), "%")
print("===================================\n")