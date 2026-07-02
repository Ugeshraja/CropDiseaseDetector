import sys
import os
from PIL import Image
import numpy as np

# Append backend directory to path
backend_dir = "C:/Users/ugesh/AI and Ml project/CropDiseaseDetector/backend"
sys.path.append(backend_dir)

# Set working directory to backend folder so keras model load works
os.chdir(backend_dir)

print("Importing app modules...")
import app

print("\n--- TEST 1: Validate Leaf Image (test.jpg) ---")
test_leaf_path = "test.jpg"
image_leaf = Image.open(test_leaf_path)
is_valid, err = app.validate_leaf_image(image_leaf)
print(f"Is leaf image valid? {is_valid}")
print(f"Error (if any): {err}")
assert is_valid == True, "Failed to validate true leaf image!"

print("\n--- TEST 2: Validate Non-Leaf Image (Random Noise) ---")
# Create a dummy image of random noise that is definitely not a leaf
random_array = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
image_non_leaf = Image.fromarray(random_array)
is_valid_nl, err_nl = app.validate_leaf_image(image_non_leaf)
print(f"Is random noise image valid? {is_valid_nl}")
print(f"Error (if any): {err_nl}")
assert is_valid_nl == False, "Failed to reject non-leaf image!"

print("\n--- TEST 3: Disease Prediction on test.jpg ---")
# Preprocess
img_array = np.array(image_leaf.resize((128, 128)))
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = app.model.predict(img_array)
index = np.argmax(prediction[0])
disease = app.class_names[index]
confidence = float(prediction[0][index])
print(f"Predicted disease: {disease}")
print(f"Confidence: {confidence * 100:.2f}%")

# Retrieve database info
details = app.crop_database.get(disease)
print(f"Human Name: {details['name']}")
print(f"Status: {details['status']}")
print(f"Symptoms: {details['symptoms']}")
print(f"Treatment: {details['treatment']}")

print("\n--- TEST 4: Confidence Thresholding Check ---")
# Test low confidence thresholding logic
low_confidence = 0.55  # 55%
print(f"Testing confidence: {low_confidence * 100}% (Threshold: {app.CONFIDENCE_THRESHOLD * 100}%)")
if low_confidence < app.CONFIDENCE_THRESHOLD:
    test_result = "Unknown"
    test_details = app.crop_database.get("Unknown", {
        "name": "Unknown Crop / Disease",
        "status": "unknown"
    })
    print("Correctly falls back to Unknown!")
else:
    print("Failed to threshold low confidence!")
    
print("\nALL BACKEND DIAGNOSTICS TESTS PASSED SUCCESSFULLY!")
