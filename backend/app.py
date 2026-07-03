from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os
import requests

# ====================================
# FLASK APP SETUP
# ====================================
app = Flask(__name__)
CORS(app)

# Limit maximum upload file size to 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ====================================
# LOAD MODELS
# ====================================
import requests

MODEL_URL = "https://huggingface.co/ugeshraja007/crop-disease-model/resolve/main/crop_disease.tflite?download=true"
MODEL_PATH = "crop_disease.tflite"

if not os.path.exists(MODEL_PATH):
    print("Downloading TensorFlow Lite model...")

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)

print("Loading TensorFlow Lite model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TensorFlow Lite model loaded!")

print("Loading Validation Model (MobileNetV2)...")
validator_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=(224, 224, 3)
)
print("Validation Model Loaded Successfully!")

# ====================================
# CONFIGURATION
# ====================================
CONFIDENCE_THRESHOLD = 0.70  # 70% confidence minimum
MODEL_VERSION = "CropNet CNN v1.2"

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
# DETAILED AGRICULTURAL DATABASE
# ====================================
crop_database = {
    "Pepper__bell___Bacterial_spot": {
        "name": "Pepper Bell - Bacterial Spot",
        "status": "diseased",
        "description": "A destructive bacterial disease caused by Xanthomonas campestris, affecting leaves, stems, and fruit.",
        "symptoms": [
            "Small, dark, water-soaked spots on leaves",
            "Leaf yellowing followed by premature defoliation",
            "Raised, blister-like lesions on pepper fruits"
        ],
        "causes": [
            "Contaminated seeds or seedlings",
            "Warm (75-85°F) and wet weather conditions",
            "Splashing rain or overhead irrigation spreading the bacteria"
        ],
        "treatment": "Apply copper-based bactericides at the first sign of disease. Prune and destroy infected leaves.",
        "prevention": "Use certified disease-free seeds. Avoid overhead irrigation. Practice crop rotation with non-solanaceous crops."
    },
    "Pepper__bell___healthy": {
        "name": "Pepper Bell - Healthy",
        "status": "healthy",
        "description": "The pepper plant shows no signs of bacterial, fungal, or viral infections. Leaves are green and robust.",
        "symptoms": [
            "Uniform green leaf color with no spots",
            "Strong, upright stem structure",
            "Normal blossom and fruit setting"
        ],
        "causes": [
            "Optimal watering and drainage",
            "Adequate nitrogen, phosphorus, and potassium levels",
            "Good air circulation around foliage"
        ],
        "treatment": "No disease treatment needed. Maintain current watering and fertilizing schedule.",
        "prevention": "Clean pruning tools regularly. Keep weeds down. Monitor weekly for early pests."
    },
    "Potato___Early_blight": {
        "name": "Potato - Early Blight",
        "status": "diseased",
        "description": "A common fungal disease caused by Alternaria solani. It affects leaves, stems, and tubers, reducing yield.",
        "symptoms": [
            "Dark, circular spots with characteristic concentric rings ('target board' look)",
            "Spots usually start on older, lower leaves",
            "Leaf yellowing and drying up"
        ],
        "causes": [
            "High humidity and frequent leaf wetness",
            "Spore survival in soil and crop debris",
            "Stress due to low soil fertility or insect damage"
        ],
        "treatment": "Apply protectant fungicides (like chlorothalonil or copper). Remove lower senescing leaves to minimize soil contact.",
        "prevention": "Ensure balanced crop nutrition. Plant resistant varieties. Avoid overhead watering. Clean crop residues after harvest."
    },
    "Potato___Late_blight": {
        "name": "Potato - Late Blight",
        "status": "diseased",
        "description": "A devastating oomycete disease caused by Phytophthora infestans. It was the cause of the Irish Potato Famine.",
        "symptoms": [
            "Large, dark green to black water-soaked lesions on leaves",
            "White fuzzy growth on the underside of infected leaves in wet weather",
            "Rapid rot of leaves, stems, and tubers, giving off a foul odor"
        ],
        "causes": [
            "Cool, wet weather (high humidity and temperatures between 60-70°F)",
            "Infected seed tubers",
            "Airborne spores carried by wind from neighboring farms"
        ],
        "treatment": "Apply systemic fungicides (like metalaxyl) immediately. Destroy infected foliage to prevent tuber contamination.",
        "prevention": "Use certified disease-free seed tubers. Keep a wide spacing between plants. Avoid overhead irrigation. Monitor weather alerts."
    },
    "Potato___healthy": {
        "name": "Potato - Healthy",
        "status": "healthy",
        "description": "The potato plant is healthy, showing excellent vegetative growth, clean foliage, and no signs of blight.",
        "symptoms": [
            "Clean, dark green leaves without spots",
            "Firm, green stems",
            "Vigorous canopy development"
        ],
        "causes": [
            "Proper soil pH (5.0-6.0)",
            "Adequate spacing and air movement",
            "Healthy seed stock used during planting"
        ],
        "treatment": "No treatment required. Continue standard cultural practices.",
        "prevention": "Harrow soil to keep tubers covered. Mulch to retain moisture. Spray organic neem oil as a preventive pest check."
    },
    "Tomato_Bacterial_spot": {
        "name": "Tomato - Bacterial Spot",
        "status": "diseased",
        "description": "A serious bacterial disease caused by Xanthomonas species. It leads to leaf spots, defoliation, and fruit lesions.",
        "symptoms": [
            "Small, dark, circular spots on leaves that may appear water-soaked",
            "Leaves yellow and curl, eventually dropping off",
            "Dark, raised scabs or spots on green and ripe tomato fruits"
        ],
        "causes": [
            "Infected seed or transplants",
            "High humidity, heavy rain, and warm temperatures",
            "Spreading through pruning tools, overhead splash, or physical handling"
        ],
        "treatment": "Apply copper fungicides combined with mancozeb. Prune lower branches to keep fruit off the ground.",
        "prevention": "Use certified seeds. Rotate crops out of the tomato family for 2-3 years. Avoid overhead watering."
    },
    "Tomato_Early_blight": {
        "name": "Tomato - Early Blight",
        "status": "diseased",
        "description": "A fungal disease caused by Alternaria solani. It is one of the most common tomato diseases, starting from lower leaves.",
        "symptoms": [
            "Dark spots with concentric target-like rings on older leaves",
            "Yellowing halos around leaf spots",
            "Stem lesions and collar rot on young seedlings"
        ],
        "causes": [
            "Fungal spores overwintering in soil debris",
            "Wet conditions and temperatures between 75-85°F",
            "Spore splashing from soil onto lower foliage"
        ],
        "treatment": "Apply copper or chlorothalonil fungicides. Prune off lower leaves up to 1-2 feet high to avoid soil splash.",
        "prevention": "Mulch the soil around tomato plants. Practice crop rotation. Water at the base of the plant using drip irrigation."
    },
    "Tomato_Late_blight": {
        "name": "Tomato - Late Blight",
        "status": "diseased",
        "description": "An aggressive, fast-spreading oomycete disease caused by Phytophthora infestans that can ruin crops in days.",
        "symptoms": [
            "Large, dark brown to greasy-black patches on leaves and stems",
            "White, velvety fungal-like growth on leaf undersides in humid weather",
            "Large, firm, golden-brown leathery spots on fruit"
        ],
        "causes": [
            "Persistent cool, wet, and humid conditions",
            "Spore transmission by wind or splashing water",
            "Infected volunteer plants or potato tubers nearby"
        ],
        "treatment": "Apply systemic fungicides immediately. Pull up and bag/destroy severely infected plants; do not compost.",
        "prevention": "Plant resistant tomato cultivars. Space plants well for fast drying. Avoid watering late in the evening."
    },
    "Tomato_Leaf_Mold": {
        "name": "Tomato - Leaf Mold",
        "status": "diseased",
        "description": "A fungal disease caused by Passalora fulva, primarily affecting greenhouse-grown tomatoes with poor air circulation.",
        "symptoms": [
            "Pale green or yellow spots on the upper leaf surfaces",
            "Olive-green to gray, velvety spore growth on the corresponding undersides",
            "Leaves curl, wither, and drop prematurely"
        ],
        "causes": [
            "High relative humidity (above 85%)",
            "Warm temperatures inside greenhouses or tunnels",
            "Inadequate spacing and ventilation"
        ],
        "treatment": "Apply fungicides containing copper or chlorothalonil. Remove infected leaves and increase greenhouse heating/ventilation.",
        "prevention": "Ensure proper spacing. Keep greenhouse humidity low. Clean greenhouse structures between seasons."
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato - Septoria Leaf Spot",
        "status": "diseased",
        "description": "A very common fungal disease caused by Septoria lycopersici, causing severe leaf spotting and defoliation.",
        "symptoms": [
            "Numerous small, circular spots with dark brown margins and gray centers",
            "Tiny black specks (fruiting bodies) inside the gray spots",
            "Leaf yellowing, drying, and dropping, exposing fruit to sunscald"
        ],
        "causes": [
            "High humidity, leaf wetness, and warm temperatures",
            "Fungus overwintering on solanaceous weeds and crop debris",
            "Splashing water spreading spores up the plant"
        ],
        "treatment": "Apply copper fungicides or bio-fungicides. Pick off infected leaves immediately. Mulch soil.",
        "prevention": "Rotate crops every 3 years. Water using drip irrigation. Clean stakes and cages with bleach solution."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato - Spider Mites (Two-Spotted)",
        "status": "diseased",
        "description": "An infestation of Tetranychus urticae, tiny arachnids that suck plant sap, leading to weak growth and leaf drop.",
        "symptoms": [
            "Fine white or yellow speckling on the upper surface of leaves",
            "Very fine silk webbing on leaf undersides and stems",
            "Leaves turn bronze or brown, dry up, and fall off"
        ],
        "causes": [
            "Hot, dry, and dusty weather conditions",
            "Over-fertilization with nitrogen, which creates lush, attractive growth",
            "Destruction of natural predators due to broad-spectrum insecticides"
        ],
        "treatment": "Spray with insecticidal soap, neem oil, or specific miticides. Wash plants down with water to blast mites off.",
        "prevention": "Maintain adequate watering (mites prefer dry plants). Encourage predatory mites and ladybugs. Avoid dusty pathways."
    },
    "Tomato_Target_Spot": {
        "name": "Tomato - Target Spot",
        "status": "diseased",
        "description": "A fungal disease caused by Corynespora cassiicola, affecting leaves, stems, and fruit with target-like markings.",
        "symptoms": [
            "Small, dark brown spots on leaves with concentric circles (target appearance)",
            "Leaf spots enlarge and coalesce, causing leaves to drop",
            "Sunken, dark circular spots on green or ripe fruits"
        ],
        "causes": [
            "High humidity and prolonged leaf wetness",
            "Warm temperatures (70-90°F)",
            "Overcrowded plants restricting airflow"
        ],
        "treatment": "Apply fungicides like azoxystrobin or copper. Prune lower foliage to increase air movement.",
        "prevention": "Maintain proper spacing. Rotate crops. Keep foliage dry using drip lines rather than sprinklers."
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "name": "Tomato - Yellow Leaf Curl Virus",
        "status": "diseased",
        "description": "A highly damaging viral disease transmitted by the silverleaf whitefly (Bemisia tabaci). It stunts growth severely.",
        "symptoms": [
            "Severe stunting of new growth and cupping/curling of leaves upwards",
            "Chlorosis (yellowing) of leaf margins and between veins",
            "Failure to set fruit; flowers drop off prematurely"
        ],
        "causes": [
            "Whitefly vectors carrying the virus from nearby weeds or crops",
            "Hot and dry weather which favors whitefly multiplication",
            "No direct chemical cure exists for the virus once a plant is infected"
        ],
        "treatment": "Remove and destroy infected plants immediately to prevent whiteflies from spreading it. Treat whitefly population.",
        "prevention": "Use whitefly-resistant tomato varieties. Apply yellow sticky traps to catch whiteflies. Cover seedlings with insect netting."
    },
    "Tomato_Tomato_mosaic_virus": {
        "name": "Tomato - Tomato Mosaic Virus",
        "status": "diseased",
        "description": "A highly infectious tobacco mosaic virus strain. It is stable and can easily spread through touch, tools, and seeds.",
        "symptoms": [
            "Light and dark green mosaic patterns on leaf surfaces",
            "Leaf distortion, narrowing ('shoestring' effect) or blistering",
            "Internal browning and spotty ripening of fruit"
        ],
        "causes": [
            "Contaminated seeds or infected seedlings",
            "Physical transmission by growers' hands, clothing, tools, or stakes",
            "Survival on surfaces for years; very resilient virus"
        ],
        "treatment": "No chemical treatment is effective. Pull up and incinerate infected plants. Disinfect hands and tools in a dry milk solution.",
        "prevention": "Wash hands with soap and water after handling tobacco. Purchase certified virus-free seeds. Disinfect cages and stakes."
    },
    "Tomato_healthy": {
        "name": "Tomato - Healthy",
        "status": "healthy",
        "description": "The tomato plant is healthy, showing vigorous growth, dark green foliage, abundant blossoms, and healthy green/red fruit.",
        "symptoms": [
            "Vibrant, dark green leaves with no lesions",
            "Robust stem growth and strong leaf petioles",
            "Normal set of flowers and glossy fruit development"
        ],
        "causes": [
            "Perfect soil nutrition and organic compost dressing",
            "Consistent moisture levels and deep root watering",
            "Good sun exposure (6-8 hours of direct light)"
        ],
        "treatment": "No treatment required. Maintain your pruning and staking routine.",
        "prevention": "Prune suckers to maintain balanced airflow. Apply mulch to stabilize soil moisture. Inspect leaves weekly."
    }
}

# ====================================
# HYBRID IMAGE VALIDATION FUNCTION
# ====================================
def validate_leaf_image(image):
    """
    Validates if the image actually contains a plant leaf using MobileNetV2.
    Returns (is_valid, error_message_or_none)
    """
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    # Run validation prediction
    predictions = validator_model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

    plant_keywords = {
        "leaf", "leaves", "plant", "tree", "flower", "vegetable", "fruit", "foliage",
        "potplant", "potted plant", "greenhouse", "buckeye", "acorn", "fig", "banana",
        "apple", "grape", "orange", "lemon", "lime", "strawberry", "pineapple",
        "daisy", "sunflower", "rose", "tulip", "orchid", "cabbage", "broccoli",
        "cucumber", "zucchini", "bell pepper", "pepper", "potato", "tomato", "corn",
        "maize", "rapeseed", "mushroom", "fungus", "bolete", "agaric", "clover",
        "hay", "straw", "log", "woodland", "forest", "grass"
    }

    blacklist_keywords = {
        "dog", "cat", "hound", "terrier", "retriever", "spaniel", "collie", "poodle",
        "puma", "leopard", "lion", "tiger", "cheetah", "bear", "elephant", "zebra",
        "giraffe", "car", "automobile", "truck", "bus", "motorcycle", "bicycle",
        "train", "airplane", "aeroplane", "boat", "ship", "computer", "monitor",
        "screen", "keyboard", "mouse", "laptop", "phone", "television", "tv",
        "desk", "table", "chair", "sofa", "bed", "house", "building", "skyscraper",
        "bridge", "person", "man", "woman", "child", "clothing", "shoe", "shirt",
        "pants", "jacket", "suit", "tie", "dress", "skirt", "sock", "plate", "dish",
        "fork", "spoon", "knife", "cup", "mug", "bottle", "food", "sandwich", "pizza"
    }

    # 1. Blacklist check on the top prediction
    top_label = decoded[0][1].lower().replace("_", " ")
    top_prob = decoded[0][2]
    
    print(f"[Validation] Top predicted ImageNet class: '{top_label}' ({top_prob * 100:.2f}%)")

    for blacklist_word in blacklist_keywords:
        if blacklist_word in top_label:
            if top_prob > 0.15:  # 15% threshold for blacklisted items
                return False, f"This image is recognized as a '{top_label}' ({top_prob * 100:.1f}%), which does not appear to be a plant leaf."

    # 2. Whitelist check on top 5 predictions
    for _, label, prob in decoded:
        label_words = label.lower().replace("_", " ").split()
        for word in label_words:
            if word in plant_keywords or any(kw in word for kw in plant_keywords):
                print(f"[Validation] Match found: '{word}' in class '{label}' ({prob * 100:.2f}%)")
                return True, None

    return False, "This image does not contain a recognized plant leaf structure."

# ====================================
# API ROUTES
# ====================================

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Crop Disease Detection Backend Running",
        "model_version": MODEL_VERSION
    })

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        print("\n========== NEW PREDICTION REQUEST ==========")

        # Check if file uploaded
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Selected file is empty"}), 400

        # Validate file extensions
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file extension. Please upload PNG, JPG, JPEG, WEBP, or BMP."}), 400

        # Load image
        image = Image.open(file)
        image = image.convert("RGB")

        # 1. Image Validation using MobileNetV2
        is_valid, validation_error = validate_leaf_image(image)
        if not is_valid:
            print(f"Validation failed: {validation_error}")
            return jsonify({
                "valid": False,
                "error": validation_error
            }), 200

        # 2. Preprocess for Crop Disease Classifier
        img_array = np.array(image.resize((128, 128)))
        img_array = img_array / 255.0  # Normalize to [0, 1] exactly like training
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Model Prediction
        interpreter.set_tensor(
        input_details[0]["index"],
        img_array.astype(np.float32)
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
        output_details[0]["index"]
        )

        # ADD THESE LINES HERE
        index = np.argmax(prediction[0])
        confidence = float(prediction[0][index])
        disease = class_names[index]

        execution_time = time.time() - start_time
        print(
            f"Inference complete: {disease} "
            f"(Confidence: {confidence * 100:.2f}%) "
            f"in {execution_time:.3f}s"
        )

        
        # 4. Confidence Threshold Check
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Confidence score {confidence * 100:.2f}% below threshold ({CONFIDENCE_THRESHOLD * 100}%)")
            return jsonify({
                "valid": True,
                "disease": "Unknown",
                "confidence": round(confidence * 100, 2),
                "prediction_time": f"{execution_time:.3f}s",
                "model_version": MODEL_VERSION,
                "details": {
                    "name": "Unknown Crop / Disease",
                    "status": "unknown",
                    "description": "The leaf was recognized, but the AI model is not confident enough to diagnose a specific disease.",
                    "symptoms": [
                        "Low confidence score (under 70%)",
                        "Image might be slightly blurry or has bad lighting",
                        "The symptoms may not match typical training samples"
                    ],
                    "causes": [
                        "Suboptimal photo angle or lighting",
                        "Symptoms are in early stages and not clearly visible",
                        "Plant species or disease is not in the trained categories"
                    ],
                    "treatment": "Try uploading a high-resolution close-up photo of the affected area under bright, indirect sunlight.",
                    "prevention": "Keep the leaf centered, flat, and in focus. Crop out excessive background."
                }
            })

        # Fetch detailed advice from database
        details = crop_database.get(disease, {
            "name": disease.replace("___", " ").replace("__", " ").replace("_", " "),
            "status": "diseased" if "healthy" not in disease.lower() else "healthy",
            "description": "No detailed information available in the agricultural database.",
            "symptoms": ["N/A"],
            "causes": ["N/A"],
            "treatment": "Consult a local agricultural extension service.",
            "prevention": "N/A"
        })

        return jsonify({
            "valid": True,
            "disease": disease,
            "confidence": round(confidence * 100, 2),
            "prediction_time": f"{execution_time:.3f}s",
            "model_version": MODEL_VERSION,
            "details": details
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# ====================================
# RUN APP
# ====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )