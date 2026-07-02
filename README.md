# 🌱 Crop Disease Detection using Deep Learning

An AI-powered web application that detects crop leaf diseases using a Convolutional Neural Network (CNN). The application allows users to upload an image of a plant leaf and instantly predicts the disease along with the confidence score and expert advice.

---

## 📌 Features

- 🌿 Detects 15 different crop leaf diseases
- 📷 Upload leaf images for prediction
- 🤖 Deep Learning (CNN) based disease classification
- 📊 Displays prediction confidence
- 💡 Provides expert advice for detected diseases
- 🎨 Responsive and user-friendly interface
- 🔗 Flask REST API backend

---

## 🧠 Diseases Supported

### Pepper
- Pepper Bell Bacterial Spot
- Pepper Bell Healthy

### Potato
- Potato Early Blight
- Potato Late Blight
- Potato Healthy

### Tomato
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Tomato Healthy

---

## 🛠 Technologies Used

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Flask
- Flask-CORS

### Deep Learning
- TensorFlow
- Keras
- NumPy
- Pillow

---

## 📁 Project Structure

```
CropDiseaseDetector/
│
├── backend/
│   ├── app.py
│   ├── trained_model.keras
│   ├── requirements.txt
│   ├── Procfile
│   ├── runtime.txt
│   └── __pycache__/
│
├── frontend/
│   ├── index.html
│   ├── upload.html
│   ├── style.css
│   ├── upload.css
│   └── script.js
│
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/CropDiseaseDetector.git
```

### 2. Navigate to Backend

```bash
cd CropDiseaseDetector/backend
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Flask Server

```bash
python app.py
```

Backend will start at

```
http://127.0.0.1:5000
```

### 5. Open Frontend

Open

```
frontend/index.html
```

or run it using the VS Code Live Server extension.

---

## 📡 API Endpoint

### Predict Disease

**POST**

```
/predict
```

### Form Data

| Key | Type |
|------|------|
| image | File |

### Sample Response

```json
{
    "disease": "Tomato_Late_blight",
    "confidence": 97.45,
    "advice": "Destroy infected plants and apply systemic fungicide."
}
```

---

## 🖼 How It Works

1. User uploads a leaf image.
2. The frontend sends the image to the Flask backend.
3. The CNN model processes the image.
4. The model predicts the disease.
5. Confidence score is calculated.
6. Expert advice is returned.
7. The frontend displays the result.

---

## 📊 Model Information

- Model Type: Convolutional Neural Network (CNN)
- Input Size: 128 × 128 × 3
- Number of Classes: 15
- Output Layer: Softmax
- Framework: TensorFlow / Keras

---

## 🎯 Future Improvements

- Support additional crop species
- Mobile application
- Real-time camera detection
- Disease severity estimation
- Fertilizer recommendations
- Multi-language support
- Cloud deployment

---

## 👨‍💻 Author

**UGESHRAJA S**

Electronics and Communication Engineering

KSR College of Engineering

---

## 📜 License

This project is developed for educational and academic purposes.