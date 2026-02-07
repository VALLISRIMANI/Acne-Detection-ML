# Acne-Detection-ML

A machine learning based web application that analyzes facial images and classifies acne severity into **mild**, **moderate**, or **severe**. The system uses a Visual Geometry Group (VGG) deep learning model for classification and provides confidence scores along with probability distributions for each class.

---

## 🌐 Live API Endpoint

**Production API:** `https://sivanagu1206-acnedetection.hf.space/predict`

This API is hosted on HuggingFace Spaces and ready to use immediately.

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.10+
- MongoDB (local or cloud)
- Cloudinary account (for image storage)

### 1. Clone the Repository

```bash
git clone https://github.com/VALLISRIMANI/Acne-Detection-ML.git
cd Acne-Detection-ML
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# MongoDB Connection URI
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/acne_ai

# Cloudinary Credentials
CLOUD_NAME=your_cloud_name
API_KEY=your_api_key
API_SECRET=your_api_secret
```

### 4. Run the Application

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000` (or use gunicorn for production):

```bash
gunicorn --workers 1 --threads 2 -b 0.0.0.0:5000 app:app
```

### 5. Run with Docker

```bash
docker build -t acne-detection .
docker run -p 7860:7860 -e MONGODB_URI -e CLOUD_NAME -e API_KEY -e API_SECRET acne-detection
```

---

## 🔧 How the Project Works

### Architecture Overview

```
User Image → Flask API → Preprocessing → VGG Model → Classification → MongoDB + Cloudinary
```

### Step-by-Step Process

1. **Image Upload**: User uploads a facial image via POST request to `/predict`
2. **Validation**: File type is checked (supports: PNG, JPG, JPEG, HEIF, HEIC, AVIF)
3. **Preprocessing**: 
   - Convert image to RGB
   - Resize to 224x224 pixels
   - Normalize pixel values (0-255 → 0-1)
   - Add batch dimension
4. **Model Inference**: VGG model processes the image and outputs probability distribution
5. **Post-processing**: 
   - Get class with highest probability
   - Calculate confidence percentage
   - Store results in MongoDB
   - Upload image to Cloudinary
6. **Response**: Return JSON with prediction, confidence, and probabilities

### Classification Classes

| Class | Description |
|-------|-------------|
| `cleanskin` | No acne detected |
| `mild` | Minor acne breakouts |
| `moderate` | Significant acne present |
| `severe` | Severe acne conditions |
| `unknown` | Unable to classify |

---

## 📡 API Documentation

### Base URL

**Local:** `http://localhost:5000`
**Production:** `https://sivanagu1206-acnedetection.hf.space`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Submit image for prediction |

---

### Predict Endpoint

Submit an image file to receive acne severity classification.

**URL:** `https://sivanagu1206-acnedetection.hf.space/predict`

**Method:** `POST`

**Content-Type:** `multipart/form-data`

#### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | Yes | Image file to analyze |

#### Example Request (cURL)

```bash
curl -X POST "https://sivanagu1206-acnedetection.hf.space/predict" \
  -F "image=@/path/to/your/image.jpg"
```

#### Example Request (Python)

```python
import requests

url = "https://sivanagu1206-acnedetection.hf.space/predict"
files = {"image": open("path/to/your/image.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

#### Example Response

```json
{
  "prediction_id": 123,
  "image_url": "https://res.cloudinary.com/.../acne_dataset/mild/mild_123.jpg",
  "prediction": "mild",
  "confidence": 87.65,
  "probabilities": {
    "cleanskin": 2.1,
    "mild": 87.65,
    "moderate": 8.45,
    "severe": 1.2,
    "unknown": 0.6
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction_id` | Integer | Unique identifier for this prediction |
| `image_url` | String | URL of uploaded image on Cloudinary |
| `prediction` | String | Predicted class label |
| `confidence` | Float | Confidence percentage (0-100) |
| `probabilities` | Object | Probability for each class (0-100) |

#### Error Responses

**400 Bad Request:**
```json
{
  "error": "image file is required"
}
```

**400 Bad Request (Unsupported type):**
```json
{
  "error": "unsupported file type",
  "allowed": ["png", "jpg", "jpeg", "heif", "heic", "avif"]
}
```

**500 Internal Server Error:**
```json
{
  "error": "Error description here"
}
```

---

## 🛠️ Tech Stack

- **Framework:** Flask
- **ML Model:** VGG (Visual Geometry Group) via TensorFlow/Keras
- **Image Processing:** PIL, Pillow-HEIF
- **Database:** MongoDB
- **Image Storage:** Cloudinary
- **Model Hosting:** HuggingFace Hub
- **Deployment:** HuggingFace Spaces, Docker

---

## 📁 Project Structure

```
Acne-Detection-ML/
├── app.py                 # Main Flask application
├── server.py              # HuggingFace Spaces compatible server
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (create this)
└── models/               # Pre-trained Keras models
    └── VisualGeometryGroup.keras
```

---

## 📄 License

MIT License


