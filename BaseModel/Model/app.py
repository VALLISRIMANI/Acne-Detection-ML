import os 

# Test Commit 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO


app = Flask(__name__)

# Custom deserialization to handle batch_shape -> batch_input_shape
def custom_input_layer(*args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    return tf.keras.layers.InputLayer(*args, **kwargs)

# Load model with custom objects
with tf.keras.utils.custom_object_scope({'InputLayer': custom_input_layer}):
    model = tf.keras.models.load_model("./acne_model_ready_for_deployment.keras")

class_names = ["cleanskin", "mild", "moderate", "severe", "unknown"]
IMG_SIZE = 224

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def apply_prediction_rules(probs, class_names):
    """
    Apply rules to avoid mispredictions:
    - If severe confidence < 50%, classify as moderate
    - If moderate confidence < 30%, classify as mild
    """
    idx = np.argmax(probs)
    predicted_class = class_names[idx]
    confidence = probs[idx] * 100
    
    # Get indices for severity levels
    severe_idx = class_names.index("severe")
    moderate_idx = class_names.index("moderate")
    mild_idx = class_names.index("mild")
    
    severe_confidence = probs[severe_idx] * 100
    moderate_confidence = probs[moderate_idx] * 100
    
    # Rule 1: If severe is predicted but confidence < 50%, change to moderate
    # Then check if moderate also needs adjustment
    if predicted_class == "severe" and severe_confidence < 50:
        # Check if moderate confidence is also < 30%
        if moderate_confidence < 30:
            predicted_class = "mild"
            confidence = probs[mild_idx] * 100
            return predicted_class, confidence, True
        else:
            predicted_class = "moderate"
            confidence = probs[moderate_idx] * 100
            return predicted_class, confidence, True
    
    # Rule 2: If moderate is predicted but confidence < 30%, change to mild
    if predicted_class == "moderate" and moderate_confidence < 30:
        predicted_class = "mild"
        confidence = probs[mild_idx] * 100
        return predicted_class, confidence, True
    
    return predicted_class, confidence, False

HTML = """
<!doctype html>
<html>
<head>
<title>Acne Detection</title>
<style>
    body {
        font-family: Arial, sans-serif;
        max-width: 800px;
        margin: 50px auto;
        padding: 20px;
        background-color: #f5f5f5;
    }
    .container {
        background: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h2 {
        color: #333;
        text-align: center;
    }
    .upload-section {
        margin: 20px 0;
        text-align: center;
    }
    input[type="file"] {
        padding: 10px;
        margin: 10px 0;
    }
    #predictBtn {
        background-color: #4CAF50;
        color: white;
        padding: 12px 30px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        display: none;
    }
    #predictBtn:hover {
        background-color: #45a049;
    }
    #predictBtn:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }
    #imagePreview {
        max-width: 100%;
        max-height: 400px;
        margin: 20px auto;
        display: none;
        border: 2px solid #ddd;
        border-radius: 5px;
    }
    .preview-container {
        text-align: center;
        margin: 20px 0;
    }
    .result {
        margin-top: 30px;
        padding: 20px;
        background-color: #e8f5e9;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    .result h3 {
        margin-top: 0;
        color: #2e7d32;
    }
    .result p {
        font-size: 18px;
        margin: 10px 0;
    }
    .rule-applied {
        color: #ff6f00;
        font-style: italic;
        font-size: 14px;
    }
    .loading {
        display: none;
        text-align: center;
        margin: 20px 0;
    }
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4CAF50;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
</head>
<body>
<div class="container">
    <h2>🔬 Acne Detection System</h2>
    
    <div class="upload-section">
        <input type="file" id="imageInput" accept="image/*" required>
    </div>
    
    <div class="preview-container">
        <img id="imagePreview" alt="Image Preview">
    </div>
    
    <div style="text-align: center;">
        <button id="predictBtn">Predict Acne Severity</button>
    </div>
    
    <div class="loading" id="loadingDiv">
        <div class="spinner"></div>
        <p>Analyzing image...</p>
    </div>
    
    <div id="resultDiv"></div>
</div>

<script>
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const loadingDiv = document.getElementById('loadingDiv');
    const resultDiv = document.getElementById('resultDiv');
    
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.onloadstart = function() {
                imagePreview.style.display = 'none';
                predictBtn.style.display = 'none';
                resultDiv.innerHTML = '';
            };
            
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            
            // Show predict button only after image is fully loaded
            imagePreview.onload = function() {
                predictBtn.style.display = 'inline-block';
            };
            
            reader.readAsDataURL(file);
        }
    });
    
    predictBtn.addEventListener('click', function() {
        const file = imageInput.files[0];
        if (!file) {
            alert('Please select an image first');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', file);
        
        predictBtn.disabled = true;
        loadingDiv.style.display = 'block';
        resultDiv.innerHTML = '';
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingDiv.style.display = 'none';
            predictBtn.disabled = false;
            
            let ruleNote = '';
            if (data.rule_applied) {
                ruleNote = '<p></p>';
            }
            
            resultDiv.innerHTML = `
                <div class="result">
                    <h3>📊 Analysis Result</h3>
                    <p><b>Prediction:</b> <span style="font-size: 24px; color: #1976d2;">${data.prediction.toUpperCase()}</span></p>
                    <p><b>Confidence:</b> ${data.confidence}</p>
                    ${ruleNote}
                </div>
            `;
        })
        .catch(error => {
            loadingDiv.style.display = 'none';
            predictBtn.disabled = false;
            resultDiv.innerHTML = '<div class="result" style="background-color: #ffebee; border-left-color: #f44336;"><p style="color: #c62828;">Error: ' + error.message + '</p></div>';
        });
    });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image provided"}), 400
        
        image = Image.open(file)
        img = preprocess_image(image)
        
        # Get predictions
        probs = model.predict(img)[0]
        
        # Apply prediction rules
        prediction, confidence, rule_applied = apply_prediction_rules(probs, class_names)
        
        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "rule_applied": rule_applied
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)    