import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Initialize Flask App
app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'chest_xray_model.h5'
LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
          'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
          'Fibrosis', 'Pleural_Thickening', 'Hernia']

print("Loading model... (This might take a minute)")
# compile=False is CRITICAL because we used a custom loss function during training.
# We don't need the loss function for prediction, only for training.
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded!")

def prepare_image(image, target_size=(224, 224)):
    """Preprocesses the image to match training format"""
    # 1. Ensure RGB (removes alpha channels if png)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. Resize
    image = image.resize(target_size)
    
    # 3. Convert to Array
    image = img_to_array(image)
    
    # 4. Normalize (Divide by 255.0 because we did rescale=1./255 in training)
    image = image / 255.0
    
    # 5. Expand dims (Make it (1, 224, 224, 3))
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/', methods=['GET'])
def home():
    return "<h1>Chest X-Ray Diagnosis API</h1><p>Send a POST request to /predict with an image file.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if image exists in request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # 2. Process Image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image)
        
        # 3. Predict
        preds = model.predict(processed_image)
        
        # 4. Format Results
        # preds is a list of probabilities like [[0.1, 0.9, 0.05...]]
        results = {}
        for i, label in enumerate(LABELS):
            prob = float(preds[0][i])
            results[label] = f"{prob:.2%}" # Format as percentage
            
        # Optional: Return only diseases with > 50% probability
        # diagnoses = {k: v for k, v in results.items() if float(v.strip('%')) > 50}
            
        return jsonify({
            "status": "success",
            "predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the server
    app.run(debug=True, port=5000)