import os
import numpy as np
from flask import Flask, request, render_template, redirect
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. LOAD LABELS ---
# Based on your ui.py, defaults are cats/dogs if json is missing
CLASS_NAMES = ["cats", "dogs"] 

# --- 2. LOAD MODEL ---
print("Loading TFLite model...")
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image_path):
    # --- PREPROCESSING MATCHING YOUR UI.PY ---
    
    # 1. Resize to 224x224 (As seen in ui.py line 76)
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    
    # 2. Convert to float32
    input_data = np.array(img, dtype=np.float32)
    
    # 3. Add batch dimension -> Shape becomes (1, 224, 224, 3)
    input_data = np.expand_dims(input_data, axis=0)

    # NOTE: We do NOT divide by 255 here because your ui.py says:
    # "Model already has a Rescaling(1./255) layer."

    # --- INFERENCE ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # --- OUTPUT PROCESSING MATCHING YOUR UI.PY ---
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Your ui.py uses argmax (Line 81), implying output shape is [1, 2]
    # e.g., [[0.98, 0.02]] where Index 0=Cat, Index 1=Dog
    pred_idx = int(np.argmax(output_data, axis=1)[0])
    
    # Calculate confidence (Line 83 of ui.py)
    confidence = float(np.max(output_data))
    
    # Get Label
    if pred_idx < len(CLASS_NAMES):
        label = CLASS_NAMES[pred_idx]
    else:
        label = f"Class {pred_idx}"
        
    return f"Prediction: {label.upper()} ({confidence:.1%} confidence)"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_url = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = predict_image(filepath)
            image_url = filepath
            
    return render_template('index.html', result=result, image_url=image_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5200, debug=True)