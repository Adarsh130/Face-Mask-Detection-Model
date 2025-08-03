from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model("face_mask_model.h5")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASS_NAMES = ["With Mask", "Without Mask"]

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((100, 100))  # same as training
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            image_url = image_path
            img_processed = preprocess_image(image_path)
            pred = model.predict(img_processed)[0]
            predicted_class = CLASS_NAMES[np.argmax(pred)]
            confidence = round(np.max(pred) * 100, 2)
            prediction = f"{predicted_class} ({confidence}%)"
    
    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

