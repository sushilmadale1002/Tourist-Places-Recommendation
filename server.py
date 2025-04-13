import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import gc  # For memory management

app = Flask(__name__)

# Initialize variables that will be loaded lazily
fe = None
features = []
img_paths = []

def load_resources():
    """Load ML resources only when needed to save memory"""
    global fe, features, img_paths
    
    if fe is None:
        # Initialize feature extractor only once
        fe = FeatureExtractor()
        
        # Load features and image paths
        features = []
        img_paths = []
        for feature_path in Path("./static/feature").glob("*.npy"):
            features.append(np.load(feature_path))
            img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
        features = np.array(features)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def recommend():
    load_resources()  # Ensure resources are loaded
    
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:30]
        recommendations = [(img_paths[id].stem, img_paths[id]) for id in ids]

        # Clean up memory
        del query, dists, ids
        gc.collect()

        return render_template('index.html',
                           query_path=uploaded_img_path,
                           recommendations=recommendations)
    
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # For Render compatibility
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
