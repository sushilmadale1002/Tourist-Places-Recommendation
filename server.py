import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)

# Serve index.html at the root URL
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload and recommendations
@app.route('/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        recommendations = [(img_paths[id].stem, img_paths[id]) for id in ids]  # Get image names and paths

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               recommendations=recommendations)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run("0.0.0.0")