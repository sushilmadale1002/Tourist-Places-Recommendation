import os
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
from feature_extractor import FeatureExtractor
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploaded'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

class ResourceLoader:
    """Lazy-loaded resources with memory management"""
    def __init__(self):
        self.fe = None
        self.features = None
        self.img_paths = None
    
    def load(self):
        if self.fe is None:
            print("Initializing feature extractor...")
            self.fe = FeatureExtractor()
            
            print("Loading features...")
            features, img_paths = [], []
            for feature_path in Path("./static/feature").glob("*.npy"):
                features.append(np.load(feature_path))
                img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
            
            self.features = np.array(features)
            self.img_paths = img_paths
            print(f"Loaded {len(self.features)} features")

resource_loader = ResourceLoader()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def recommend():
    resource_loader.load()
    
    if request.method == 'POST':
        if 'query_img' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['query_img']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        try:
            # Save uploaded image
            timestamp = datetime.now().isoformat().replace(":", ".")
            uploaded_img_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{file.filename}")
            img = Image.open(file.stream)
            img.save(uploaded_img_path)

            # Process image
            query = resource_loader.fe.extract(img)
            dists = np.linalg.norm(resource_loader.features - query, axis=1)
            ids = np.argsort(dists)[:30]  # Top 30 results
            recommendations = [
                (resource_loader.img_paths[id].stem, resource_loader.img_paths[id]) 
                for id in ids
            ]

            # Memory cleanup
            del query, dists, ids
            gc.collect()

            return render_template('index.html',
                               query_path=uploaded_img_path,
                               recommendations=recommendations)

        except Exception as e:
            return render_template('index.html', error=f"Error processing image: {str(e)}")
    
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
