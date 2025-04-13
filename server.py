import os
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template
from PIL import Image
from feature_extractor import FeatureExtractor

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
app.config['MAX_RECOMMENDATIONS'] = 5  # Reduced from 30 to 5
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB limit

# Create upload directory if needed
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

class ResourceLoader:
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
            for feature_path in sorted(Path("./static/feature").glob("*.npy")):
                features.append(np.load(feature_path))
                img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
            
            self.features = np.array(features)
            self.img_paths = img_paths
            print(f"Loaded {len(self.features)} features")

resource_loader = ResourceLoader()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def recommend():
    if 'query_img' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['query_img']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    try:
        # Process upload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = Path(app.config['UPLOAD_FOLDER']) / f"{timestamp}_{file.filename}"
        img = Image.open(file.stream)
        img.save(upload_path)

        # Get recommendations
        resource_loader.load()
        query = resource_loader.fe.extract(img)
        dists = np.linalg.norm(resource_loader.features - query, axis=1)
        ids = np.argsort(dists)[:app.config['MAX_RECOMMENDATIONS']]
        
        recommendations = [
            (resource_loader.img_paths[id].stem, 
             f"img/{resource_loader.img_paths[id].name}")  # Simplified path
            for id in ids
        ]

        # Cleanup
        del query, dists, ids
        gc.collect()

        return render_template('index.html',
                           query_path=f"uploaded/{upload_path.name}",
                           recommendations=recommendations)

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('index.html', error="Processing failed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
