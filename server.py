import os
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template, url_for
from PIL import Image
from feature_extractor import FeatureExtractor

app = Flask(__name__)

# Configuration
app.config.update({
    'UPLOAD_FOLDER': 'static/uploaded',
    'IMG_FOLDER': 'static/img',  # Explicit image folder
    'MAX_RECOMMENDATIONS': 5,
    'MAX_CONTENT_LENGTH': 8 * 1024 * 1024,
    'ALLOWED_EXTENSIONS': {'jpg', 'jpeg', 'png'}
})

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

class ResourceLoader:
    def __init__(self):
        self.fe = None
        self.features = None
        self.img_paths = None
    
    def load(self):
        """Load resources with path validation"""
        if self.fe is None:
            print("Initializing feature extractor...")
            self.fe = FeatureExtractor()
            
            print("Loading features and validating images...")
            features, img_paths = [], []
            for feature_path in sorted(Path("./static/feature").glob("*.npy")):
                img_path = Path(app.config['IMG_FOLDER']) / (feature_path.stem + ".jpg")
                
                if not img_path.exists():
                    print(f"Warning: Missing image {img_path}")
                    continue
                    
                try:
                    features.append(np.load(feature_path))
                    img_paths.append(img_path)
                    print(f"Loaded: {img_path.name}")
                except Exception as e:
                    print(f"Error loading {feature_path}: {str(e)}")
            
            if not features:
                raise ValueError("No valid features/images found!")
                
            self.features = np.array(features)
            self.img_paths = img_paths
            print(f"Successfully loaded {len(self.features)} valid image-feature pairs")

resource_loader = ResourceLoader()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
    
    if not allowed_file(file.filename):
        return render_template('index.html', error="Only JPG/PNG images allowed")

    try:
        # Save upload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"{timestamp}_{file.filename}"
        upload_path = Path(app.config['UPLOAD_FOLDER']) / upload_filename
        Image.open(file.stream).save(upload_path)

        # Get recommendations
        resource_loader.load()
        query = resource_loader.fe.extract(Image.open(upload_path))
        dists = np.linalg.norm(resource_loader.features - query, axis=1)
        ids = np.argsort(dists)[:app.config['MAX_RECOMMENDATIONS']]
        
        # Prepare results with verified paths
        recommendations = []
        for id in ids:
            img_path = resource_loader.img_paths[id]
            if img_path.exists():
                recommendations.append((
                    img_path.stem,
                    url_for('static', filename=f'img/{img_path.name}')  # Correct URL
                ))
            else:
                print(f"Missing recommendation image: {img_path}")

        if not recommendations:
            raise ValueError("No valid recommendations found")

        return render_template('index.html',
                           query_url=url_for('static', filename=f'uploaded/{upload_filename}'),
                           recommendations=recommendations)

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
