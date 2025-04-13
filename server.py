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
app.config['MAX_RECOMMENDATIONS'] = 5  # Show top 5 matches
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB file limit
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

class ResourceLoader:
    def __init__(self):
        self.fe = None
        self.features = None
        self.img_paths = None
    
    def load(self):
        """Lazy-load resources with shape validation"""
        if self.fe is None:
            print("Initializing MobileNetV2 feature extractor...")
            self.fe = FeatureExtractor()
            
            print("Loading features...")
            features, img_paths = [], []
            for feature_path in sorted(Path("./static/feature").glob("*.npy")):
                feat = np.load(feature_path)
                # Validate feature dimensions
                if feat.shape != (1280,):
                    print(f"Skipping invalid feature: {feature_path} (shape: {feat.shape})")
                    continue
                    
                features.append(feat)
                img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
            
            if not features:
                raise ValueError("No valid features found! Re-run offline.py")
                
            self.features = np.array(features)
            self.img_paths = img_paths
            print(f"Successfully loaded {len(self.features)} features (shape: {self.features.shape})")

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
        return render_template('index.html', error="Invalid file type (only JPG/PNG)")

    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        upload_path = Path(app.config['UPLOAD_FOLDER']) / filename
        img = Image.open(file.stream)
        img.save(upload_path)

        # Process image
        resource_loader.load()
        print("Extracting features from query image...")
        query = resource_loader.fe.extract(img)
        print(f"Query feature shape: {query.shape}")
        
        # Calculate similarities
        dists = np.linalg.norm(resource_loader.features - query, axis=1)
        ids = np.argsort(dists)[:app.config['MAX_RECOMMENDATIONS']]
        
        # Prepare results
        recommendations = [
            (resource_loader.img_paths[id].stem, 
             f"img/{resource_loader.img_paths[id].name}")
            for id in ids
        ]
        print(f"Top recommendations: {recommendations[:2]}...")

        # Cleanup
        del query, dists, ids
        gc.collect()

        return render_template('index.html',
                           query_path=f"uploaded/{filename}",
                           recommendations=recommendations)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return render_template('index.html', error=f"Processing error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
