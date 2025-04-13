import gc
from pathlib import Path
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import argparse

def process_image_batch(fe, img_paths, batch_size=10):
    """Process images in batches to reduce memory usage"""
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i + batch_size]
        for img_path in batch:
            try:
                print(f"Processing {img_path.name}")
                with Image.open(img_path) as img:
                    feature = fe.extract(img=img)
                    feature_path = Path("./static/feature") / (img_path.stem + ".npy")
                    np.save(feature_path, feature)
                gc.collect()  # Clean up after each image
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5, 
                      help='Number of images to process at once')
    args = parser.parse_args()

    # Create directories if they don't exist
    Path("./static/feature").mkdir(parents=True, exist_ok=True)
    
    fe = FeatureExtractor()
    img_paths = sorted(Path("./static/img").glob("*.jpg"))
    
    print(f"Starting feature extraction for {len(img_paths)} images")
    process_image_batch(fe, img_paths, args.batch_size)
    print("Feature extraction completed")
