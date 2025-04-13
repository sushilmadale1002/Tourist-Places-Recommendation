import gc
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from feature_extractor import FeatureExtractor

# Configure TensorFlow for low memory usage
tf.config.set_soft_device_placement(True)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.keras.backend.set_floatx('float16')  # Use half-precision

def process_images(fe, img_paths):
    """Process images one at a time with memory cleanup"""
    Path("./static/feature").mkdir(parents=True, exist_ok=True)
    
    for img_path in img_paths:
        try:
            print(f"Processing {img_path.name}")
            with Image.open(img_path) as img:
                feature = fe.extract(img)
                feature_path = Path("./static/feature") / (img_path.stem + ".npy")
                np.save(feature_path, feature.astype('float16'))  # Save space
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
        finally:
            tf.keras.backend.clear_session()
            gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Skip existing features')
    args = parser.parse_args()

    fe = FeatureExtractor()
    all_images = sorted(Path("./static/img").glob("*.jpg"))
    
    if args.resume:
        existing = {f.stem for f in Path("./static/feature").glob("*.npy")}
        todo = [img for img in all_images if img.stem not in existing]
        print(f"Resuming: {len(todo)} images remaining")
    else:
        todo = all_images
        print(f"Starting fresh processing of {len(todo)} images")

    process_images(fe, todo)
    print("Feature extraction completed")
