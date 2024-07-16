import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, IOError):
        return False

def preprocess_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((8, 8))
            img_array = np.array(img)
            if np.mean(img_array) > 128:
                img_array = 255 - img_array
            img_array = img_array.flatten() / 16.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def predict_digit(model, img_array):
    prediction = model.predict_proba([img_array])[0]
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return digit, confidence

# Directory containing your images
image_directory = 'uploads'
current_directory = os.path.dirname(os.path.abspath(__file__))
image_directory = os.path.join(current_directory, image_directory)

# Directory containing saved models
model_directory = 'saved_models'

# Get list of image files
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Get list of model files
model_files = [f for f in os.listdir(model_directory) if f.endswith('.joblib')]

# Load all models
models = []
for model_file in model_files:
    model_path = os.path.join(model_directory, model_file)
    model = joblib.load(model_path)
    models.append(model)

print(f"Loaded {len(models)} models")

# Process each image with each model
results = {}
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)

    if not is_valid_image(image_path):
        print(f"Error: Cannot identify image file {image_path}")
        print(f"Skipping image: {image_file}")
        continue

    img_array = preprocess_image(image_path)
    if img_array is None:
        continue

    results[image_file] = []
    for i, model in enumerate(models):
        digit, confidence = predict_digit(model, img_array)
        results[image_file].append((i+1, digit, confidence))

# Print results
for image_file, predictions in results.items():
    print(f"Image: {image_file}")
    for model_num, digit, confidence in predictions:
        print(f"  Model {model_num}:")
        print(f"    Predicted digit: {digit}")
        print(f"    Confidence: {confidence:.2f}%")
    print("-----------------------------")