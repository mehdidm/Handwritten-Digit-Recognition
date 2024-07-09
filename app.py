import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import matplotlib.pyplot as plt

# Load the saved model
model = tf.saved_model.load('handwritten_digit_model')
predict_fn = model.signatures['serving_default']

# Get the input tensor name
input_name = list(predict_fn.structured_input_signature[1].keys())[0]


def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Verify that it is an image
        return True
    except (UnidentifiedImageError, IOError):
        return False


def preprocess_image(image_path):
    try:
        # Open the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {image_path}")
        return None

    # Resize to 28x28 pixels
    img = img.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(img)

    # Invert colors if necessary (assuming dark digit on light background)
    if np.mean(img_array) > 128:
        img_array = 255 - img_array

    # Normalize and reshape
    img_array = img_array.reshape((1, 28, 28, 1)) / 255.0

    return img_array


def predict_digit(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    if img_array is None:
        return None, None  # Skip if image could not be processed

    # Debugging: Display the preprocessed image
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f"Preprocessed: {os.path.basename(image_path)}")
    plt.show()

    # Make prediction
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    prediction = predict_fn(**{input_name: input_tensor})

    # Get the output tensor (it should be the last item in the prediction dictionary)
    output_key = list(prediction.keys())[-1]
    output_tensor = prediction[output_key]

    # Get the predicted digit and confidence
    digit = tf.argmax(output_tensor, axis=1).numpy()[0]
    confidence = tf.reduce_max(output_tensor).numpy() * 100

    # Debugging: Print raw prediction values
    print(f"Raw prediction: {output_tensor.numpy()}")

    return digit, confidence


# Directory containing your images
image_directory = 'uploads'  # This folder is in the same directory as the script

# Get the full path to the uploads directory
current_directory = os.path.dirname(os.path.abspath(__file__))
image_directory = os.path.join(current_directory, image_directory)

# Get list of image files
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)

    # Validate the image file
    if not is_valid_image(image_path):
        print(f"Error: Cannot identify image file {image_path}")
        print(f"Skipping image: {image_file}")
        continue

    predicted_digit, confidence = predict_digit(image_path)
    if predicted_digit is not None and confidence is not None:
        print(f"Image: {image_file}")
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}%")
        print("-----------------------------")
    else:
        print(f"Skipping image: {image_file}")