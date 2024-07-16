import time
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, x_test, y_train, y_test

def train_model(model, x_train, y_train):
    start_time = time.time()
    with tqdm(total=1, desc="Training", unit="batch") as pbar:
        model.fit(x_train, y_train)
        pbar.update(1)
    end_time = time.time()
    return end_time - start_time

def evaluate_model(model, x_test, y_test):
    start_time = time.time()
    with tqdm(total=1, desc="Evaluating", unit="batch") as pbar:
        y_pred = model.predict(x_test)
        pbar.update(1)
    end_time = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, end_time - start_time

x_train, x_test, y_train, y_test = load_mnist()

x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)

y_train = y_train.flatten()
y_test = y_test.flatten()

unique_classes = np.unique(y_train)
if len(unique_classes) < 2:
    raise ValueError(f"The dataset contains only {len(unique_classes)} class. It should contain at least 2 classes.")

model = LogisticRegression(max_iter=200)

start_time = time.time()
training_duration = train_model(model, x_train, y_train)
eval_accuracy, evaluation_duration = evaluate_model(model, x_test, y_test)
total_time = time.time() - start_time

print(f'Training accuracy: {model.score(x_train, y_train):.4f}')
print(f'Training duration: {training_duration:.2f} seconds')
print(f'Evaluation accuracy: {eval_accuracy:.4f}')
print(f'Evaluation duration: {evaluation_duration:.2f} seconds')
print(f'Total execution time: {total_time:.2f} seconds')


# Save the trained model using TensorFlow's SavedModel format
saved_model_path = './saved_model/'
tf.saved_model.save(model, saved_model_path)
print(f'Model saved to: {saved_model_path}')