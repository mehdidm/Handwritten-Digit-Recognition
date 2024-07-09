import tensorflow as tf
import numpy as np
import time
import threading
import queue

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape((-1, 28, 28, 1)) / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a function for data augmentation
def augment_data(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

# Define a worker function for data preprocessing
def preprocess_worker(input_queue, output_queue):
    while True:
        try:
            idx = input_queue.get(block=False)
            augmented_image, label = augment_data(x_train[idx], y_train[idx])
            output_queue.put((augmented_image.numpy(), label))
            input_queue.task_done()
        except queue.Empty:
            break

# Create queues for input and output
input_queue = queue.Queue()
output_queue = queue.Queue()

# Fill the input queue with indices
for i in range(len(x_train)):
    input_queue.put(i)

# Create and start worker threads
num_threads = 4
threads = []
for _ in range(num_threads):
    t = threading.Thread(target=preprocess_worker, args=(input_queue, output_queue))
    t.start()
    threads.append(t)

# Wait for all tasks to be completed
input_queue.join()

# Stop worker threads
for t in threads:
    t.join()

# Collect preprocessed data
x_train_augmented = []
y_train_augmented = []
while not output_queue.empty():
    img, label = output_queue.get()
    x_train_augmented.append(img)
    y_train_augmented.append(label)

x_train_augmented = np.array(x_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# Record the start time
start_time = time.time()

# Train the model
model.fit(x_train_augmented, y_train_augmented, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Record the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
print(f'Training duration: {duration:.2f} seconds')