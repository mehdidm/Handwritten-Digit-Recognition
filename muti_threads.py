import threading
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

MAX_THREADS = 5
MAX_MODELS = 2
SEED = 42


class HandwritingRecognition:
    def __init__(self):
        self.models_available = threading.Semaphore(MAX_MODELS)
        self.thread_lock = threading.Lock()
        self.customers_waiting = []
        self.pbar = tqdm(total=MAX_THREADS, desc="Threads completed")

        digits = load_digits()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=SEED)

    def start_recognition(self):
        threads = []
        start_time = time.perf_counter()
        for i in range(MAX_THREADS):
            thread = threading.Thread(target=self.customer)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        total_time = time.perf_counter() - start_time
        self.pbar.close()
        print(f'Total execution time: {total_time:.4f} seconds')

    def customer(self):
        while True:
            self.thread_lock.acquire()
            if self.models_available.acquire(blocking=False):
                self.thread_lock.release()

                model = KNeighborsClassifier()

                # Run training multiple times and take average
                n_runs = 10
                total_training_time = 0
                for _ in range(n_runs):
                    start_time = time.perf_counter()
                    model.fit(self.X_train, self.y_train)
                    total_training_time += time.perf_counter() - start_time
                avg_training_time = total_training_time / n_runs

                start_time = time.perf_counter()
                predictions = model.predict(self.X_test)
                evaluation_time = time.perf_counter() - start_time

                accuracy = accuracy_score(self.y_test, predictions)
                print(f"Thread {threading.current_thread().name}:")
                print(f"  Accuracy = {accuracy:.4f}")
                print(f"  Average Training time = {avg_training_time:.6f} seconds")
                print(f"  Evaluation time = {evaluation_time:.6f} seconds")

                self.models_available.release()
                self.pbar.update(1)
                break
            else:
                print(f"No available models. Thread {threading.current_thread().name} must wait.")
                self.customers_waiting.append(threading.current_thread())
                self.thread_lock.release()
                time.sleep(np.random.uniform(0.5, 2))


if __name__ == "__main__":
    recognition = HandwritingRecognition()
    recognition.start_recognition()