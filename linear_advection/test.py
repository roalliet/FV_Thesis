import numpy as np
import time

# Create a large 1D numpy array
array_size = 10_000_000  # 10 million elements
original_array = np.random.rand(array_size)

# Number of repetitions for averaging
num_repetitions = 1000

# Function to benchmark a given technique
def benchmark_squaring(method_name, squaring_function, array, repetitions):
    times = []
    for _ in range(repetitions):
        start_time = time.time()
        result = squaring_function(array)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / repetitions
    print(f"{method_name}: Average time over {repetitions} repetitions = {avg_time:.6f} seconds")
    return avg_time

# Define different squaring techniques
def numpy_square(array):
    return np.square(array)

def elementwise_multiplication(array):
    return array * array


def power_operator(array):
    return array ** 2

# Perform benchmarking
print("Benchmarking different squaring techniques:\n")

benchmark_squaring("Numpy np.square", numpy_square, original_array, num_repetitions)
benchmark_squaring("Elementwise multiplication", elementwise_multiplication, original_array, num_repetitions)
benchmark_squaring("Power operator", power_operator, original_array, num_repetitions)