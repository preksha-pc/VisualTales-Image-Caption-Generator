import numpy as np

# < START >
arr = np.array([[1, 2, 4],
                [7, 13, 21]])
# < END >

print(arr)
print("Shape:", arr.shape)

n_rows = 2
n_columns = 3

# <START>
x = np.random.randn(n_rows, n_columns)
# <END>

print(x)

# < START >
# Initialize an array ZERO_ARR of dimensions (4, 5, 2) whose every element is 0
ZERO_ARR = np.zeros((4, 5, 2))
# < END >

print(ZERO_ARR)

# < START >
# Initialize an array ONE_ARR of dimensions (4, 5, 2) whose every element is 1
ONE_ARR = np.ones((4, 5, 2))
# < END >

print(ONE_ARR)

y = np.array([[1, 2, 3],
              [4, 5, 6]])

# < START >
# Create a new array y_transpose that is the transpose of matrix y
y_transpose = np.transpose(y)
# < END >

print(y_transpose)

# < START >
# Create a new array y_flat that contains the same elements as y but has been flattened to a column array
y_flat = y.flatten()
# < END >

print(y_flat)

# <START>
# Initialize the column matrix here
y = np.array([[4],
              [7],
              [11]])
# <END>

assert y.shape == (3, 1)
# The above line is an assert statement, which halts the program if the given condition evaluates to False.
# Assert statements are frequently used in neural network programs to ensure our matrices are of the right dimensions.

print(y)

# <START>
# Multiply both the arrays here
z = np.dot(arr, y)
# <END>

assert z.shape == (2, 1)

print(z)

x = np.array([4, 1, 5, 6, 11])

# <START>
# Create a new array y with the middle 3 elements of x
y = x[1:4]
# <END>

print(y)

z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# <START>
# Create a new array w with alternating elements of z
w = z[::2]
# <END>

print(w)

arr_2d = np.array([[4, 5, 2],
          [3, 7, 9],
          [1, 4, 5],
          [6, 6, 1]])

# <START>
# Create a 2D array sliced_arr_2d that is of the form [[5, 2], [7, 9], [4, 5]]
sliced_arr_2d = arr_2d[:3, 1:]
# <END>

print(sliced_arr_2d)

arr1 = np.array([1, 2, 3, 4])
b = 1

# <START>
# Implement broadcasting to add b to each element of arr1
arr1 += b
# <END>

print(arr1)

arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
arr3 = np.array([[4],
                 [5]])

# <START>
# Multiply each element of the first row of arr2 by 4 and each element of the second row by 5, using only arr2 and arr3
arr2[0] *= arr3[0]
arr2[1] *= arr3[1]
# <END>

print(arr2)

import time

arr_nonvectorized = np.random.rand(1000, 1000)
arr_vectorized = np.array(arr_nonvectorized) # making a deep copy of the array

start_nv = time.time()

# Non-vectorized approach
# <START>
for i in range(len(arr_nonvectorized)):
    for j in range(len(arr_nonvectorized[0])):
        arr_nonvectorized[i][j] *= 3
# <END>

end_nv = time.time()
print("Time taken in non-vectorized approach:", 1000*(end_nv-start_nv), "ms")

start_v = time.time()

# Vectorization is just performing operations on entire arrays rather than on individual elements one by one...

# Vectorized approach
# <START>
arr_vectorized *= 3
# <END>

end_v = time.time()
print("Time taken in vectorized approach:", 1000*(end_v-start_v), "ms")