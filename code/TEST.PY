import tensorflow as tf

# Create a sample dataset with shape (10, 3)
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24],
    [25, 26, 27],
    [28, 29, 30]
]

# Create a tf.data.Dataset from the data
dataset = tf.data.Dataset.from_tensor_slices(data)

# Print the shape of the dataset (should be empty since it's not a property)
print("Shape of the dataset:", int(dataset.cardinality().numpy()))

print()


