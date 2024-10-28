import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .npz file
data_path = os.path.join(script_dir, 'flatland_train.npz')

# Load the data
data = np.load(data_path)  # Load from the same directory
X = data['X']  # Features (images)
y = data['y']  # Labels

# Preprocess the data
X = X / 255.0  # Scale to [0, 1]
y[y != 0] -= 2  # Adjust labels as specified

# Build a more compact CNN model
model = Sequential([
    Input(shape=(50, 50, 1)),  # Input shape for a single channel
    Conv2D(8, (3, 3), activation='relu'),  # Fewer filters
    MaxPooling2D(pool_size=(2, 2)),  # First pooling layer
    Conv2D(16, (3, 3), activation='relu'),  # Increased filters, but limited
    MaxPooling2D(pool_size=(2, 2)),  # Second pooling layer
    Flatten(),  # Flatten the output
    Dense(16, activation='relu'),  # Fewer units in the dense layer
    Dense(5, activation='softmax')  # Output layer for 5 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32)  # Adjust epochs and batch size as needed

# Construct the output path for the model
model_path = os.path.join(script_dir, 'compact_model.h5')

# Save the model in HDF5 format
model.save(model_path)

# Check the size of the saved model
model_size = os.path.getsize(model_path) / 1024  # Size in KB
print(f'Model size: {model_size:.2f} KB')
