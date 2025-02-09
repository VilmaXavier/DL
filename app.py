import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from PIL import Image
import cv2
import os
from tensorflow.keras.datasets import mnist

# Ensure the model file path
MODEL_FILE = "digit_recognition_model.h5"

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape to add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build a simple digit recognition model if not already trained
if not os.path.exists(MODEL_FILE):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test))

    # Save the trained model
    model.save(MODEL_FILE)
else:
    model = load_model(MODEL_FILE)

# Streamlit App
st.title("Handwritten Digit Recognition (MNIST Dataset)")

st.write("Upload an image of a single handwritten digit (0-9) for recognition.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = np.array(image)

    # Invert colors if background is white
    if np.mean(image) > 127:
        image = cv2.bitwise_not(image)

    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Find bounding box of the digit and crop
    coords = cv2.findNonZero(255 - image)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit_roi = image[y:y+h, x:x+w]

        # Pad to maintain aspect ratio
        max_dim = max(w, h)
        top_pad = (max_dim - h) // 2
        bottom_pad = (max_dim - h) - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = (max_dim - w) - left_pad

        padded_image = np.pad(digit_roi, 
                               [(top_pad, bottom_pad), (left_pad, right_pad)], 
                               mode='constant', constant_values=0)

        # Resize to 28x28
        resized_image = cv2.resize(padded_image, (28, 28))

        # Normalize and reshape for prediction
        digit = resized_image / 255.0
        digit = np.expand_dims(digit, axis=(0, -1))

        # Predict the digit
        prediction = model.predict(digit)
        predicted_digit = np.argmax(prediction)

        # Display the uploaded image and prediction
        st.image(resized_image, caption="Uploaded and Processed Image", width=150)
        st.write(f"**Predicted Digit:** {predicted_digit}")

        # Display confidence scores
        st.write("**Confidence Scores:**")
        for i, score in enumerate(prediction[0]):
            st.write(f"Digit {i}: {score:.4f}")
    else:
        st.write("No valid digit detected in the image.")
