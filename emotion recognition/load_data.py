import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load dataset
file_path = r"C:\Users\vsriv\OneDrive\Desktop\OpenCV\emotion recognition\fer2013.csv"  # Adjust if needed
df = pd.read_csv(file_path)

# Extract data
X = np.array([np.fromstring(image, dtype=int, sep=' ').reshape(48, 48, 1) for image in df['pixels']])
y = to_categorical(df['emotion'], num_classes=7)  # 7 emotion classes

# Normalize pixel values
X = X / 255.0  

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display a sample image
plt.imshow(X_train[0].reshape(48, 48), cmap='gray')
plt.title(f"Emotion: {np.argmax(y_train[0])}")
plt.show()
