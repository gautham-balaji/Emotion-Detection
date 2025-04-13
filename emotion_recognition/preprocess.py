import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

df = pd.read_csv(r"C:\Users\vsriv\OneDrive\Desktop\OpenCV\emotion recognition\fer2013.csv")

X = np.array([np.fromstring(p, sep=' ').reshape(48, 48, 1) for p in df['pixels']])
X = X / 255.0

y = to_categorical(df['emotion'], num_classes=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],  
    horizontal_flip=True
)

datagen.fit(X_train)
