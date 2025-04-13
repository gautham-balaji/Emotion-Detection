# 🎭 Real-Time Emotion Detection using OpenCV, TensorFlow & Flask

This project is a real-time emotion recognition system that uses a Convolutional Neural Network (CNN) to detect human facial expressions from live webcam input. It's seamlessly integrated with a Flask API to provide a simple, user-friendly web interface with camera access.

## 🚀 Features

- 🎥 Real-time face detection using OpenCV  
- 🧠 Emotion prediction using a trained CNN model (`.h5`)  
- 🌐 Clean Flask-based web frontend with camera support  
- 📦 Modular code with separate scripts for training, preprocessing, and real-time detection  
- ⚡ Lightweight and responsive interface  

## 😄 Supported Emotions

The model is trained to recognize the following facial expressions:
- Angry 😠  
- Disgust 😖  
- Fear 😨  
- Happy 😄  
- Sad 😢  
- Surprise 😲  
- Neutral 😐  

## 🛠️ Tech Stack

| Component        | Tech Used                  |
|------------------|----------------------------|
| ML Framework     | TensorFlow / Keras         |
| Face Detection   | OpenCV (Haar Cascades)     |
| Frontend         | HTML + CSS + JS (via Flask)|
| Backend API      | Python (Flask)             |
| Model Format     | `.h5` (Keras saved model)  |

## 📂 Folder Structure
```
OPEN_CV/
│
├── .venv/                          # Virtual environment
├── emotion recognition/
│   ├── __pycache__/               # Compiled Python cache
│   ├── static/                    # Static files (images, CSS)
│   │   └── placeholder.png
│   ├── templates/                 # HTML templates for Flask
│   │   └── index.html
│   ├── app.py                     # Flask web server
│   ├── fer2013.csv                # Dataset used for training
│   ├── haarcascade_eye.xml
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_smile.xml
│   ├── load_data.py               # Data loading and preprocessing
│   ├── madhav.jpg                 # Test image (optional)
│   ├── preprocess.py              # Image preprocessing functions
│   ├── real_time_detection.py     # Real-time webcam emotion detector
│   ├── train_model.py             # CNN model architecture and training script
│
├── myenv/                         # Alternate virtual environment (if any)
├── emotion_model.h5               # Trained Keras model
├── requirements.txt               # Dependencies
├── .gitignore
├── .gitattributes
```

## 🧪 How It Works

1. **Face Detection**: OpenCV identifies faces using Haar Cascades.  
2. **Preprocessing**: Extracted face regions are converted to grayscale, resized, and normalized.  
3. **Prediction**: The processed face image is fed into the trained CNN model to classify the emotion.  
4. **Web Integration**: The Flask server handles video input and overlays predicted emotion labels on the frontend.  

## 🔧 Installation

```bash
git clone https://github.com/gautham-balaji/emotion-detection.git
cd emotion-detection
python -m venv venv
venv\Scripts\activate       # On Windows
pip install -r requirements.txt
```

▶️ Run the Project
🖥️ For Real-Time Detection via Python Script:
bash
Copy
Edit
python real_time_detection.py
🌐 For Web Interface:
bash
Copy
Edit
python app.py
Open your browser at http://localhost:5000 to use the live emotion detector with webcam access.
