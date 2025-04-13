from flask import Flask, Response, render_template
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("emotion_model.h5")
face_cascade = cv2.CascadeClassifier(r"C:\Users\vsriv\OneDrive\Desktop\OpenCV\emotion recognition\haarcascade_frontalface_default.xml")
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 255),
    'Sad': (255, 0, 0),
    'Surprise': (255, 255, 0),
    'Neutral': (200, 200, 200)
}


def gen_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    detected_faces = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 3 == 0:
            # Downscale for faster detection
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = face_cascade.detectMultiScale(small_gray, 1.1, 4)
            detected_faces = []

            for (x, y, w, h) in faces:
                # Scale back face location to original frame
                x, y, w, h = [i * 2 for i in (x, y, w, h)]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = np.expand_dims(face_roi, axis=0).astype('float32') / 255.0

                prediction = model.predict(face_roi, verbose=0)
                emotion_index = np.argmax(prediction)
                emotion = emotion_labels[emotion_index]

                detected_faces.append((x, y, w, h, emotion))

        for (x, y, w, h, emotion) in detected_faces:
             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
             cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
