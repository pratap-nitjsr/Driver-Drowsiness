# Import necessary libraries
import cv2
import numpy as np
import keras
from keras.models import load_model
from pygame import mixer
import streamlit as st

# Use ResNet50 as the image processor
img_processor = keras.applications.ResNet50

# Initialize the mixer for audio alerts and load the alarm sound
mixer.init()
mixer.music.load('alarm.mp3')
mixer.music.play(loops=9999)
mixer.music.pause()

# Load Haar cascades for face and eyes detection
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_alt.xml')
l_eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_lefteye_2splits.xml')
r_eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_righteye_2splits.xml')

# Load the pre-trained model for closed eye detection
model = load_model('final_model.h5')

# Set the threshold for consecutive closed eye detections
threshold = 20
counter = 0


# Function to detect closed eyes
def detect_closed_eyes(frame):
    global counter
    pred = []

    # Detect faces in the frame
    face = face_cascade.detectMultiScale(frame, 1.05, 5)

    if len(face):
        x, y, w, h = face[0]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Detect left eye within the detected face region
        l_eye = l_eye_cascade.detectMultiScale(frame, 1.05, 5)
        if len(l_eye):
            l_eye = frame[l_eye[0][1]:l_eye[0][1] + l_eye[0][3], l_eye[0][0]:l_eye[0][0] + l_eye[0][2]]
            l_eye = cv2.resize(l_eye, (81, 81))[:, :, ::-1]
            l_eye = np.expand_dims(l_eye, 0)
            pred.append(np.argmax(model.predict(l_eye)))

        # Detect right eye within the detected face region
        r_eye = r_eye_cascade.detectMultiScale(frame, 1.05, 5)
        if len(r_eye):
            r_eye = frame[r_eye[0][1]:r_eye[0][1] + r_eye[0][3], r_eye[0][0]:r_eye[0][0] + r_eye[0][2]]
            r_eye = cv2.resize(r_eye, (81, 81))[:, :, ::-1]
            r_eye = np.expand_dims(r_eye, 0)
            pred.append(np.argmax(model.predict(r_eye)))

        # If both eyes are closed, increase the closed eye counter and display a warning
        if pred.count(1) == 0:
            if (counter<threshold):
                counter = counter + 1
            frame = cv2.putText(frame, 'WARNING', (0, 100), 3, 3, (0, 0, 255))
        else:
            counter = max(counter-1,0)
            frame = cv2.putText(frame, 'KEEP GOING', (100, 100), 3, 1, (0, 255, 0))

    return frame, counter


# Streamlit app for closed eye detection
def main():
    st.title("Driver Drowsiness Detection App")

    # Open a video capture object for the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Create an empty list to store frames
    frames = st.image([])

    while True:
        # Read a frame from the video capture
        success, frame = cap.read()
        if not success:
            st.error("Failed to capture video.")
            break

        # Process the frame and add it to the list
        frame, counter = detect_closed_eyes(frame)

        # If the closed eye counter exceeds the threshold, unpause the alarm
        if counter >= threshold:
            mixer.music.unpause()
        else:
            # Stop and pause the alarm
            mixer.music.pause()

        # Display the frame with the processed information
        frames.image(frame, channels="BGR", caption="Drowsiness Detection")

    # Release the video capture object
    cap.release()


# Run the Streamlit app if this script is executed
if __name__ == "__main__":
    main()