import cv2
import numpy as np
import keras
from keras.models import load_model
from pygame import mixer

mixer.init()
mixer.music.load('alarm.mp3')

img_processor = keras.applications.ResNet50

face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_alt.xml')
l_eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_lefteye_2splits.xml')
r_eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)

model = load_model('final_model.h5')
print (model.summary())

threshold = 20
counter = 0

while True:
    pred = []
    success, frame = cap.read()
    face = face_cascade.detectMultiScale(frame,1.05, 5)
    if len(face):
        x,y,w,h = face[0]
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)

        l_eye = l_eye_cascade.detectMultiScale(frame,1.05, 5)
        if (len(l_eye)):
            l_eye = frame[l_eye[0][1]:l_eye[0][1]+l_eye[0][3], l_eye[0][0]:l_eye[0][0]+l_eye[0][2]]
            l_eye = cv2.resize(l_eye, (81,81))[:,:,::-1]
            # cv2.imshow("leye", l_eye)
            l_eye = np.expand_dims(l_eye,0)
            pred.append(np.argmax(model.predict(l_eye)))

        r_eye = r_eye_cascade.detectMultiScale(frame, 1.05, 5)
        if (len (r_eye)):
            r_eye = frame[r_eye[0][1]:r_eye[0][1]+r_eye[0][3], r_eye[0][0]:r_eye[0][0]+r_eye[0][2]]
            r_eye = cv2.resize(r_eye, (81,81))[:,:,::-1]
            # cv2.imshow("reye", r_eye)
            r_eye = np.expand_dims(r_eye,0)
            pred.append(np.argmax(model.predict(r_eye)))

        if (pred.count(1) == 0):
            counter = counter + 1
            frame = cv2.putText(frame,f'closed counter:{counter}',(0,100),3, 1,(0,255,0))
            if (counter >= threshold) :
                mixer.music.play()
            else:
                mixer.music.pause()
        else:
            mixer.music.pause()
            counter = 0
            frame = cv2.putText(frame, 'open', (100, 100), 3, 3,(0,255,0))

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)