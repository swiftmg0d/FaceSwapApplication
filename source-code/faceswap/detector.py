import cv2
import dlib
import os

model_path = os.path.abspath(
    os.path.expanduser(os.path.expandvars("assets/shape_predictor_68_face_landmarks.dat")))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

img = cv2.imread('img/1.1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray)

for face in faces:


    landmarks = predictor(gray, face)

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 15, (255, 0, 0), -1)

resized_img = cv2.resize(img, (600, 600))

cv2.imshow('Output', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()