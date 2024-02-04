import numpy as np
import cv2
import dlib
from sklearn.cluster import KMeans
import math
from math import degrees

def load_image(image_path, face_cascade_path, predictor_path):
    image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    return image, original, gray, gauss

def detect_faces(gray, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    return faces

def detect_landmarks(image, faces, predictor):
    landmarks = []
    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        detected_landmarks = predictor(image, dlib_rect).parts()
        landmarks.append(np.array([[p.x, p.y] for p in detected_landmarks]))
    return landmarks

def draw_lines(image, landmarks):
    results = image.copy()

    for face_landmarks in landmarks:
        left_eye_point = face_landmarks[0]
        right_eye_point = face_landmarks[2]
        top_nose_point = face_landmarks[1]
        bottom_nose_point = face_landmarks[4]

        line1 = np.subtract(right_eye_point, left_eye_point)[0]
        cv2.line(results, tuple(left_eye_point), tuple(right_eye_point), color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 1', tuple(left_eye_point), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=2)
        cv2.circle(results, tuple(left_eye_point), 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, tuple(right_eye_point), 5, color=(255, 0, 0), thickness=-1)

        linepointleft = tuple(face_landmarks[1])
        linepointright = tuple(face_landmarks[15])
        line2 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 2', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

        linepointleft = tuple(face_landmarks[3])
        linepointright = tuple(face_landmarks[13])
        line3 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 3', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

        linepointtop = tuple(face_landmarks[8])
        linepointbottom = (face_landmarks[8][0], face_landmarks[0][1])
        line4 = np.subtract(linepointbottom, linepointtop)[1]
        cv2.line(results, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 4', linepointbottom, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointtop, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointbottom, 5, color=(255, 0, 0), thickness=-1)

    return results, line1, line2, line3, line4

def analyze_face_shape(line1, line2, line3, line4, landmarks):
    similarity = np.std([line1, line2, line3])
    oval_similarity = np.std([line2, line4])

    ax, ay = landmarks[0][3]
    bx, by = landmarks[0][4]
    cx, cy = landmarks[0][5]
    dx, dy = landmarks[0][6]

    alpha0 = math.atan2(cy - ay, cx - ax)
    alpha1 = math.atan2(dy - by, dx - bx)
    alpha = alpha1 - alpha0
    angle = abs(degrees(alpha))
    angle = 180 - angle

    if similarity < 10:
        if angle < 160:
            print('Squared shape. Jawlines are more angular')
        else:
            print('Round shape. Jawlines are not that angular')
    elif line3 > line1:
        if angle < 160:
            print('Triangle shape. Forehead is more wider')
    elif oval_similarity < 10:
        print('Diamond shape. Line2 & Line4 are similar, and Line2 is slightly larger')
    elif line4 > line2:
        if angle < 160:
            print('Rectangular. Face length is largest, and jawlines are angular')
        else:
            print('Oblong. Face length is largest, and jawlines are not angular')
    else:
        print("Unable to classify face shape. Contact the developer")

def main():
    image_path = "./imgs/random.jpg"
    face_cascade_path = "./haarcascade_frontalface_default.xml"
    predictor_path = "./shape_predictor_68_face_landmarks.dat"

    image, original, gray, gauss = load_image(image_path, face_cascade_path, predictor_path)

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = detect_faces(gauss, face_cascade)

    predictor = dlib.shape_predictor(predictor_path)
    landmarks = detect_landmarks(image, faces, predictor)

    results, line1, line2, line3, line4 = draw_lines(image, landmarks)

    analyze_face_shape(line1, line2, line3, line4, landmarks)

    output = np.concatenate((original, results), axis=1)
    filename = 'savedImage.jpg'
    cv2.imwrite(filename, output)

if __name__ == "__main__":
    main()
