import pickle
import cv2
import numpy as np
import dlib

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)

    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0.0

    cosine_angle = np.dot(vector1, vector2) / (norm_vector1 * norm_vector2)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    try:
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    except ValueError:
        return 0.0

def get_landmarks(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if not faces:
        return None

    landmarks = predictor(gray, faces[0])
    landmarks = [(landmark.x, landmark.y) for landmark in landmarks.parts()]

    return np.array(landmarks)

def extract_features(landmarks):
    line1_length = calculate_distance(landmarks[17], landmarks[27])
    line2_length = calculate_distance(landmarks[1], landmarks[15])
    line3_length = calculate_distance(landmarks[31], landmarks[8])
    line4_length = calculate_distance(landmarks[51], landmarks[57])

    line1_angle = calculate_angle(landmarks[17], landmarks[27], landmarks[27])
    line2_angle = calculate_angle(landmarks[1], landmarks[15], landmarks[1])
    line3_angle = calculate_angle(landmarks[31], landmarks[8], landmarks[31])
    line4_angle = calculate_angle(landmarks[51], landmarks[57], landmarks[51])

    print(line1_length, line2_length, line3_length, line4_length, line1_angle, line2_angle, line3_angle, line4_angle)
    return [line1_length, line2_length, line3_length, line4_length, line1_angle, line2_angle, line3_angle, line4_angle]

def predict_shape(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = get_landmarks(gray)

    if landmarks is not None:
        features = extract_features(landmarks)
        prediction = model.predict([features])[0]
        return prediction
    else:
        return None

# Load the model
with open("./face_shape_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict shape for a new image
predicted_shape = predict_shape("./imgs/random.jpg")
print(f"Predicted face shape: {predicted_shape}")
