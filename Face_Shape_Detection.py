import cv2
import dlib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import csv


# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to calculate angle between three points
def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)

    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # Check for zero division
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0.0

    cosine_angle = np.dot(vector1, vector2) / (norm_vector1 * norm_vector2)

    # Avoid invalid values in arccosine
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    try:
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    except ValueError:
        return 0.0

# Function to extract facial landmarks

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

# Function to extract features from landmarks
def extract_features(landmarks):
    # Line 1: Between the centers of both eyebrows
    line1_length = calculate_distance(landmarks[17], landmarks[27])

    # Line 2: Between the widest points of the jaw
    line2_length = calculate_distance(landmarks[1], landmarks[15])

    # Line 3: Between the tip of the nose and the center of the chin
    line3_length = calculate_distance(landmarks[31], landmarks[8])
    
    # Line 4: Between the center of the top lip and the center of the bottom lip
    line4_length = calculate_distance(landmarks[51], landmarks[57])

    # Angle of Line 1
    line1_angle = calculate_angle(landmarks[17], landmarks[27], landmarks[27])

    # Angle of Line 2
    line2_angle = calculate_angle(landmarks[1], landmarks[15], landmarks[1])

    # Angle of Line 3
    line3_angle = calculate_angle(landmarks[31], landmarks[8], landmarks[31])

    # Angle of Line 4
    line4_angle = calculate_angle(landmarks[51], landmarks[57], landmarks[51])
    print(line1_length, line2_length, line3_length, line4_length, line1_angle, line2_angle, line3_angle, line4_angle)
    return [line1_length, line2_length, line3_length, line4_length, line1_angle, line2_angle, line3_angle, line4_angle]

# Load dataset and extract features and labels
data = []
labels = []
def write_to_csv(features, label, filename):
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write data row
        row = features + [label]
        csvwriter.writerow(row)

for shape in ["heart", "oblong", "oval", "round", "square"]:
    start_index = {"heart": 1, "oblong": 101, "oval": 201, "round": 301, "square": 401}.get(shape, 1)
    for i in range(start_index, start_index + 99):
        image_path = f"Pre-Classified-Dataset/{shape}/img_no_{i}.jpg"
        print("Index: ",i,"\nShape:",shape,"\nImage Path:",image_path)
        landmarks = get_landmarks(image_path)
        if landmarks is not None:
            features = extract_features(landmarks)
            data.append(features)
            labels.append(shape)
        
            write_to_csv(features, shape, './csv/shape_data.csv')

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# print("Data shape:", data)
# print("Labels shape:", labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

def write_to_csv(features, labels, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write header
        header = ['Line1_length', 'Line2_length', 'Line3_length', 'Line4_length',
                  'Line1_angle', 'Line2_angle', 'Line3_angle', 'Line4_angle', 'Shape']
        csvwriter.writerow(header)

        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Write data rows
        for feature, encoded_label in zip(features, encoded_labels):
            row = list(feature) + [encoded_label]
            csvwriter.writerow(row)
            
# Write training data to CSV
write_to_csv(X_train, y_train, './csv/training_data.csv')

# Write testing data to CSV
write_to_csv(X_test, y_test, './csv/testing_data.csv')

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a k-nearest neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Cross-validate the model
cv_accuracy = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_accuracy) * 100:.2f}%")

# Train the model
knn.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
test_accuracy = knn.score(X_test_scaled, y_test)
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")

# Save the model for future use
model_filename = "face_shape_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(knn, model_file)

# Function to predict shape for a new image
# shapes = ["heart", "oblong", "oval", "round", "square"]

# def predict_shape(image_path):
#     landmarks = get_landmarks(image_path)
#     if landmarks is not None:
#         features = extract_features(landmarks)
#         features_scaled = scaler.transform([features])
#         prediction = knn.predict(features_scaled)[0]
#         return shapes[prediction]
#     else:
#         return None

# # Predict shape for a new image
# new_image_path = "./imgs/random.jpg"
# predicted_shape = predict_shape(new_image_path)
# print("Predicted shape:", predicted_shape)
