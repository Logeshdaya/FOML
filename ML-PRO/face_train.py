import os
import cv2
import pickle
import face_recognition

dataset_path = 'dataset/faces'
knownEncodings = []
knownNames = []

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(person)

data = {"encodings": knownEncodings, "names": knownNames}

os.makedirs('trained_models', exist_ok=True)
with open("trained_models/face_encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Face Model Trained Successfully")
