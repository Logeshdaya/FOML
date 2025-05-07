import cv2
import pickle
import face_recognition
from datetime import datetime
import csv

# Load face encodings
data = pickle.load(open("trained_models/face_encodings.pickle", "rb"))

# Initialize camera
cam = cv2.VideoCapture(0)
print("[INFO] Scanning Face...")

name = "Unknown"

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        if True in matches:
            matchedIdx = matches.index(True)
            name = data["names"][matchedIdx]
            break

    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)

    if name != "Unknown":
        print(f"[INFO] Face Detected: {name}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# Mark attendance if recognized
if name != "Unknown":
    now = datetime.now()
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S")])
    print(f"[INFO] Attendance Marked for {name}")
else:
    print("[INFO] No known face detected. Attendance not marked.")
