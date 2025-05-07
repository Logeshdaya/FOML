import cv2
import os
import sounddevice as sd
from scipy.io.wavfile import write

name = input("Enter your name: ")

face_path = f"dataset/faces/{name}"
voice_path = f"dataset/voice/{name}"

os.makedirs(face_path, exist_ok=True)
os.makedirs(voice_path, exist_ok=True)

# Face Capture
cam = cv2.VideoCapture(0)
print("[INFO] Capturing Face Images...")

for i in range(30):
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imwrite(f"{face_path}/face_{i}.jpg", frame)
    cv2.imshow("Capturing Face", frame)
    cv2.waitKey(100)

cam.release()
cv2.destroyAllWindows()



print("[INFO] Dataset Collection Complete!")
