import cv2
import os

cam = cv2.VideoCapture(0)
name = input("Enter your Name : ")

path = 'dataset/' + name
os.makedirs(path, exist_ok=True)

count = 0
while count < 10:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("Capture Face - Press SPACE to capture", frame)
    k = cv2.waitKey(1)
    if k % 256 == 32:
        img_name = f"{path}/{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved!")
        count += 1

cam.release()
cv2.destroyAllWindows()
