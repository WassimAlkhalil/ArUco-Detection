import cv2
import os

if not os.path.exists("data/captured_images"):
    os.makedirs("data/captured_images")

cap = cv2.VideoCapture(0)
num = 0

while cap.isOpened():
    success, img = cap.read()

    if not success:
        print("Failed to capture image")
        break

    cv2.imshow('Img', img)

    k = cv2.waitKey(5) & 0xFF

    if k == ord('q'):
        print("Program terminated by user.")
        break
    elif k == ord('s'):
        if num < 25:
            filename = f'data/captured_images/image_{num}.png'
            cv2.imwrite(filename, img)
            print(f"Image saved as {filename}")
            num += 1
            if num == 25:
                print("25 images saved. Program terminating.")
                break
        else:
            print("Maximum image limit reached.")

cap.release()
cv2.destroyAllWindows()
