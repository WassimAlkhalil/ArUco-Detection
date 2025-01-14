import cv2

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():
    success, img = cap.read()

    if not success:
        print("Failed to capture image")
        break

    k = cv2.waitKey(5) & 0xFF
    
    if k == ord('q'): 
        break
    elif k == ord('s'):
        cv2.imwrite(f'images/img{num}.png', img)
        print("Image saved!")
        num += 1

    cv2.imshow('Img', img)

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()