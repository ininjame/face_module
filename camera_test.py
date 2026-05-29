import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Use MJPG format - more reliable than YUYV over usbipd
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"Camera opened: {cap.isOpened()}")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
