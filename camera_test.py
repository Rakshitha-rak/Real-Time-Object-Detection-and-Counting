import cv2

for index in [0, 1, 2]:
    print(f"Trying camera {index}")

    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print("Cannot open camera")
        continue

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        cv2.imshow(f"Camera {index}", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

cv2.destroyAllWindows()