import cv2
import numpy as np
import sys

# ---------------- Load YOLO ----------------
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
except Exception as e:
    print("Error loading YOLO:")
    print(e)
    sys.exit()

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ---------------- Open Camera ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Trying camera 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    sys.exit()

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    ret, img = cap.read()

    if not ret or img is None:
        print("Failed to capture frame.")
        continue

    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img,
        1 / 255,
        (320, 320),
        (0, 0, 0),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    object_counts = {}

    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]

            label = classes[class_ids[i]]

            object_counts[label] = object_counts.get(label, 0) + 1

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                img,
                label,
                (x, y - 5),
                font,
                0.6,
                (0, 255, 255),
                2
            )

    overlay = img.copy()

    cv2.rectangle(overlay, (10, 10), (260, 420), (0, 0, 0), -1)

    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    cv2.putText(
        img,
        "Object Count",
        (20, 40),
        font,
        0.8,
        (0, 255, 0),
        2
    )

    y = 80

    for label, count in object_counts.items():

        cv2.putText(
            img,
            f"{label}: {count}",
            (20, y),
            font,
            0.6,
            (255, 255, 255),
            2
        )

        y += 30

    cv2.imshow("Object Detection & Counting", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()