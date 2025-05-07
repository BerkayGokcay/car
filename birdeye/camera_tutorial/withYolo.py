from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read()
    if not ret:
        
        break

    results = model.predict(source=frame, show=False, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = box  # bounding box koordinatları

        # bounding box'tan 4 köşe noktası üret
        pts_src = np.float32([
            [x1, y1],  # sol üst
            [x2, y1],  # sağ üst
            [x2, y2],  # sağ alt
            [x1, y2]   # sol alt
        ])

        width, height = 400, 300
        pts_dst = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(frame, matrix, (width, height))

        cv2.imshow("Bird's Eye View", warped)

        # bounding box çiz
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

    cv2.imshow("Original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
