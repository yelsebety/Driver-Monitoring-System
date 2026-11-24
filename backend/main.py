import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")          # downloads on first run (~6 MB)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW helps on Windows
if not cap.isOpened():
    raise RuntimeError("Could not open the webcam. Try index 1, or close other apps using it.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0, 67, 39, 41], conf=0.5, verbose=False)  # class 0 = person
    annotated = results[0].plot()                                  # draws boxes + confidence

    cv2.imshow("YOLO person detection (press q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()