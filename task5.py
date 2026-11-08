import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "input.mp4"          
OUTPUT_PATH = "output_task5.mp4"  # result 
MODEL_NAME = "yolo11n.pt"
CONF = 0.4                       

LINE_Y_RATIO = 0.5                # line video ki height ke 50% par (0 = top, 1 = bottom)
ZONE_RATIO = [(0.05, 0.55), (0.35, 0.55), (0.35, 0.95), (0.05, 0.95)]
# -----------------------------------------------------------

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit("Video open nahi hui, VIDEO_PATH check karo")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_y = int(h * LINE_Y_RATIO)
zone = np.array([(int(x * w), int(y * h)) for x, y in ZONE_RATIO], dtype=np.int32)

prev_y = {}        # har ID ka pichla y position
entered = set()    # jin IDs ne entry ki
exited = set()     # jin IDs ne exit ki
alert_log = {}     # ID -> kis frame mein zone mein ghusa
frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    results = model.track(frame, persist=True, classes=[0], conf=CONF, verbose=False)
    boxes = results[0].boxes
    alert_now = False

    if boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()

        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if tid in prev_y:
                py = prev_y[tid]
                if py < line_y <= cy:        # upar se neeche
                    entered.add(tid)
                elif py >= line_y > cy:      # neeche se upar
                    exited.add(tid)
            prev_y[tid] = cy

            inside = cv2.pointPolygonTest(zone, (float(cx), float(cy)), False) >= 0
            if inside:
                alert_now = True
                if tid not in alert_log:
                    alert_log[tid] = frame_no

            color = (0, 0, 255) if inside else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.line(frame, (0, line_y), (w, line_y), (255, 0, 0), 2)
    cv2.polylines(frame, [zone], True, (0, 165, 255), 2)

    cv2.putText(frame, f"Entry: {len(entered)}  Exit: {len(exited)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if alert_now:
        cv2.putText(frame, "ALERT: RESTRICTED ZONE!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    writer.write(frame)

cap.release()
writer.release()

print("===== RESULT =====")
print("Entry count :", len(entered))
print("Exit count  :", len(exited))
print("Zone alerts (ID -> frame):", alert_log)
print("Output video saved:", OUTPUT_PATH)