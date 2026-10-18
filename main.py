import cv2
from ultralytics import YOLO

input_video_path = "input.mp4"
output_video_path = "output.mp4"
min_confidence = 0.5
person_class_id = 0

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture(input_video_path)
if not video.isOpened():
    raise FileNotFoundError(f"Video nahi khuli: {input_video_path}")

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

while True:
    frame_read_ok, frame = video.read()
    if not frame_read_ok:
        break

    results = model(frame, classes=[person_class_id], conf=min_confidence, verbose=False)

    for box in results[0].boxes:
        left, top, right, bottom = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        label_text = f"person {confidence:.2f}"
        cv2.putText(
            frame, label_text, (left, max(top - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    video_writer.write(frame)

video.release()
video_writer.release()
print(f"Done! Output save ho gayi: {output_video_path}")