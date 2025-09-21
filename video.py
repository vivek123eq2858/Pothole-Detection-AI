from ultralytics import YOLO
import cv2
import os

# 1️⃣ Load trained YOLOv8 model
model = YOLO("runs/detect/train4/weights/best.pt")  # change path if different

# 2️⃣ Input and output video paths
video_path = "dataset/videos/Dashcam-Front.mp4"       # your input video
output_folder = "dataset/output_videos"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "output.mp4")

# 3️⃣ Open video
cap = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# 4️⃣ Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 5️⃣ Create full screen window
cv2.namedWindow("Pothole Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pothole Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 6️⃣ Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect potholes
    results = model(frame)
    frame_with_boxes = results[0].plot()

    # Check for potholes
    if len(results[0].boxes) > 0:
        print("[ALERT] Pothole detected in current frame")

    # Write frame to output video
    out.write(frame_with_boxes)

    # Display live video in full screen
    cv2.imshow("Pothole Detection", frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
        break

# 7️⃣ Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Detection completed! Output video saved at:", output_path)
