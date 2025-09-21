from ultralytics import YOLO
import cv2
import os

# 1️⃣ Load your trained YOLOv8 model
model = YOLO("runs/detect/train4/weights/best.pt")  # change path if different

# 2️⃣ Folder with images to test
image_folder = "dataset/images/train"  # your dataset folder
output_folder = "dataset/output_images"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# 3️⃣ Loop through images
for image_name in os.listdir(image_folder):
    if not (image_name.endswith(".jpg") or image_name.endswith(".png")):
        continue

    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    # 4️⃣ Detect potholes
    results = model(image)
    
    # Check if any potholes detected
    if len(results[0].boxes) > 0:
        print(f"[ALERT] Pothole detected in image: {image_name}")
    else:
        print(f"No potholes in image: {image_name}")

    # 5️⃣ Draw bounding boxes on the image
    image_with_boxes = results[0].plot()

    # 6️⃣ Save the output image
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image_with_boxes)

    # 7️⃣ Display the image (optional)
    # cv2.imshow("Pothole Detection", image_with_boxes)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()
print("Detection completed! Results saved in:", output_folder)
