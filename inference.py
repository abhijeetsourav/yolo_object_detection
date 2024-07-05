from ultralytics import YOLO
import cv2
from PIL import Image

# Load a model
model = YOLO("./last.pt")
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model

results = model(source="./cats-and-dogs.jpg")

for i, result in enumerate(results):
    print(result.boxes)
    # im_bgr = result.plot()  # BGR-order numpy array
    # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    result.show()

    # Save results to disk
    result.save(filename=f"results{i}.jpg")

# Wait for a key press indefinitely
# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # Press 'q' to exit
#         break

# # Destroy all windows
# cv2.destroyAllWindows()
