from ultralytics import YOLO
import cv2
import os

def get_image_files(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    image_files = []
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(directory, file))
    return image_files

# Example usage
image_directory = './chess-pieces-2/test/images'  # Replace with the path to your directory
images = get_image_files(image_directory)



# Load a pretrained YOLOv8n model
model = YOLO("./best3.pt")

results = model.predict([*images], conf=0.6)  # results list

# Visualize the results
for i, r in enumerate(results):
    b_boxs = [list(j) for j in r.boxes.xywh.cpu().numpy()[:]]

    cls_s = list(r.boxes.cls.cpu().numpy())
    confs = [str(round(flt, 2)) for flt in list(r.boxes.conf.cpu().numpy())]

    for index, rect in enumerate(b_boxs):
      cv2.rectangle(r.orig_img, (int(rect[0])-int(rect[2]/2), int(rect[1])-int(rect[3]/2)), (int(rect[0])+int(rect[2]/2), int(rect[1])+int(rect[3]/2)), (0, 255, 0), 2)
      cv2.putText(r.orig_img, f'{confs[index]}:{r.names[cls_s[index]]}', (int(rect[0])-int(rect[2]/2), int(rect[1])-int(rect[3]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow(f'image{i}', r.orig_img)

# Wait for the 'q' key press to close the window
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all OpenCV windows
cv2.destroyAllWindows()