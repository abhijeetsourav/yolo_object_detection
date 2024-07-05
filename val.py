from ultralytics import YOLO

model = YOLO("./last.pt")

validation_results = model.val(data="./data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="cpu")

print(validation_results)