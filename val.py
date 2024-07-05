from ultralytics import YOLO

model = YOLO("./best3.pt")

validation_results = model.val(data="./chess-pieces-2/data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="cpu")

print(validation_results)