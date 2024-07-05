from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch


model.tune(data="./chess-pieces-2/data.yaml", epochs=30, iterations=100, optimizer="AdamW", plots=False, save=False, val=False)

# Use the model
# model.train(data="./chess-pieces-2/data.yaml", epochs=3)  # train the model

# model.val(data="data.yaml")