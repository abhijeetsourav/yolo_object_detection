from roboflow import Roboflow


rf = Roboflow(api_key="RCListxHJYWtaULGEBTw")
project = rf.workspace("roboflow-100").project("chess-pieces-mjzgj")
version = project.version(2)
dataset = version.download("yolov8")
