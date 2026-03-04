from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("models/personInCar/modelAssistant.pt")

# Export the model to ONNX format
model.export(format="onnx", simplify=True, half=True, batch=1, opset=11)  # creates 'yolo26n.onnx'