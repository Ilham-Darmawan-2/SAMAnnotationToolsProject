from ultralytics import YOLO

model = YOLO("models/licensePlateSegmentation/modelAssistantSeg.pt")

print(f"Exporting to ONNX....")
model.export(
format="onnx",
opset=11,
dynamic=True,
half=True,
simplify=True,
verbose=False
)


print("Selesai export ONNX.")