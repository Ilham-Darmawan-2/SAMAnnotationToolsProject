from ultralytics import YOLO

def main():
    # pilih model dasar
    model = YOLO("yolo11s.pt")  # n = nano, ringan
    
    model.train(
        data="inference/fireSmoke/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,        # pakai GPU 0, kalau CPU pakai "cpu"
        project="runs",
        name="fire_smoke",
        pretrained=True,
        optimizer="auto",
        patience=20
    )

if __name__ == "__main__":
    main()