from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    # 加载刚刚导出的模型，指定关键点任务
    model = YOLO(
        str(root / "models" / "best.pt"),
        task="pose",
    )

    # 逐帧检测原视频，并保存带标注的结果
    results = model.predict(
        source=str(root / "data" / "Robot_test.mp4"),
        imgsz=640,
        rect=False,
        conf=0.35,
        iou=0.70,
        max_det=300,
        device="cpu",
        half=False,
        batch=1,
        stream=True,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(root / "results"),
        name="pt",
        exist_ok=True,
        verbose=False,
    )

    frame_count = 0
    detection_count = 0

    # stream=True 返回逐帧结果，需要遍历才能完成整个视频推理
    for result in results:
        frame_count += 1
        detection_count += len(result.boxes)

        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧")

    print(f"推理完成，共处理 {frame_count} 帧")
    print(f"逐帧检测数量之和：{detection_count}")
    print("结果目录：", root / "results" / "pt")