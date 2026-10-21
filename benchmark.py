from pathlib import Path
import csv
import time
import statistics

import cv2
import numpy as np
from ultralytics import YOLO


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    video_path = root / "data" / "Robot_test.mp4"
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)

    # 1. 从原视频中均匀抽取 30 帧，提前读入内存
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 30:
        cap.release()
        raise RuntimeError("视频不足 30 帧，无法按当前方案测试")

    indices = np.linspace(0, total_frames - 1, 30, dtype=int)
    frames = []

    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"无法读取第 {index + 1} 帧")
            frames.append(frame)
    finally:
        cap.release()

    print(f"已读取 {len(frames)} 张测试画面")

    # 2. 加载两种格式的模型
    paths = {
        "pt": root / "models" / "best.pt",
        "torchscript": root / "models" / "best.torchscript",
    }
    models = {
        name: YOLO(str(path), task="pose")
        for name, path in paths.items()
    }

    # 两种模型使用完全相同的推理参数
    options = dict(
        imgsz=640,
        rect=False,
        conf=0.35,
        iou=0.70,
        max_det=300,
        device="cpu",
        half=False,
        batch=1,
        save=False,
        save_txt=False,
        show=False,
        verbose=False,
    )

    # 3. 预热：排除首次加载和初始化对正式计时的影响
    for name, model in models.items():
        print(f"{name}：正在预热 10 次")
        for i in range(10):
            model.predict(source=frames[i], **options)

    # 4. 正式计时：3 轮，每轮 30 帧
    times = {name: [] for name in models}
    measurements = []

    for round_index in range(3):
        # 每轮交替先测哪种模型，减小固定先后顺序的影响
        order = ["pt", "torchscript"]
        if round_index % 2 == 1:
            order.reverse()

        for name in order:
            model = models[name]

            for frame_index, frame in zip(indices, frames):
                start = time.perf_counter()
                result = model.predict(source=frame, **options)[0]
                elapsed_ms = (time.perf_counter() - start) * 1000

                times[name].append(elapsed_ms)
                measurements.append({
                    "format": name,
                    "round": round_index + 1,
                    "frame_number": int(frame_index) + 1,
                    "elapsed_ms": elapsed_ms,
                    "detections": len(result.boxes),
                })

            print(f"第 {round_index + 1} 轮：{name} 测试完成")

    # 5. 汇总模型大小和耗时
    summary = []

    for name, path in paths.items():
        values = times[name]
        row = {
            "format": name,
            "size_bytes": path.stat().st_size,
            "size_MiB": path.stat().st_size / (1024 ** 2),
            "measurements": len(values),
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "std_ms": statistics.stdev(values),
        }
        summary.append(row)

        print(
            f"\n{name}: "
            f"大小 {row['size_MiB']:.2f} MiB，"
            f"平均 {row['mean_ms']:.2f} ms/帧，"
            f"中位数 {row['median_ms']:.2f} ms/帧，"
            f"标准差 {row['std_ms']:.2f} ms"
        )

    # 6. 保存汇总表和每次测量的原始数据
    for filename, rows in [
        ("benchmark_summary.csv", summary),
        ("benchmark_raw.csv", measurements),
    ]:
        with (output_dir / filename).open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\n测速结果已保存到：", output_dir)