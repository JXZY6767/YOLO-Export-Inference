from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    # 获取当前脚本所在的文件夹
    root = Path(__file__).resolve().parent

    # 加载原始模型
    model = YOLO(str(root / "models" / "best.pt"))

    # 导出为 TorchScript 格式
    output = model.export(
        format="torchscript",
        imgsz=640,
        batch=1,
        device="cpu",
        half=False,
        optimize=False,
        nms=False,
        end2end=False,
    )

    print("导出文件：", output)