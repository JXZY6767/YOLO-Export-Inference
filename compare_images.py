from pathlib import Path
import cv2


def find_video(folder):
    """在结果文件夹中找到视频。"""
    videos = [
        path for path in folder.iterdir()
        if path.suffix.lower() in {".avi", ".mp4", ".mkv", ".mov"}
    ]
    if len(videos) != 1:
        raise RuntimeError(
            f"{folder} 中找到 {len(videos)} 个视频，预期应为 1 个"
        )
    return videos[0]


def read_frame(cap, frame_number):
    """按从 1 开始的帧号读取画面。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"读取第 {frame_number} 帧失败")
    return frame


def add_title(frame, title):
    """在画面上方加标题，不遮挡原检测结果。"""
    image = cv2.copyMakeBorder(
        frame, 60, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(35, 35, 35),
    )
    cv2.putText(
        image, title, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 2,
    )
    return image


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    output_dir = root / "results" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_video = find_video(root / "results" / "pt")
    ts_video = find_video(root / "results" / "torchscript")

    pt_cap = cv2.VideoCapture(str(pt_video))
    ts_cap = cv2.VideoCapture(str(ts_video))

    try:
        if not pt_cap.isOpened() or not ts_cap.isOpened():
            raise RuntimeError("无法打开结果视频")

        for number in [1, 80, 250]:
            left = read_frame(pt_cap, number)
            right = read_frame(ts_cap, number)

            if left.shape != right.shape:
                raise RuntimeError("两份结果视频的画面尺寸不同")

            left = add_title(left, f"PyTorch (.pt) | Frame {number}")
            right = add_title(
                right, f"TorchScript | Frame {number}"
            )
            comparison = cv2.hconcat([left, right])

            output_path = output_dir / f"frame_{number:03d}.jpg"

            # 编码后写入，兼容包含中文的文件路径
            ok, encoded = cv2.imencode(
                ".jpg", comparison,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if not ok:
                raise RuntimeError("图片编码失败")

            encoded.tofile(str(output_path))
            print("已保存：", output_path)

    finally:
        pt_cap.release()
        ts_cap.release()