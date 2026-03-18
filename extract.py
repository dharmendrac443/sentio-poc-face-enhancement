import os

import cv2
import imagehash
from PIL import Image

video_path = os.environ.get("SENTIO_VIDEO_PATH", "video.mov")
output_dir = os.environ.get("SENTIO_OUTPUT_DIR", "raw_faces")

MAX_FACES = 100
FRAME_SKIP = 15
FACE_CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml",
)


def build_detector():
    detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if detector.empty():
        raise SystemExit(f"Unable to load face detector: {FACE_CASCADE_PATH}")
    return detector


def main():
    os.makedirs(output_dir, exist_ok=True)

    detector = build_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {video_path}")

    hashes = set()
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )

        for x, y, w, h in faces:
            if count >= MAX_FACES:
                break

            if w < 12 or h < 12:
                continue

            factor = 1.5
            new_w = int(w * factor)
            new_h = int(h * factor)
            dx = (new_w - w) // 2
            dy = (new_h - h) // 2
            x1 = max(0, x - dx)
            y1 = max(0, y - dy)
            x2 = min(frame.shape[1], x1 + new_w)
            y2 = min(frame.shape[0], y1 + new_h)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_faces = detector.detectMultiScale(
                crop_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24),
            )
            if len(crop_faces) == 0:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_hash = imagehash.phash(pil_img)

            if img_hash in hashes:
                continue

            hashes.add(img_hash)

            filename = f"face_{count:03d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, crop)
            count += 1

        frame_id += 1

        if count >= MAX_FACES:
            break

    cap.release()
    print("Faces extracted:", count)


if __name__ == "__main__":
    main()
