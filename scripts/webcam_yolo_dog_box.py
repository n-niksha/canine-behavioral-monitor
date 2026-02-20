import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# --------- MODEL PATH (bulletproof) ---------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dog_best.pt")

# Your custom model almost certainly uses class 0 for dog
DOG_CLASS_ID = 0


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main():
    print("Loading model from:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("MODEL NAMES:", model.names)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing index 0->1.")

    prev_roi_gray = None
    prev_shape = None
    last_print = 0

    DIFF_THRESHOLD = 25
    MOTION_PIXELS = 2500
    PRINT_EVERY_SEC = 1.0

    smooth_box = None
    ALPHA = 0.6

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        results = model.predict(frame, imgsz=640, conf=0.15, verbose=False)[0]

        best_xyxy = None
        best_conf = 0.0

        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item())

                if cls == DOG_CLASS_ID and conf > best_conf:
                    best_xyxy = b.xyxy[0].cpu().numpy()
                    best_conf = conf

        status = "NO_DOG"
        motion_count = 0
        box = None

        if best_xyxy is not None:
            x1, y1, x2, y2 = best_xyxy
            box = clamp_box(x1, y1, x2, y2, w, h)

        if box is not None:
            if smooth_box is None:
                smooth_box = np.array(box, dtype=np.float32)
            else:
                smooth_box = ALPHA * smooth_box + (1 - ALPHA) * np.array(box, dtype=np.float32)

            x1, y1, x2, y2 = smooth_box.astype(int).tolist()
            box = clamp_box(x1, y1, x2, y2, w, h)

        if box is not None:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.GaussianBlur(roi_gray, (21, 21), 0)

            if prev_shape != roi_gray.shape:
                prev_roi_gray = None
                prev_shape = roi_gray.shape

            if prev_roi_gray is None:
                status = "WARMING_UP"
            else:
                diff = cv2.absdiff(prev_roi_gray, roi_gray)
                _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                thresh = cv2.dilate(thresh, None, iterations=2)

                motion_count = int(np.sum(thresh > 0))
                status = "MOVING" if motion_count > MOTION_PIXELS else "STILL"

                cv2.imshow("roi_motion_mask", thresh)

            prev_roi_gray = roi_gray

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"dog conf={best_conf:.2f}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"{status} pixels={motion_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("webcam_yolo", frame)

        now = time.time()
        if now - last_print >= PRINT_EVERY_SEC:
            print(time.strftime("%H:%M:%S"), status, "pixels=", motion_count)
            last_print = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()