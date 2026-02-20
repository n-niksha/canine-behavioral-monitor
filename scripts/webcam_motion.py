import cv2
import numpy as np
import time
from collections import deque

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing index 0->1.")

    prev_gray = None

    DIFF_THRESHOLD = 25
    WARMUP_SEC = 2.0
    PRINT_EVERY_SEC = 1.0

    # Smoothing (less sticky)
    N = 3
    motion_hist = deque(maxlen=N)

    # Hysteresis thresholds (tuned from your output)
    HIGH = 12000   # enter MOVING
    LOW  = 6000    # return to STILL

    start_time = time.time()
    last_print = start_time
    last_t = start_time

    active_seconds = 0.0
    still_seconds = 0.0

    is_moving = False  # state

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        now = time.time()
        dt = now - last_t
        last_t = now

        status = "warming_up"
        raw = 0
        avg = 0

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            raw = int(np.sum(thresh > 0))
            motion_hist.append(raw)
            avg = int(sum(motion_hist) / len(motion_hist))

            if (now - start_time) < WARMUP_SEC:
                status = "warming_up"
            else:
                # hysteresis state machine
                if not is_moving and avg > HIGH:
                    is_moving = True
                elif is_moving and avg < LOW:
                    is_moving = False

                status = "MOVING" if is_moving else "STILL"

                if is_moving:
                    active_seconds += dt
                else:
                    still_seconds += dt

            cv2.putText(frame, f"{status} raw={raw} avg={avg}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"active={active_seconds:.1f}s still={still_seconds:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("motion_mask", thresh)

        cv2.imshow("webcam", frame)

        if prev_gray is not None and (now - last_print) >= PRINT_EVERY_SEC:
            print(time.strftime("%H:%M:%S"), status, "raw=", raw, "avg=", avg,
                  "| active=", round(active_seconds,1), "still=", round(still_seconds,1))
            last_print = now

        prev_gray = gray

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()