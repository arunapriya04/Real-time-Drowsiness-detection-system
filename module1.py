import cv2
import mediapipe as mp
import numpy as np
import time
import threading

from module3 import (
    EAR_THRESH, MAR_THRESH,
    EAR_FRAMES, YAWN_FRAMES,
    CALIB_FRAMES, ADAPT_RATIO,
    LEFT_EYE, RIGHT_EYE, MOUTH,
)
from module2 import ear, mar, draw_contour, text, banner, beep


def main():
    # ── MediaPipe setup ────────────────────────
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.2,   # lowered for better detection
        min_tracking_confidence=0.2,
    )

    # ── Camera setup ───────────────────────────
    cap = cv2.VideoCapture(0)

    time.sleep(2)

    if not cap.isOpened():
      print("Error: Could not open camera")
    return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # ── Session state ──────────────────────────
    eye_cnt    = 0
    yawn_cnt   = 0
    alerts     = 0
    yawns      = 0
    calib_ears = []
    thresh     = EAR_THRESH
    last_beep  = 0
    start      = time.time()

    # ── Main loop ──────────────────────────────
    elapsed = 0  # add before while loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # ✅ Resize + RGB conversion (correct placement)
        frame = cv2.resize(frame, (640, 480))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        h, w = frame.shape[:2]

        if not results.multi_face_landmarks:
            banner(frame, "NO FACE DETECTED", (80, 80, 80))

        else:
            lms = results.multi_face_landmarks[0].landmark

            avg_ear = (
                ear(lms, LEFT_EYE, w, h) +
                ear(lms, RIGHT_EYE, w, h)
            ) / 2

            m = mar(lms, MOUTH, w, h)

            # ── Calibration ─────────────────────
            if len(calib_ears) < CALIB_FRAMES:
                calib_ears.append(avg_ear)
                if len(calib_ears) == CALIB_FRAMES:
                    thresh = float(np.clip(
                        np.mean(calib_ears) * ADAPT_RATIO,
                        0.18, 0.30
                    ))

            # ── Drowsiness detection ────────────
            eye_cnt = eye_cnt + 1 if avg_ear < thresh else 0
            drowsy  = eye_cnt >= EAR_FRAMES

            if drowsy and eye_cnt == EAR_FRAMES:
                alerts += 1

            # ── Yawn detection ─────────────────
            if m > MAR_THRESH:
                yawn_cnt += 1
            else:
                if yawn_cnt >= YAWN_FRAMES:
                    yawns += 1
                yawn_cnt = 0

            yawning = yawn_cnt >= YAWN_FRAMES

            # ── Alert sound ────────────────────
            if (drowsy or yawning) and time.time() - last_beep > 2:
                threading.Thread(target=beep, daemon=True).start()
                last_beep = time.time()

            # ── Draw overlays ──────────────────
            eye_col   = (0, 0, 255) if drowsy else (0, 255, 0)
            mouth_col = (0, 165, 255) if yawning else (0, 255, 200)

            for idx in [LEFT_EYE, RIGHT_EYE]:
                draw_contour(frame, lms, idx, w, h, eye_col)

            draw_contour(frame, lms, MOUTH, w, h, mouth_col)

            if drowsy:
                banner(frame, "DROWSINESS DETECTED!", (0, 0, 200))
            elif yawning:
                banner(frame, "YAWN DETECTED!", (0, 120, 200))
            else:
                banner(frame, "SYSTEM ACTIVE", (0, 140, 0))

            # ── HUD ────────────────────────────
            text(
                frame,
                f"EAR: {avg_ear:.2f} | MAR: {m:.2f} | Alerts: {alerts} | Yawns: {yawns}",
                (10, 70),
            )

        # ── Timer ─────────────────────────────
        elapsed = int(time.time() - start)
        text(
            frame,
            f"Time: {elapsed//60:02d}:{elapsed%60:02d} | Q to quit",
            (w - 220, h - 10),
        )

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nSession: {elapsed//60}m {elapsed%60}s | Alerts: {alerts} | Yawns: {yawns}")


if __name__ == "__main__":
    main()