import numpy as np
import cv2

# ── Optional pygame for audio ──────────────────
try:
    import pygame
    pygame.mixer.init()
    PYGAME = True
except ImportError:
    PYGAME = False


# ── Geometry ───────────────────────────────────

def get_points(lms, indices, w, h):
    """Return (x, y) pixel coords for the given landmark indices."""
    return [(lms[i].x * w, lms[i].y * h) for i in indices]


def dist(p1, p2):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def ear(lms, idx, w, h):
    """Eye Aspect Ratio — lower value means more closed."""
    p = get_points(lms, idx, w, h)
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]) + 1e-6)


def mar(lms, idx, w, h):
    """Mouth Aspect Ratio — higher value means more open (yawn)."""
    p = get_points(lms, idx, w, h)
    return (dist(p[2], p[3]) + dist(p[4], p[5])) / (2 * dist(p[0], p[1]) + 1e-6)


# ── Drawing ────────────────────────────────────

def draw_contour(frame, lms, idx, w, h, color):
    """Draw a closed polygon over the given landmark indices."""
    pts = [tuple(map(int, p)) for p in get_points(lms, idx, w, h)]
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], color, 1)


def text(frame, txt, pos, color=(255, 255, 255)):
    """Render small HUD text onto the frame."""
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def banner(frame, txt, color=(0, 0, 255)):
    """Draw a full-width status banner at the top of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
    cv2.putText(frame, txt, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ── Audio ──────────────────────────────────────

def beep():
    """Play an 880 Hz alert tone (pygame) or fall back to terminal bell."""
    if PYGAME:
        t    = np.linspace(0, 0.5, 22050, False)
        wave = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
        pygame.sndarray.make_sound(np.column_stack([wave, wave])).play()
    else:
        print("\a", end="", flush=True)
