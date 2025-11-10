"""
Estimate and visualize the vanishing point in a video.

Approach:
- Canny edge detector
- Probabilistic Hough lines
- Compute intersections between line pairs
- Cluster intersections (kmeans with k=1) to get dominant vanishing point
- Overlay lines and VP on frames and write an output video + preview image

Usage: python scripts/run_q4.py --video /path/to/vanishing_point.mp4 --out outputs/q4_vp.mp4
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def line_to_params(line):
    x1, y1, x2, y2 = line
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c

def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return (x, y)

def estimate_vp_from_lines(lines, img_shape):
    params = [line_to_params(l[0]) for l in lines]
    inters = []
    H, W = img_shape[:2]
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            pt = intersect(params[i], params[j])
            if pt is None:
                continue
            x, y = pt
            if not (-W * 5 < x < W * 6 and -H * 5 < y < H * 6):
                continue
            inters.append([x, y])
    if len(inters) == 0:
        return None, inters
    pts = np.array(inters, dtype=np.float32)

    Z = pts.reshape((-1, 2))
    try:
        K = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        cx, cy = float(center[0][0]), float(center[0][1])
        return (cx, cy), inters
    except Exception:
        # fallback to median
        med = np.median(pts, axis=0)
        return (float(med[0]), float(med[1])), inters


def draw_overlay(frame, lines, vp, intersections):
    out = frame.copy()
    # draw lines
    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    # draw intersection points (sample)
    for pt in intersections[:200]:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    # draw vanishing point
    if vp is not None:
        vx, vy = int(round(vp[0])), int(round(vp[1]))
        cv2.drawMarker(out, (vx, vy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    return out


def process_video(video_path, out_path, preview_img_path=None, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    frame_idx = 0
    vp_history = []
    preview_saved = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # focus on upper half for vanishing lines (road horizon)
        mask = np.zeros_like(gray)
        mask[: int(h * 0.8), :] = 255
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, mask)
        # Hough lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=15)
        vp, intersections = None, []
        if lines is not None and len(lines) >= 2:
            vp, intersections = estimate_vp_from_lines(lines, frame.shape)
        overlay = draw_overlay(frame, lines if lines is not None else [], vp, intersections)
        out.write(overlay)
        if not preview_saved and preview_img_path is not None:
            cv2.imwrite(preview_img_path, overlay)
            preview_saved = True
        if vp is not None:
            vp_history.append(vp)
        frame_idx += 1
        if frame_idx >= max_frames:
            break
    cap.release()
    out.release()
    # compute average VP
    if len(vp_history) == 0:
        return None
    avg = np.median(np.array(vp_history), axis=0)
    return tuple(float(x) for x in avg)


def main():
    import argparse

    p = argparse.ArgumentParser()
    default_video = os.path.join(os.path.dirname(__file__), "..", "vanishing_point.mp4")
    p.add_argument("--video", type=str, default=default_video)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "outputs", "q4_vp.mp4"))
    p.add_argument("--preview", type=str, default=os.path.join(os.path.dirname(__file__), "..", "outputs", "q4_vp_frame.png"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Processing {args.video} -> {args.out}")
    vp = process_video(args.video, args.out, preview_img_path=args.preview, max_frames=400)
    if vp is None:
        print("No vanishing point found")
    else:
        print(f"Estimated vanishing point (median over frames): {vp}")


if __name__ == "__main__":
    main()
