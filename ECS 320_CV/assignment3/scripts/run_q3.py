"""Question 3: Basic translation stabilisation

Uses `camera_shake.mp4` by default and writes `outputs/q3_stabilized.mp4`.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.optical_flow import to_gray, compute_gradients, window_lucas_kanade


def process(video_path, out_path, smooth_window=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    n = len(frames)
    if n < 2:
        raise RuntimeError("Need at least 2 frames")
    trans = []
    for i in range(n - 1):
        g1 = to_gray(frames[i])
        g2 = to_gray(frames[i + 1])
        Ix, Iy, It = compute_gradients(g1, g2)
        pts, flows = window_lucas_kanade(Ix, Iy, It, win_size=9, stride=16)
        if flows.shape[0] == 0:
            tx, ty = 0.0, 0.0
        else:
            tx = float(np.median(flows[:, 0]))
            ty = float(np.median(flows[:, 1]))
        trans.append((tx, ty))
    trans = np.array(trans)
    path = np.vstack([[0.0, 0.0], np.cumsum(trans, axis=0)])
    # smooth each coordinate
    kernel = np.ones(smooth_window) / smooth_window
    sx = np.convolve(path[:, 0], kernel, mode='same')
    sy = np.convolve(path[:, 1], kernel, mode='same')
    smooth_path = np.stack([sx, sy], axis=1)
    # apply inverse transform per frame
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for i in range(n):
        dx = smooth_path[i, 0] if i < smooth_path.shape[0] else 0.0
        dy = smooth_path[i, 1] if i < smooth_path.shape[0] else 0.0
        M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        warped = cv2.warpAffine(frames[i], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.write(warped)
    out.release()
    return trans, smooth_path


def main():
    import argparse
    p = argparse.ArgumentParser()
    default_video = os.path.join(os.path.dirname(__file__), "..", "camera_shake.mp4")
    p.add_argument("--video", type=str, default=default_video)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "outputs", "q3_stabilized.mp4"))
    p.add_argument("--smooth", type=int, default=15)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print("Processing Q3 on", args.video)
    trans, smooth_path = process(args.video, args.out, smooth_window=args.smooth)
    print("Wrote stabilized video to", args.out)


if __name__ == "__main__":
    main()
