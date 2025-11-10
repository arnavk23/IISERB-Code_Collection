"""Question 2: Video retiming (one intermediate frame per pair)

Uses `video_retiming.mp4` by default and writes middle-frame images into `outputs/`.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.optical_flow import to_gray, compute_gradients, window_lucas_kanade, sparse_to_dense_flow, warp_image


def process(video_path, out_dir, num_pairs=12):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    n = len(frames)
    saved = 0
    for i in range(min(n - 1, num_pairs)):
        f1 = frames[i]
        f2 = frames[i + 1]
        g1 = to_gray(f1)
        g2 = to_gray(f2)
        Ix, Iy, It = compute_gradients(g1, g2)
        pts_fwd, flows_fwd = window_lucas_kanade(Ix, Iy, It, win_size=9, stride=8)
        # backward
        Ix_b, Iy_b, It_b = compute_gradients(g2, g1)
        pts_bwd, flows_bwd = window_lucas_kanade(Ix_b, Iy_b, It_b, win_size=9, stride=8)
        h, w = g1.shape
        flow_fwd_dense = sparse_to_dense_flow(pts_fwd, flows_fwd, (h, w))
        flow_bwd_dense = sparse_to_dense_flow(pts_bwd, flows_bwd, (h, w))
        mid1 = warp_image(f1, 0.5 * flow_fwd_dense)
        mid2 = warp_image(f2, 0.5 * flow_bwd_dense)
        mid = cv2.addWeighted(mid1, 0.5, mid2, 0.5, 0)
        out_fn = os.path.join(out_dir, f"q2_mid_{i:03d}.png")
        cv2.imwrite(out_fn, mid)
        saved += 1
    return saved


def main():
    import argparse

    p = argparse.ArgumentParser()
    default_video = os.path.join(os.path.dirname(__file__), "..", "video_retiming.mp4")
    p.add_argument("--video", type=str, default=default_video)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("Processing Q2 on", args.video)
    saved = process(args.video, args.out)
    print(f"Wrote {saved} interpolated mid-frames to {args.out}")


if __name__ == "__main__":
    main()
