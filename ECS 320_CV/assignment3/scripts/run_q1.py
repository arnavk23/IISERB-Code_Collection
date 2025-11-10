"""Question 1: Velocity proxy via sparse optical flow

Uses `speed_proxy.mp4` in project root by default. Writes outputs to `outputs/`.
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.optical_flow import to_gray, compute_gradients, window_lucas_kanade, sparse_to_dense_flow, flow_to_rgb


def process(video_path, out_dir, max_frames=200, stride=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    speeds = []
    frame_idx = 0
    saved_examples = 0
    while True:
        ret, f1 = cap.read()
        if not ret:
            break
        ret, f2 = cap.read()
        if not ret:
            break
        g1 = to_gray(f1)
        g2 = to_gray(f2)
        Ix, Iy, It = compute_gradients(g1, g2)
        pts, flows = window_lucas_kanade(Ix, Iy, It, win_size=9, stride=12)
        mag = np.sqrt(np.sum(flows * flows, axis=1)) if flows.shape[0] > 0 else np.array([0.0])
        speed_px_per_sec = float(np.median(mag)) * fps
        speeds.append(speed_px_per_sec)
        # save a couple example overlays
        if saved_examples < 3 and pts.shape[0] > 0:
            h, w = g1.shape
            flow_dense = sparse_to_dense_flow(pts, flows, (h, w))
            viz = flow_to_rgb(flow_dense)
            out_fn = os.path.join(out_dir, f"q1_flow_example_{saved_examples}.png")
            cv2.imwrite(out_fn, viz)
            saved_examples += 1
        frame_idx += stride
        if frame_idx >= max_frames:
            break

        for _ in range(stride - 1):
            ret = cap.grab()
            if not ret:
                break
    cap.release()
    # plot speeds
    plt.figure(figsize=(6, 3))
    plt.plot(speeds)
    plt.xlabel('Frame (sampled)')
    plt.ylabel('Median speed (px/s)')
    plt.title('Q1: Speed proxy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'q1_speed_plot.png'))
    return speeds


def main():
    import argparse

    p = argparse.ArgumentParser()
    default_video = os.path.join(os.path.dirname(__file__), "..", "speed_proxy.mp4")
    p.add_argument("--video", type=str, default=default_video)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("Processing Q1 on", args.video)
    speeds = process(args.video, args.out)
    print("Saved Q1 outputs to", args.out)


if __name__ == "__main__":
    main()
