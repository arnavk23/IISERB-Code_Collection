"""Run all assignment scripts (Q1..Q4) on the provided videos and collect outputs.

This script calls the individual scripts using the current Python interpreter.
It expects the videos to be in the project root with names:
 - speed_proxy.mp4
 - video_retiming.mp4
 - camera_shake.mp4
 - vanishing_point.mp4

Outputs are written to `outputs/`.
"""
import os
import subprocess
import sys


def run(cmd):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR (returncode=", res.returncode, "):")
        print(res.stderr)
        return False
    return True


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py = sys.executable
    os.makedirs(os.path.join(root, "outputs"), exist_ok=True)

    tasks = [
        [py, os.path.join(root, "scripts", "run_q1.py"), "--video", os.path.join(root, "speed_proxy.mp4"), "--out", os.path.join(root, "outputs")],
        [py, os.path.join(root, "scripts", "run_q2.py"), "--video", os.path.join(root, "video_retiming.mp4"), "--out", os.path.join(root, "outputs")],
        [py, os.path.join(root, "scripts", "run_q3.py"), "--video", os.path.join(root, "camera_shake.mp4"), "--out", os.path.join(root, "outputs", "q3_stabilized.mp4")],
        [py, os.path.join(root, "scripts", "run_q4.py"), "--video", os.path.join(root, "vanishing_point.mp4"), "--out", os.path.join(root, "outputs", "q4_vp.mp4"), "--preview", os.path.join(root, "outputs", "q4_vp_frame.png")],
    ]

    all_ok = True
    for cmd in tasks:
        ok = run(cmd)
        all_ok = all_ok and ok
        if not ok:
            print("Task failed, aborting remaining tasks.")
            break

    if all_ok:
        print("All tasks completed successfully. Check the outputs/ folder.")
    else:
        print("One or more tasks failed. See messages above.")


if __name__ == "__main__":
    main()
