"""
Structure-from-Motion (SfM) pipeline for a video.

This script extracts frames from a video, computes local features, matches them pairwise, 
recovers relative poses via the essential matrix, triangulates matched points to form a 
sparse 3D point cloud, and writes the result to a PLY file.

Limitations / assumptions:
- Uses a simple incremental/pairwise SfM approach (no sophisticated track management).
- Assumes unknown intrinsics: we use a simple pinhole focal estimate f = 0.8*width.
- Uses OpenCV for features, matching, pose recovery and triangulation.
- Intended as a generic, runnable starting point you can extend.

Usage:
    python3 sfm_from_video.py --video path/to/video.mp4 --out cloud.ply --step 5 --max-frames 80

Dependencies:
    pip install opencv-contrib-python numpy scipy matplotlib

Output:
- A PLY file with the sparse 3D points (`--out`).
- A simple 3D scatterplot if matplotlib is available.
"""

import os
import sys
import argparse
from collections import defaultdict

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

def extract_frames(video_path, step=5, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    idx = 0
    taken = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
            taken += 1
            if max_frames and taken >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def init_feature_detector():
    # Prefer SIFT if available, otherwise ORB
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create()
    try:
        return cv2.SIFT_create()
    except Exception:
        return cv2.ORB_create(nfeatures=2000)


def detect_and_compute(detector, gray):
    kp, des = detector.detectAndCompute(gray, None)
    return kp, des


def match_descriptors(des1, des2, use_sift=True):
    if des1 is None or des2 is None:
        return []
    if use_sift:
        # use FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # Lowe's ratio test
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches


def make_intrinsics(width, height, focal=None):
    if focal is None:
        focal = 0.8 * max(width, height)
    K = np.array([[focal, 0, width / 2.0],
                  [0, focal, height / 2.0],
                  [0, 0, 1]])
    return K


def normalize_points(pts):
    # pts: Nx2
    return pts.reshape(-1, 2)


def triangulate_points(P1, P2, pts1, pts2):
    # pts1, pts2 are Nx2 in pixel coordinates
    pts1_h = np.array(pts1).T
    pts2_h = np.array(pts2).T
    X = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X = X / X[3]
    return X[:3].T


def save_ply(filename, points, colors=None):
    # points: Nx3
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        for i, p in enumerate(points):
            if colors is not None:
                c = colors[i]
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")
            else:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

def incremental_sfm(frames, step_result=1, max_init_matches=2000):
    detector = init_feature_detector()
    use_sift = hasattr(cv2, 'SIFT_create')

    H, W = frames[0].shape[:2]
    K = make_intrinsics(W, H)

    # detect features for all frames
    print('Detecting features...')
    keypoints = []
    descriptors = []
    gray_frames = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
        kp, des = detect_and_compute(detector, gray)
        keypoints.append(kp)
        descriptors.append(des)

    poses = []  # list of (R, t) global poses
    poses.append((np.eye(3), np.zeros((3, 1))))  # frame0 at origin

    points3d = []
    point_colors = []

    # find initial pair: frame0 and frame1 (or next) with enough matches
    print('Finding initial pair...')
    init_i = 0
    init_j = None
    for j in range(1, len(frames)):
        matches = match_descriptors(descriptors[0], descriptors[j], use_sift=use_sift)
        if len(matches) > 50:
            init_j = j
            break
    if init_j is None:
        raise RuntimeError('Could not find initial pair with enough matches')

    print(f'Initial pair: 0 and {init_j} with {len(matches)} matches')

    # compute essential and recover pose between 0 and init_j
    pts0 = np.array([keypoints[0][m.queryIdx].pt for m in matches])
    ptsj = np.array([keypoints[init_j][m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts0, ptsj, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts0, ptsj, K)

    poses = []
    poses.append((np.eye(3), np.zeros((3, 1))))
    poses.insert(init_j, (R, t))

    # triangulate initial points between 0 and init_j
    print('Triangulating initial points...')
    P0 = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    Pj = K.dot(np.hstack((R, t)))

    pts3 = triangulate_points(P0, Pj, pts0[mask.ravel()==1], ptsj[mask.ravel()==1])
    for p in pts3:
        points3d.append(p)
    # color from frame0
    for m, mk in zip(matches, mask.ravel()):
        if mk:
            x, y = keypoints[0][m.queryIdx].pt
            colors = frames[0][int(y), int(x)]
            point_colors.append(colors)

    # Process frames sequentially: for simplicity triangulate pairwise with previous
    prev_idx = init_j
    for i in range(init_j+1, len(frames)):
        print(f'Processing frame {i} (pair {prev_idx} - {i})')
        matches = match_descriptors(descriptors[prev_idx], descriptors[i], use_sift=use_sift)
        if len(matches) < 30:
            print('  Too few matches, skipping')
            prev_idx = i
            poses.insert(i, (np.eye(3), np.zeros((3, 1))))
            continue
        pts_prev = np.array([keypoints[prev_idx][m.queryIdx].pt for m in matches])
        pts_i = np.array([keypoints[i][m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts_prev, pts_i, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print('  Essential matrix failed, skipping')
            prev_idx = i
            poses.insert(i, (np.eye(3), np.zeros((3, 1))))
            continue
        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_prev, pts_i, K)
        # global pose of prev_idx
        R_prev, t_prev = poses[prev_idx]
        R_i = R_rel.dot(R_prev)
        t_i = R_rel.dot(t_prev) + t_rel
        poses.insert(i, (R_i, t_i))
        # triangulate inlier matches
        in1 = pts_prev[mask.ravel()==1]
        in2 = pts_i[mask.ravel()==1]
        P_prev = K.dot(np.hstack((R_prev, t_prev)))
        P_i = K.dot(np.hstack((R_i, t_i)))
        try:
            new_pts3 = triangulate_points(P_prev, P_i, in1, in2)
        except Exception as e:
            print('  Triangulation failed:', e)
            prev_idx = i
            continue
        for p in new_pts3:
            points3d.append(p)
        for m, mk in zip(matches, mask.ravel()):
            if mk:
                x, y = keypoints[prev_idx][m.queryIdx].pt
                colors = frames[prev_idx][int(y), int(x)]
                point_colors.append(colors)
        prev_idx = i

    points3d = np.array(points3d)
    point_colors = np.array(point_colors) if len(point_colors) else None

    return K, poses, points3d, point_colors

def bundle_adjustment(K, poses, points3d, observations):
    # observations: list of (frame_idx, point_idx, x, y)
    # Very small/simple BA: refine camera poses (rvec,t) and 3D points jointly
    if not SCIPY_AVAILABLE:
        print('Scipy not available: skipping bundle adjustment')
        return poses, points3d
    # Build parameter vector
    cam_params = []
    for R, t in poses:
        rvec, _ = cv2.Rodrigues(R)
        cam_params.append(np.hstack((rvec.ravel(), t.ravel())))
    cam_params = np.array(cam_params).ravel()
    X0 = points3d.ravel()

    cam_idx = np.array([o[0] for o in observations], dtype=int)
    pt_idx = np.array([o[1] for o in observations], dtype=int)
    pts2d = np.array([[o[2], o[3]] for o in observations], dtype=float)

    def pack(params_cam, params_X):
        return params_cam.ravel(), params_X.ravel()

    def project(points, cam_param):
        rvec = cam_param[:3]
        t = cam_param[3:6]
        R, _ = cv2.Rodrigues(rvec)
        P = K.dot(np.hstack((R, t.reshape(3,1))))
        X_h = np.hstack((points, np.ones((points.shape[0],1)))).T
        x = P.dot(X_h)
        x = (x[:2]/x[2]).T
        return x

    def residuals(params):
        n_cams = len(poses)
        n_points = len(points3d)
        cam_params = params[:n_cams*6].reshape((n_cams,6))
        X = params[n_cams*6:].reshape((n_points,3))
        pred = np.zeros_like(pts2d)
        for i in range(len(observations)):
            c = cam_idx[i]
            p = pt_idx[i]
            cam = cam_params[c]
            Xp = X[p:p+1]
            xproj = project(Xp, cam)
            pred[i] = xproj[0]
        return (pred - pts2d).ravel()

    print('Running bundle adjustment (this may take a while)...')
    x0 = np.hstack((cam_params.ravel(), X0.ravel()))
    res = least_squares(residuals, x0, verbose=2, x_scale='jac', ftol=1e-4, method='lm', max_nfev=200)
    x_opt = res.x
    n_cams = len(poses)
    cam_params_opt = x_opt[:n_cams*6].reshape((n_cams,6))
    X_opt = x_opt[n_cams*6:].reshape((len(points3d),3))
    new_poses = []
    for cam in cam_params_opt:
        rvec = cam[:3]
        t = cam[3:6].reshape(3,1)
        R, _ = cv2.Rodrigues(rvec)
        new_poses.append((R, t))
    return new_poses, X_opt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', default='cloud.ply')
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--max-frames', type=int, default=80)
    args = parser.parse_args()

    frames = extract_frames(args.video, step=args.step, max_frames=args.max_frames)
    print(f'Extracted {len(frames)} frames')
    if len(frames) < 2:
        print('Need at least 2 frames for SfM')
        return
    K, poses, points3d, colors = incremental_sfm(frames)

    if points3d is None or len(points3d) == 0:
        print('No points reconstructed')
        return

    print(f'Reconstructed {len(points3d)} 3D points')
    save_ply(args.out, points3d, colors)
    print('Saved', args.out)

    if MPL_AVAILABLE:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], s=1)
        ax.set_title('Sparse point cloud')
        plt.show()

if __name__ == '__main__':
    main()
