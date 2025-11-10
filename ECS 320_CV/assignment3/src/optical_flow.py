import numpy as np


def to_gray(im):
    """Convert image to grayscale (expects HxW or HxWx3). Returns float32.
    Implemented using NumPy operations (no OpenCV gradients) to match "from-scratch" requirement.
    """
    if im is None:
        return None
    arr = np.asarray(im)
    if arr.ndim == 3:
        r = arr[..., 0].astype(np.float32)
        g = arr[..., 1].astype(np.float32)
        b = arr[..., 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.float32)
    return arr.astype(np.float32)


def _conv2d(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(ih):
        for x in range(iw):
            patch = padded[y : y + kh, x : x + kw]
            out[y, x] = np.sum(patch * kernel)
    return out


def compute_gradients(img1, img2):
    """Compute spatial gradients Ix, Iy and temporal gradient It using NumPy.

    img1, img2: HxW float32 arrays.
    Returns Ix, Iy, It as float32 arrays.
    """
    I1 = img1.astype(np.float32)
    I2 = img2.astype(np.float32)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32) / 6.0
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32) / 6.0
    Ix = _conv2d(I1, kx)
    Iy = _conv2d(I1, ky)
    It = I2 - I1
    return Ix, Iy, It


def window_lucas_kanade(Ix, Iy, It, win_size=9, stride=16, eig_thresh=1e-3):
    h, w = Ix.shape[:2]
    half = win_size // 2
    pts = []
    flows = []
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            Wx = Ix[y - half : y + half + 1, x - half : x + half + 1]
            Wy = Iy[y - half : y + half + 1, x - half : x + half + 1]
            Wt = It[y - half : y + half + 1, x - half : x + half + 1]
            Ix_vec = Wx.reshape(-1)
            Iy_vec = Wy.reshape(-1)
            It_vec = Wt.reshape(-1)
            A11 = np.sum(Ix_vec * Ix_vec)
            A12 = np.sum(Ix_vec * Iy_vec)
            A22 = np.sum(Iy_vec * Iy_vec)
            b1 = -np.sum(Ix_vec * It_vec)
            b2 = -np.sum(Iy_vec * It_vec)
            A = np.array([[A11, A12], [A12, A22]], dtype=np.float32)
            eigs = np.linalg.eigvals(A)
            if np.min(eigs) < eig_thresh:
                continue
            try:
                nu = np.linalg.solve(A, np.array([b1, b2], dtype=np.float32))
            except np.linalg.LinAlgError:
                continue
            pts.append((x, y))
            flows.append((float(nu[0]), float(nu[1])))
    return np.array(pts, dtype=np.float32), np.array(flows, dtype=np.float32)


def _gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    g = np.exp(-(x ** 2) / (2 * sigma * sigma))
    g = g / np.sum(g)
    return g.astype(np.float32)


def _separable_gaussian(image, sigma=1.0):
    k = _gaussian_kernel1d(sigma)
    ih, iw = image.shape
    pad = len(k) // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')
    tmp = np.zeros_like(padded)
    for y in range(padded.shape[0]):
        row = padded[y]
        tmp[y, pad:pad+iw] = np.convolve(row, k, mode='valid')
    out = np.zeros((ih, iw), dtype=np.float32)
    for x in range(pad, pad+iw):
        col = tmp[:, x]
        out[:, x - pad] = np.convolve(col, k, mode='valid')
    return out


def build_gaussian_pyramid(img, levels, sigma=1.0):
    pyr = [img]
    cur = img
    for i in range(1, levels):
        blurred = _separable_gaussian(cur, sigma=sigma)
        down = blurred[::2, ::2]
        pyr.append(down)
        cur = down
    return pyr


def sparse_to_dense_flow(pts, flows, out_shape, downscale=8):
    h, w = out_shape[:2]
    flow = np.zeros((h, w, 2), dtype=np.float32)
    if pts.shape[0] == 0:
        return flow
    sparse_u = np.zeros((h, w), dtype=np.float32)
    sparse_v = np.zeros((h, w), dtype=np.float32)
    for (x, y), (u, v) in zip(pts.astype(int), flows):
        if 0 <= y < h and 0 <= x < w:
            sparse_u[y, x] = u
            sparse_v[y, x] = v
    sw = max(1, w // downscale)
    sh = max(1, h // downscale)
    small_u = sparse_u[::downscale, ::downscale]
    small_v = sparse_v[::downscale, ::downscale]
    big_u = np.repeat(np.repeat(small_u, downscale, axis=0), downscale, axis=1)
    big_v = np.repeat(np.repeat(small_v, downscale, axis=0), downscale, axis=1)
    big_u = big_u[:h, :w]
    big_v = big_v[:h, :w]
    flow[:, :, 0] = big_u
    flow[:, :, 1] = big_v
    return flow


def _bilinear_sample(img, xs, ys):
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(int)
    x1 = x0 + 1
    y0 = np.floor(ys).astype(int)
    y1 = y0 + 1
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)
    if img.ndim == 2:
        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]
        return wa * Ia + wb * Ib + wc * Ic + wd * Id
    else:
        Ia = img[y0, x0, :]
        Ib = img[y0, x1, :]
        Ic = img[y1, x0, :]
        Id = img[y1, x1, :]
        out = (wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id)
        return out


def warp_image(img, flow):
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    xs = grid_x + flow[..., 0]
    ys = grid_y + flow[..., 1]
    return _bilinear_sample(img, xs, ys)


def flow_to_rgb(flow, max_magnitude=None):
    h, w = flow.shape[:2]
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    if max_magnitude is None:
        max_magnitude = np.max(mag) + 1e-9
    hsv_h = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv_s = np.clip((mag / max_magnitude) * 255, 0, 255).astype(np.uint8)
    hsv_v = np.ones_like(hsv_s) * 255
    # convert HSV to RGB roughly
    import colorsys

    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            hval = hsv_h[y, x] / 180.0
            sval = hsv_s[y, x] / 255.0
            vval = hsv_v[y, x] / 255.0
            r, g, b = colorsys.hsv_to_rgb(hval, sval, vval)
            out[y, x, 0] = int(255 * r)
            out[y, x, 1] = int(255 * g)
            out[y, x, 2] = int(255 * b)
    return out


if __name__ == "__main__":
    print("optical_flow helpers available (NumPy implementations)")
