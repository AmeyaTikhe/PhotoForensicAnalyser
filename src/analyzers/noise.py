"""
noise.py — PhotoForensics noise analysis module
Cumulative build: V1.0 -> V5.0

V5.0 adds a real fusion layer. Previously (V1.0-V4.0), the variance-based
detector, PRNU extraction, the wavelet multiscale map, and the Isolation
Forest classifier were each computed independently and never combined —
only the variance-based z-score ever produced the final suspicious_boxes.
PRNU and the wavelet map were computed and discarded. The ML detector
produced its own independent boxes that never interacted with anything else.

V5.0 fixes this: every signal is resampled onto one shared patch grid,
z-scored, weighted, summed into a single "noise consistency score" map, and
that fused map is what gets thresholded into suspicious_regions(). There is
one decision, not four independent ones.

Honesty note on PRNU: true PRNU-based camera fingerprinting requires a
reference fingerprint averaged from many images known to come from the same
sensor (see prnu_average/prnu_correlate below, still provided for that use
case). Given a single image, there's no reference to correlate against, so
V5.0 only uses the PRNU-domain residual as a *self-consistency* signal
(local variance of the denoised-noise-residual) — flagging patches where
that residual behaves statistically differently from the rest of the image.
That's weaker than genuine sensor-identification PRNU analysis and is
labeled as such in the result dict (see `prnu_self_consistency` vs `prnu`).

Dependencies:
    pip install opencv-python numpy scipy PyWavelets scikit-learn matplotlib
"""

import sys
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# ============================================================
# V1.0 — denoise, residual, visualization, statistics
# ============================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def denoise_gaussian(img, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def denoise_median(img, ksize=5):
    return cv2.medianBlur(img, ksize)


def denoise_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def compute_residual(img, denoised):
    return img.astype(np.float32) - denoised.astype(np.float32)


def visualize_residual(residual, gain=5.0):
    vis = np.abs(residual) * gain
    return np.clip(vis, 0, 255).astype(np.uint8)


def noise_statistics(residual):
    gray = residual if residual.ndim == 2 else residual.mean(axis=2)
    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "var": float(np.var(gray)),
        "min": float(np.min(gray)),
        "max": float(np.max(gray)),
    }


# ============================================================
# V1.1 — adaptive denoising, residual normalization, heatmap
# ============================================================

def denoise_adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:
        return denoise_gaussian(img, 3, 0.8)
    elif lap_var < 200:
        return denoise_bilateral(img)
    else:
        return denoise_median(img, 3)


def normalize_residual(residual):
    r = residual.copy()
    r -= r.mean()
    std = r.std()
    if std > 1e-6:
        r /= std
    return r


def noise_heatmap(residual, colormap=cv2.COLORMAP_JET):
    gray = np.abs(residual if residual.ndim == 2 else residual.mean(axis=2))
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap)


# ============================================================
# V1.2 — sliding-window local noise estimation, variance map
# ============================================================

def local_noise_map(residual, window=16, stride=8):
    """Per-patch variance of a residual/noise map. Works on any 2D or 3D
    array — reused later (V5.0) for the PRNU residual too, not just the
    denoise residual, so the grid layout stays identical across signals."""
    gray = residual if residual.ndim == 2 else residual.mean(axis=2)
    h, w = gray.shape
    out_h = max((h - window) // stride + 1, 1)
    out_w = max((w - window) // stride + 1, 1)
    var_map = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            y, x = i * stride, j * stride
            patch = gray[y:y + window, x:x + window]
            var_map[i, j] = patch.var()
    return var_map


def upsample_map(small_map, target_shape):
    return cv2.resize(small_map, (target_shape[1], target_shape[0]),
                       interpolation=cv2.INTER_CUBIC)


# ============================================================
# V2.0 — sensor noise consistency, suspicious region detection
# ============================================================

def sensor_noise_consistency(var_map, z_thresh=2.0):
    """Kept for standalone/legacy use. V5.0 does this same z-score+threshold
    step but on the *fused* map instead of var_map alone — see
    fuse_noise_consistency() and threshold_consistency_map()."""
    mean, std = var_map.mean(), var_map.std()
    if std < 1e-6:
        return np.zeros_like(var_map, dtype=bool)
    z = (var_map - mean) / std
    return np.abs(z) > z_thresh


def suspicious_regions(mask, min_area=20):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            boxes.append(cv2.boundingRect(c))
    return boxes


def draw_suspicious_regions(img, boxes, scale_x=1.0, scale_y=1.0, color=(0, 0, 255)):
    out = img.copy()
    for (x, y, w, h) in boxes:
        x0, y0 = int(x * scale_x), int(y * scale_y)
        w0, h0 = int(w * scale_x), int(h * scale_y)
        cv2.rectangle(out, (x0, y0), (x0 + w0, y0 + h0), color, 2)
    return out


# ============================================================
# V2.1 — PRNU preparation (camera fingerprint foundation)
# ============================================================

def prnu_extract(img, wavelet_denoise_func=None):
    if wavelet_denoise_func is None:
        wavelet_denoise_func = wavelet_denoise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    denoised = wavelet_denoise_func(gray)
    return gray - denoised


def prnu_average(prnu_list):
    """Build a reference camera fingerprint from MANY images known to be
    from the same sensor. Requires a corpus — not usable on a single image.
    Still here for when that corpus is available; not called by analyze()."""
    return np.stack(prnu_list, axis=0).mean(axis=0)


def prnu_correlate(fingerprint, candidate_prnu):
    """Correlate a candidate image's PRNU residual against a reference
    fingerprint (see prnu_average). Genuine camera-identification PRNU
    analysis — needs a reference, so it's not part of the single-image
    fused pipeline in analyze(); call it directly if you have one."""
    f = fingerprint.flatten() - fingerprint.mean()
    c = candidate_prnu.flatten() - candidate_prnu.mean()
    denom = np.linalg.norm(f) * np.linalg.norm(c)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(f, c) / denom)


# ============================================================
# V3.0 — multi-scale wavelet noise estimation
# ============================================================

def wavelet_denoise(gray, wavelet="db8", level=4):
    coeffs = pywt.wavedec2(gray, wavelet, level=level)
    cA, details = coeffs[0], coeffs[1:]
    sigma = np.median(np.abs(details[-1][2])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(gray.size))
    out = [cA]
    for (cH, cV, cD) in details:
        out.append((
            pywt.threshold(cH, thresh, mode="soft"),
            pywt.threshold(cV, thresh, mode="soft"),
            pywt.threshold(cD, thresh, mode="soft"),
        ))
    denoised = pywt.waverec2(out, wavelet)
    return denoised[:gray.shape[0], :gray.shape[1]]


def multiscale_noise_map(gray, wavelet="db8", levels=(1, 2, 3, 4)):
    maps = []
    for lvl in levels:
        denoised = wavelet_denoise(gray, wavelet, lvl)
        residual = gray.astype(np.float32) - denoised.astype(np.float32)
        maps.append(np.abs(residual))
    return np.stack(maps, axis=0).mean(axis=0)


# ============================================================
# V4.0 — ML-assisted noise consistency (legacy: independent boxes)
# ============================================================

def extract_patch_features(gray, window=16, stride=8):
    h, w = gray.shape
    feats, coords = [], []
    for y in range(0, h - window + 1, stride):
        for x in range(0, w - window + 1, stride):
            patch = gray[y:y + window, x:x + window]
            feats.append([
                patch.mean(), patch.std(), patch.var(),
                np.percentile(patch, 25), np.percentile(patch, 75),
            ])
            coords.append((x, y))
    return np.array(feats), coords


def ml_noise_consistency(gray, window=16, stride=8, contamination=0.05):
    """Legacy V4.0 entry point: returns a binary mask/boxes from
    IsolationForest.predict(). Kept for backward compatibility, but this is
    what produced ~980 flagged patches on an ordinary image — predict()
    forces a hard in/out call per patch with no way to weigh it against
    other evidence. V5.0 uses ml_anomaly_score_map() instead, which returns
    the continuous decision_function score so it can be z-scored and fused
    rather than voting on its own."""
    feats, coords = extract_patch_features(gray, window, stride)
    if len(feats) < 10:
        return np.zeros(gray.shape, dtype=bool), []
    model = IsolationForest(contamination=contamination, random_state=0)
    labels = model.fit_predict(feats)
    mask = np.zeros(gray.shape, dtype=bool)
    boxes = []
    for (x, y), label in zip(coords, labels):
        if label == -1:
            mask[y:y + window, x:x + window] = True
            boxes.append((x, y, window, window))
    return mask, boxes


def ml_anomaly_score_map(gray, window=16, stride=8):
    """V5.0: continuous anomaly score per patch, on the SAME grid layout as
    local_noise_map (row-major, y outer / x inner, same window/stride), so
    it can be stacked with the other signals with no resampling needed.
    Higher score = more anomalous (we negate sklearn's decision_function,
    where lower/negative already means 'more anomalous', to keep every
    fused signal oriented the same way: high = suspicious)."""
    h, w = gray.shape
    out_h = max((h - window) // stride + 1, 1)
    out_w = max((w - window) // stride + 1, 1)
    feats, coords = extract_patch_features(gray, window, stride)
    if len(feats) < 10:
        return np.zeros((out_h, out_w), dtype=np.float32)
    model = IsolationForest(contamination="auto", random_state=0)
    model.fit(feats)
    raw_scores = -model.decision_function(feats)  # higher = more anomalous
    grid = np.zeros((out_h, out_w), dtype=np.float32)
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            if idx < len(raw_scores):
                grid[i, j] = raw_scores[idx]
            idx += 1
    return grid


# ============================================================
# V5.0 — fusion layer: one shared grid, one consistency score
# ============================================================

def edge_density_map(gray, window=16, stride=8, canny_low=50, canny_high=150):
    """Fraction of edge pixels per patch. Used to DOWN-WEIGHT texture-heavy
    patches (window blinds, floor grooves, railings) in the fused score,
    since those naturally produce high local variance / wavelet energy
    without being forensically suspicious — this is the "texture vs noise"
    gap identified in review."""
    gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(gray_u8, canny_low, canny_high)
    h, w = edges.shape
    out_h = max((h - window) // stride + 1, 1)
    out_w = max((w - window) // stride + 1, 1)
    dmap = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            y, x = i * stride, j * stride
            patch = edges[y:y + window, x:x + window]
            dmap[i, j] = (patch > 0).mean()
    return dmap


def patch_reduce(full_map, window=16, stride=8, func=np.mean):
    """Downsample a full-resolution map onto the same patch grid used by
    local_noise_map, via `func` per patch (mean by default). This is what
    lets multiscale_noise_map (full-res) and the PRNU residual (full-res)
    be combined with var_map/ml grid (patch-res) without misalignment."""
    h, w = full_map.shape
    out_h = max((h - window) // stride + 1, 1)
    out_w = max((w - window) // stride + 1, 1)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            y, x = i * stride, j * stride
            patch = full_map[y:y + window, x:x + window]
            out[i, j] = func(patch)
    return out


def zscore_map(m):
    mean, std = m.mean(), m.std()
    if std < 1e-6:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mean) / std).astype(np.float32)


def fuse_noise_consistency(var_map, wave_grid, prnu_var_grid, ml_grid,
                            edge_map, weights=None, edge_penalty=0.6):
    """Combine four independent signals into ONE noise-consistency score:

        z(var_map)   — local residual variance (V1.2/V2.0 signal)
        z(wave_grid) — multiscale wavelet energy (V3.0 signal, now used)
        z(prnu_grid) — PRNU-residual self-consistency (V2.1 signal, now used)
        z(ml_grid)   — IsolationForest continuous anomaly score (V4.0 signal,
                       now continuous instead of a separate hard vote)

    All four are z-scored onto comparable scales, weighted, and summed.
    Edge-dense (textured) patches are then down-weighted, since texture
    inflates variance/wavelet energy without being forensically meaningful.
    One threshold on this single map produces suspicious_regions() — not
    four independent detectors producing four independent box sets.
    """
    if weights is None:
        weights = {"var": 0.30, "wave": 0.25, "prnu": 0.20, "ml": 0.25}

    z_var = zscore_map(var_map)
    z_wave = zscore_map(wave_grid)
    z_prnu = zscore_map(prnu_var_grid)
    z_ml = zscore_map(ml_grid)

    fused = (weights["var"] * z_var +
             weights["wave"] * z_wave +
             weights["prnu"] * z_prnu +
             weights["ml"] * z_ml)

    fused = fused * (1.0 - edge_penalty * np.clip(edge_map, 0, 1))
    return fused.astype(np.float32)


def threshold_consistency_map(fused_map, z_thresh=1.5):
    """Single decision point for the whole pipeline: one threshold, on the
    one fused map, produces the one suspicious-region mask.

    Two-sided on purpose: tampering can show up as MORE local noise
    (spliced-in content from a noisier source) or LESS local noise
    (a smoothed/blurred patch pasted in, or over-denoised region) — both
    are inconsistent with the rest of the image. A one-sided threshold
    would miss the smoothing case, which is arguably the more common
    tamper signature in practice."""
    mean, std = fused_map.mean(), fused_map.std()
    if std < 1e-6:
        return np.zeros_like(fused_map, dtype=bool)
    z = (fused_map - mean) / std
    return np.abs(z) > z_thresh


# ============================================================
# Pipeline orchestration
# ============================================================

def analyze(path, version="v5.0", window=16, stride=8):
    img = load_image(path)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = {"path": path, "version": version, "original": img}

    denoised = denoise_adaptive(img)
    residual = compute_residual(img, denoised)
    result["residual_vis"] = visualize_residual(residual)
    result["stats"] = noise_statistics(residual)
    result["heatmap"] = noise_heatmap(residual)

    var_map = local_noise_map(residual, window=window, stride=stride)
    result["local_var_map"] = var_map
    sy = img.shape[0] / var_map.shape[0]
    sx = img.shape[1] / var_map.shape[1]

    # --- legacy independent versions, kept for comparison/debugging ---
    if version in ("v2.0", "v2.1", "v3.0", "v4.0"):
        mask = sensor_noise_consistency(var_map)
        boxes = suspicious_regions(mask)
        result["suspicious_boxes"] = boxes
        result["suspicious_vis"] = draw_suspicious_regions(img, boxes, sx, sy)

    if version in ("v2.1", "v3.0", "v4.0"):
        result["prnu"] = prnu_extract(img)

    if version in ("v3.0", "v4.0"):
        result["multiscale_map"] = multiscale_noise_map(gray_full)

    if version == "v4.0":
        ml_mask, ml_boxes = ml_noise_consistency(gray_full, window=window, stride=stride)
        result["ml_mask"] = ml_mask
        result["ml_boxes"] = ml_boxes

    # --- V5.0: the actual fused pipeline ---
    if version == "v5.0":
        prnu_residual = prnu_extract(img)
        multiscale_map = multiscale_noise_map(gray_full)

        wave_grid = patch_reduce(multiscale_map, window, stride, func=np.mean)
        prnu_var_grid = local_noise_map(prnu_residual, window=window, stride=stride)
        ml_grid = ml_anomaly_score_map(gray_full, window=window, stride=stride)
        edge_map = edge_density_map(gray_full, window=window, stride=stride)

        fused = fuse_noise_consistency(var_map, wave_grid, prnu_var_grid, ml_grid, edge_map)
        fused_mask = threshold_consistency_map(fused)
        boxes = suspicious_regions(fused_mask)

        result["prnu_self_consistency"] = prnu_residual  # see docstring: not camera ID
        result["multiscale_map"] = multiscale_map
        result["edge_density_map"] = edge_map
        result["ml_score_map"] = ml_grid
        result["fused_score_map"] = fused
        result["suspicious_boxes"] = boxes
        result["suspicious_vis"] = draw_suspicious_regions(img, boxes, sx, sy)

    return result


# ============================================================
# VISUALIZATION (popup window, no files written)
# ============================================================

def show_results(results):
    has_fusion = "fused_score_map" in results
    n_panels = 4 if has_fusion else 3

    plt.figure("Noise Analysis", figsize=(5 * n_panels, 5))
    plt.clf()

    plt.subplot(1, n_panels, 1)
    plt.imshow(cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, n_panels, 2)
    plt.imshow(results["residual_vis"])
    plt.title("Residual")
    plt.axis("off")

    if has_fusion:
        plt.subplot(1, n_panels, 3)
        fused = results["fused_score_map"]
        vmax = np.percentile(np.abs(fused), 99) or 1.0
        plt.imshow(fused, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.title("Fused Noise-Consistency Score")
        plt.axis("off")

        plt.subplot(1, n_panels, 4)
        plt.imshow(cv2.cvtColor(results["suspicious_vis"], cv2.COLOR_BGR2RGB))
        plt.title(f"Suspicious Regions ({len(results['suspicious_boxes'])})")
        plt.axis("off")
    else:
        plt.subplot(1, n_panels, 3)
        if "suspicious_vis" in results:
            plt.imshow(cv2.cvtColor(results["suspicious_vis"], cv2.COLOR_BGR2RGB))
            plt.title("Suspicious Regions")
        else:
            plt.imshow(cv2.cvtColor(results["heatmap"], cv2.COLOR_BGR2RGB))
            plt.title("Heatmap")
        plt.axis("off")

    plt.tight_layout()
    plt.show(block=True)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python noise.py <image_path> [version]")
        sys.exit(1)

    image_path = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else "v5.0"

    results = analyze(image_path, version)

    print()
    print("=" * 50)
    print("NOISE ANALYSIS")
    print("=" * 50)
    print()

    print("Statistics:")
    for k, v in results["stats"].items():
        print(f"{k:>8}: {v:.3f}")

    if "suspicious_boxes" in results:
        label = "Fused suspicious regions" if version == "v5.0" else "Suspicious regions"
        print(f"\n{label}:", len(results["suspicious_boxes"]))

    if "ml_boxes" in results:
        print("ML-flagged regions (legacy, independent):", len(results["ml_boxes"]))

    print("Opening visualization...")
    show_results(results)