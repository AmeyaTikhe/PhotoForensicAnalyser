# Final Iteration

import cv2
import numpy as np
import matplotlib.pyplot as plt


class IlluminationAnalyzer:
    """
    Pipeline: L channel -> local illumination (blur) -> discontinuity
    map -> adaptive shadow threshold -> shape filtering -> Hough lines
    -> dominant axis -> light direction (via brightness comparison).
    """


    def __init__(
        self,
        illum_blur_size=61,
        adaptive_block_size=51,
        adaptive_C=8,
        morph_kernel_size=5,
        min_component_area=150,
        max_aspect_ratio=6.0,
        min_extent=0.15,
        max_border_touch_frac=0.6,
        use_hue_filter=False,
        hue_tol=12,
        floor_roi_top_frac=0.55,
        canny_low=40,
        canny_high=120,
        hough_threshold=40,
        min_line_length=30,
        max_line_gap=10,
        # fraction of image diagonal used as sampling distance/patch
        # size when disambiguating light direction
        light_sample_frac=0.06,
    ):
        self.illum_blur_size = illum_blur_size

        self.adaptive_block_size = adaptive_block_size
        self.adaptive_C = adaptive_C

        self.morph_kernel_size = morph_kernel_size
        self.min_component_area = min_component_area

        self.max_aspect_ratio = max_aspect_ratio
        self.min_extent = min_extent
        self.max_border_touch_frac = max_border_touch_frac

        self.use_hue_filter = use_hue_filter
        self.hue_tol = hue_tol

        self.floor_roi_top_frac = floor_roi_top_frac

        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        self.light_sample_frac = light_sample_frac

    # =====================================================
    # IMAGE PROCESSING
    # =====================================================

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        return image

    def preprocess(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        L = cv2.GaussianBlur(L, (5, 5), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return L, hsv

    # =====================================================
    # LOCAL ILLUMINATION / DISCONTINUITY MAP
    # =====================================================

    def estimate_local_illumination(self, L):
        k = self.illum_blur_size
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(L, (k, k), 0)

    def compute_discontinuity_map(self, L, local_illum):
        diff = local_illum.astype(np.float32) - L.astype(np.float32)
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        return diff

    # =====================================================
    # FLOOR ROI GATING
    # =====================================================

    def build_floor_roi_mask(self, shape):
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        top = int(h * self.floor_roi_top_frac)
        mask[top:, :] = 255
        return mask

    # =====================================================
    # SHADOW CANDIDATES (ADAPTIVE, LOCAL)
    # =====================================================

    def detect_shadow_candidates(self, L):
        block = self.adaptive_block_size
        if block % 2 == 0:
            block += 1

        mask = cv2.adaptiveThreshold(
            L,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            self.adaptive_C,
        )
        return mask

    # =====================================================
    # CLEANUP
    # =====================================================

    def clean_shadow_mask(self, shadow_mask):
        kernel = np.ones(
            (self.morph_kernel_size, self.morph_kernel_size), np.uint8
        )
        cleaned = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned

    # =====================================================
    # SHAPE FILTERING
    # =====================================================

    def filter_shadow_components(self, mask):
        h, w = mask.shape
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        keep = np.zeros_like(mask)

        for label in range(1, n_labels):
            x, y, cw, ch, area = stats[label]

            if area < self.min_component_area:
                continue

            long_side = max(cw, ch)
            short_side = max(min(cw, ch), 1)
            aspect_ratio = long_side / short_side

            if aspect_ratio > self.max_aspect_ratio:
                continue

            extent = area / float(cw * ch)
            if extent < self.min_extent:
                continue

            component_mask = labels == label
            border_pixels = (
                np.count_nonzero(component_mask[0, :])
                + np.count_nonzero(component_mask[-1, :])
                + np.count_nonzero(component_mask[:, 0])
                + np.count_nonzero(component_mask[:, -1])
            )
            perimeter_est = 2 * (cw + ch)
            border_frac = border_pixels / max(perimeter_est, 1)
            if border_frac > self.max_border_touch_frac:
                continue

            keep[component_mask] = 255

        return keep

    # =====================================================
    # OPTIONAL: HUE-INVARIANCE FILTER
    # =====================================================

    def apply_hue_filter(self, mask, hsv, local_illum):
        H = hsv[:, :, 0].astype(np.float32)

        k = self.illum_blur_size
        if k % 2 == 0:
            k += 1
        H_local_mean = cv2.GaussianBlur(H, (k, k), 0)

        hue_diff = np.abs(H - H_local_mean)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)

        hue_ok = hue_diff <= self.hue_tol
        filtered = mask.copy()
        filtered[~hue_ok] = 0
        return filtered

    # =====================================================
    # SHADOW EDGES / LINES
    # =====================================================

    def detect_shadow_edges(self, shadow_mask):
        return cv2.Canny(shadow_mask, self.canny_low, self.canny_high)

    def detect_shadow_lines(self, edges):
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if lines is None:
            return np.empty((0, 4), dtype=int)
        return np.asarray(lines).reshape(-1, 4)

    # =====================================================
    # LINE UTILITIES
    # =====================================================

    def line_angle(self, line):
        """Undirected line orientation in [0, 180)."""
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return angle % 180

    def line_length(self, line):
        x1, y1, x2, y2 = line
        return np.hypot(x2 - x1, y2 - y1)

    def merge_duplicate_lines(self, lines, angle_tol=5.0, distance_tol=20.0):
        if len(lines) == 0:
            return lines

        used = np.zeros(len(lines), dtype=bool)
        merged = []

        angles = np.array([self.line_angle(l) for l in lines])
        midpoints = np.array(
            [((l[0] + l[2]) / 2, (l[1] + l[3]) / 2) for l in lines]
        )

        for i in range(len(lines)):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue
                angle_diff = min(
                    abs(angles[i] - angles[j]),
                    180 - abs(angles[i] - angles[j]),
                )
                distance = np.linalg.norm(midpoints[i] - midpoints[j])
                if angle_diff < angle_tol and distance < distance_tol:
                    group.append(j)
                    used[j] = True

            group_lines = lines[group]
            lengths = np.array([self.line_length(l) for l in group_lines])
            merged.append(group_lines[np.argmax(lengths)])

        return np.array(merged)

    # =====================================================
    # SHADOW / LIGHT DIRECTION  (FIXED)
    # =====================================================

    def estimate_dominant_shadow_direction(self, lines):
        """Circular mean of axial (mod-180) line orientations, using
        the double-angle trick so opposite-looking but parallel lines
        don't cancel out. Weighted by length^2 so long, reliable lines
        dominate over short noisy ones."""
        if len(lines) == 0:
            return None

        angles = np.array([self.line_angle(l) for l in lines])
        lengths = np.array([self.line_length(l) for l in lines])
        weights = lengths ** 2

        theta2 = np.deg2rad(2.0 * angles)
        vx = np.average(np.cos(theta2), weights=weights)
        vy = np.average(np.sin(theta2), weights=weights)

        mean_angle = np.degrees(np.arctan2(vy, vx)) / 2.0
        return mean_angle % 180

    def estimate_dominant_shadow_direction_from_mask(self, shadow_mask, min_elongation=1.8):
        """PCA orientation of each blob's own pixels, instead of
        Canny+Hough on the mask. Canny on a filled blob traces its
        outline (top/bottom/left/right), so axis-aligned end-caps and
        blocky noise contaminate the angle just as much as the true
        diagonal streak edges do. PCA on the pixel coordinates gives
        each blob's actual long axis directly, weighted by area and
        how elongated (line-like) it is; round/blocky blobs are
        skipped since they carry no reliable direction."""
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            shadow_mask, connectivity=8
        )

        angles, weights = [], []

        for label in range(1, n_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_component_area:
                continue

            ys, xs = np.nonzero(labels == label)
            pts = np.stack([xs, ys], axis=1).astype(np.float32)

            _, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
            elongation = eigenvalues[0, 0] / (eigenvalues[1, 0] + 1e-6)
            if elongation < min_elongation:
                continue

            vx, vy = eigenvectors[0]
            angles.append(np.degrees(np.arctan2(vy, vx)) % 180)
            weights.append(area * elongation)

        if not angles:
            return None

        angles = np.array(angles)
        weights = np.array(weights)

        theta2 = np.deg2rad(2.0 * angles)
        vx = np.average(np.cos(theta2), weights=weights)
        vy = np.average(np.sin(theta2), weights=weights)

        return (np.degrees(np.arctan2(vy, vx)) / 2.0) % 180

    def estimate_light_direction(self, shadow_angle, shadow_mask=None, local_illum=None):
        """Pick whichever of the two directions along the shadow axis
        is brighter, by comparing average illumination in a patch on
        each side of the mask's centroid. Sample distance/patch size
        scale with image size so this isn't dominated by local noise."""
        if shadow_angle is None:
            return None

        if shadow_mask is None or local_illum is None:
            return shadow_angle % 360

        ys, xs = np.nonzero(shadow_mask)
        if len(xs) == 0:
            return shadow_angle % 360

        h, w = local_illum.shape[:2]
        diag = np.hypot(h, w)
        step = max(int(diag * self.light_sample_frac), 10)
        half_patch = max(step // 2, 3)

        cx, cy = xs.mean(), ys.mean()
        theta = np.deg2rad(shadow_angle)
        dx, dy = np.cos(theta), np.sin(theta)

        def patch_mean(sign):
            x = int(np.clip(cx + sign * step * dx, half_patch, w - 1 - half_patch))
            y = int(np.clip(cy + sign * step * dy, half_patch, h - 1 - half_patch))
            patch = local_illum[y - half_patch:y + half_patch, x - half_patch:x + half_patch]
            return float(patch.mean())

        bright_forward = patch_mean(+1)
        bright_backward = patch_mean(-1)

        return shadow_angle if bright_forward >= bright_backward else (shadow_angle + 180) % 360

    # =====================================================
    # VISUALIZATION
    # =====================================================

    def draw_shadow_lines(self, image, lines):
        output = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return output

    def draw_light_direction(self, image, light_angle):
        output = image.copy()
        if light_angle is None:
            return output

        h, w = output.shape[:2]
        cx, cy = w // 2, h // 2
        length = int(min(h, w) * 0.25)
        theta = np.deg2rad(light_angle)
        x2 = int(cx + length * np.cos(theta))
        y2 = int(cy + length * np.sin(theta))

        cv2.arrowedLine(
            output, (cx, cy), (x2, y2), (0, 0, 255), 4, tipLength=0.15
        )
        return output

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def analyze(self, image_path):
        image = self.load_image(image_path)

        L, hsv = self.preprocess(image)

        local_illum = self.estimate_local_illumination(L)
        discontinuity_map = self.compute_discontinuity_map(L, local_illum)

        shadow_mask = self.detect_shadow_candidates(L)

        floor_roi = self.build_floor_roi_mask(L.shape)
        shadow_mask = cv2.bitwise_and(shadow_mask, floor_roi)

        shadow_mask = self.clean_shadow_mask(shadow_mask)
        shadow_mask = self.filter_shadow_components(shadow_mask)

        if self.use_hue_filter:
            shadow_mask = self.apply_hue_filter(shadow_mask, hsv, local_illum)
            shadow_mask = self.clean_shadow_mask(shadow_mask)

        shadow_edges = self.detect_shadow_edges(shadow_mask)
        shadow_lines = self.detect_shadow_lines(shadow_edges)
        shadow_lines = self.merge_duplicate_lines(shadow_lines)

        shadow_angle = self.estimate_dominant_shadow_direction_from_mask(shadow_mask)
        light_angle = self.estimate_light_direction(
            shadow_angle, shadow_mask=shadow_mask, local_illum=local_illum
        )

        line_image = self.draw_shadow_lines(image, shadow_lines)
        direction_image = self.draw_light_direction(line_image, light_angle)

        return {
            "original": image,
            "L_channel": L,
            "local_illumination": local_illum,
            "discontinuity_map": discontinuity_map,
            "shadow_mask": shadow_mask,
            "shadow_edges": shadow_edges,
            "shadow_lines_image": line_image,
            "light_direction_image": direction_image,
            "shadow_lines": shadow_lines,
            "shadow_angle": shadow_angle,
            "light_angle": light_angle,
            "num_shadow_lines": len(shadow_lines),
            "shadow_candidate_pixels": int(np.count_nonzero(shadow_mask)),
            "shadow_edge_pixels": int(np.count_nonzero(shadow_edges)),
            "illumination_min": int(local_illum.min()),
            "illumination_max": int(local_illum.max()),
        }


# =====================================================
# VISUALIZATION
# =====================================================

def show_results(results, save_path=None):
    plt.figure(figsize=(24, 6))

    panels = [
        ("Original", cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB), None),
        ("L channel", results["L_channel"], "gray"),
        ("Local illumination", results["local_illumination"], "gray"),
        ("Discontinuity map", results["discontinuity_map"], "gray"),
        ("Shadow mask (filtered)", results["shadow_mask"], "gray"),
        ("Light direction", cv2.cvtColor(results["light_direction_image"], cv2.COLOR_BGR2RGB), None),
    ]

    for i, (title, img, cmap) in enumerate(panels, start=1):
        plt.subplot(1, len(panels), i)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =====================================================
# STANDALONE TEST
# =====================================================

if __name__ == "__main__":
    import sys

    # matches main.py's IMAGE_PATH; run from the project root
    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/shadows/Man_Modern_Window.jpeg"
    )

    analyzer = IlluminationAnalyzer()
    results = analyzer.analyze(image_path)

    print()
    print("=" * 50)
    print("ILLUMINATION ANALYSIS")
    print("=" * 50)
    print()
    print(f"Shadow candidates : {results['shadow_candidate_pixels']}")
    print(f"Shadow edges      : {results['shadow_edge_pixels']}")
    print(f"Shadow lines      : {results['num_shadow_lines']}")

    if results["shadow_angle"] is None:
        print("Dominant shadow   : Not detected")
    else:
        print(f"Dominant shadow   : {results['shadow_angle']:.2f}°")

    if results["light_angle"] is None:
        print("Light direction   : Not detected")
    else:
        print(f"Light direction   : {results['light_angle']:.2f}°")

    show_results(results)