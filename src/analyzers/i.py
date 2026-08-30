# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# class IlluminationAnalyzer:

#     def __init__(
#         self,
#         blur_size=41,
#         shadow_percentile=20,
#         canny_low=40,
#         canny_high=120,
#         morph_kernel_size=5,
#         hough_threshold=60,
#         min_line_length=60,
#         max_line_gap=10,
#     ):

#         self.blur_size = blur_size
#         self.shadow_percentile = shadow_percentile

#         self.canny_low = canny_low
#         self.canny_high = canny_high

#         self.morph_kernel_size = morph_kernel_size

#         self.hough_threshold = hough_threshold
#         self.min_line_length = min_line_length
#         self.max_line_gap = max_line_gap

#     # =====================================================
#     # IMAGE PROCESSING
#     # =====================================================

#     def load_image(self, image_path):

#         image = cv2.imread(image_path)

#         if image is None:
#             raise FileNotFoundError(image_path)

#         return image

#     def preprocess(self, image):

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         gray = cv2.GaussianBlur(
#             gray,
#             (5, 5),
#             0
#         )

#         return gray

#     # =====================================================
#     # ILLUMINATION MAP
#     # =====================================================

#     def estimate_illumination_map(self, gray):

#         illumination = cv2.GaussianBlur(
#             gray,
#             (self.blur_size, self.blur_size),
#             0
#         )

#         return illumination

#     # =====================================================
#     # SHADOW CANDIDATES
#     # =====================================================

#     def detect_shadow_candidates(self, illumination):

#         threshold = np.percentile(
#             illumination,
#             self.shadow_percentile
#         )

#         mask = np.zeros_like(
#             illumination,
#             dtype=np.uint8
#         )

#         mask[illumination <= threshold] = 255

#         return mask

#     # =====================================================
#     # CLEANUP
#     # =====================================================

#     def clean_shadow_mask(self, shadow_mask):

#         kernel = np.ones(

#             (
#                 self.morph_kernel_size,
#                 self.morph_kernel_size
#             ),

#             np.uint8

#         )

#         cleaned = cv2.morphologyEx(
#             shadow_mask,
#             cv2.MORPH_OPEN,
#             kernel
#         )

#         cleaned = cv2.morphologyEx(
#             cleaned,
#             cv2.MORPH_CLOSE,
#             kernel
#         )

#         return cleaned

#     # =====================================================
#     # SHADOW EDGES
#     # =====================================================

#     def detect_shadow_edges(self, shadow_mask):

#         edges = cv2.Canny(
#             shadow_mask,
#             self.canny_low,
#             self.canny_high
#         )

#         return edges

#     # =====================================================
#     # SHADOW LINES
#     # =====================================================

#     def detect_shadow_lines(self, edges):

#         lines = cv2.HoughLinesP(
#             edges,
#             rho=1,
#             theta=np.pi / 180,
#             threshold=self.hough_threshold,
#             minLineLength=self.min_line_length,
#             maxLineGap=self.max_line_gap
#         )

#         if lines is None:
#             return np.empty((0, 4), dtype=int)

#         return np.asarray(lines).reshape(-1, 4)

#     # =====================================================
#     # LINE UTILITIES
#     # =====================================================

#     def line_angle(self, line):

#         x1, y1, x2, y2 = line

#         angle = np.degrees(

#             np.arctan2(
#                 y2 - y1,
#                 x2 - x1
#             )

#         )

#         return angle % 180

#     def line_length(self, line):

#         x1, y1, x2, y2 = line

#         return np.hypot(
#             x2 - x1,
#             y2 - y1
#         )
        
#     # =====================================================
#     # LINE MERGING
#     # =====================================================

#     def merge_duplicate_lines(
#         self,
#         lines,
#         angle_tol=5.0,
#         distance_tol=20.0,
#     ):

#         if len(lines) == 0:
#             return lines

#         used = np.zeros(len(lines), dtype=bool)
#         merged = []

#         angles = np.array([self.line_angle(l) for l in lines])

#         midpoints = np.array([
#             (
#                 (l[0] + l[2]) / 2,
#                 (l[1] + l[3]) / 2
#             )
#             for l in lines
#         ])

#         for i in range(len(lines)):

#             if used[i]:
#                 continue

#             group = [i]
#             used[i] = True

#             for j in range(i + 1, len(lines)):

#                 if used[j]:
#                     continue

#                 angle_diff = min(
#                     abs(angles[i] - angles[j]),
#                     180 - abs(angles[i] - angles[j])
#                 )

#                 distance = np.linalg.norm(
#                     midpoints[i] - midpoints[j]
#                 )

#                 if (
#                     angle_diff < angle_tol and
#                     distance < distance_tol
#                 ):
#                     group.append(j)
#                     used[j] = True

#             group_lines = lines[group]

#             lengths = np.array([
#                 self.line_length(l)
#                 for l in group_lines
#             ])

#             merged.append(
#                 group_lines[np.argmax(lengths)]
#             )

#         return np.array(merged)

#     # =====================================================
#     # SHADOW DIRECTION
#     # =====================================================

#     def estimate_dominant_shadow_direction(self, lines):

#         if len(lines) == 0:
#             return None

#         angles = np.array([
#             self.line_angle(l)
#             for l in lines
#         ])

#         lengths = np.array([
#             self.line_length(l)
#             for l in lines
#         ])

#         x = np.cos(np.deg2rad(angles))
#         y = np.sin(np.deg2rad(angles))

#         vx = np.average(x, weights=lengths)
#         vy = np.average(y, weights=lengths)

#         angle = np.degrees(
#             np.arctan2(vy, vx)
#         )

#         return angle % 180

#     # =====================================================
#     # LIGHT DIRECTION
#     # =====================================================

#     def estimate_light_direction(self, shadow_angle):

#         if shadow_angle is None:
#             return None

#         return (shadow_angle + 180) % 360

#     # =====================================================
#     # VISUALIZATION
#     # =====================================================

#     def draw_shadow_lines(
#         self,
#         image,
#         lines
#     ):

#         output = image.copy()

#         for line in lines:

#             x1, y1, x2, y2 = line

#             cv2.line(
#                 output,
#                 (x1, y1),
#                 (x2, y2),
#                 (0, 255, 0),
#                 2
#             )

#         return output

#     def draw_light_direction(
#         self,
#         image,
#         light_angle
#     ):

#         output = image.copy()

#         if light_angle is None:
#             return output

#         h, w = output.shape[:2]

#         cx = w // 2
#         cy = h // 2

#         length = int(
#             min(h, w) * 0.25
#         )

#         theta = np.deg2rad(light_angle)

#         x2 = int(
#             cx + length * np.cos(theta)
#         )

#         y2 = int(
#             cy + length * np.sin(theta)
#         )

#         cv2.arrowedLine(
#             output,
#             (cx, cy),
#             (x2, y2),
#             (0, 0, 255),
#             4,
#             tipLength=0.15
#         )

#         return output

#     # =====================================================
#     # MAIN PIPELINE
#     # =====================================================

#     def analyze(self, image_path):

#         image = self.load_image(image_path)

#         gray = self.preprocess(image)

#         illumination = self.estimate_illumination_map(gray)

#         shadow_mask = self.detect_shadow_candidates(
#             illumination
#         )

#         shadow_mask = self.clean_shadow_mask(
#             shadow_mask
#         )

#         shadow_edges = self.detect_shadow_edges(
#             shadow_mask
#         )

#         shadow_lines = self.detect_shadow_lines(
#             shadow_edges
#         )

#         shadow_lines = self.merge_duplicate_lines(
#             shadow_lines
#         )

#         shadow_angle = self.estimate_dominant_shadow_direction(
#             shadow_lines
#         )

#         light_angle = self.estimate_light_direction(
#             shadow_angle
#         )

#         line_image = self.draw_shadow_lines(
#             image,
#             shadow_lines
#         )

#         direction_image = self.draw_light_direction(
#             line_image,
#             light_angle
#         )

#         return {

#             "original": image,

#             "gray": gray,

#             "illumination": illumination,

#             "shadow_mask": shadow_mask,

#             "shadow_edges": shadow_edges,

#             "shadow_lines_image": line_image,

#             "light_direction_image": direction_image,

#             "shadow_lines": shadow_lines,

#             "shadow_angle": shadow_angle,

#             "light_angle": light_angle,

#             "num_shadow_lines": len(shadow_lines),

#             "shadow_candidate_pixels": int(
#                 np.count_nonzero(shadow_mask)
#             ),

#             "shadow_edge_pixels": int(
#                 np.count_nonzero(shadow_edges)
#             ),

#             "illumination_min": int(
#                 illumination.min()
#             ),

#             "illumination_max": int(
#                 illumination.max()
#             ),
#         }
        
        
# # =====================================================
# # VISUALIZATION
# # =====================================================

# def show_results(results, save_path=None):

#     plt.figure(figsize=(24, 6))

#     plt.subplot(1, 6, 1)
#     plt.imshow(cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB))
#     plt.title("Original")
#     plt.axis("off")

#     plt.subplot(1, 6, 2)
#     plt.imshow(results["gray"], cmap="gray")
#     plt.title("Grayscale")
#     plt.axis("off")

#     plt.subplot(1, 6, 3)
#     plt.imshow(results["illumination"], cmap="gray")
#     plt.title("Illumination")
#     plt.axis("off")

#     plt.subplot(1, 6, 4)
#     plt.imshow(results["shadow_mask"], cmap="gray")
#     plt.title("Shadow Mask")
#     plt.axis("off")

#     plt.subplot(1, 6, 5)
#     plt.imshow(results["shadow_edges"], cmap="gray")
#     plt.title("Shadow Edges")
#     plt.axis("off")

#     plt.subplot(1, 6, 6)
#     plt.imshow(cv2.cvtColor(results["light_direction_image"], cv2.COLOR_BGR2RGB))
#     plt.title("Light Direction")
#     plt.axis("off")

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=120, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()


# # =====================================================
# # STANDALONE TEST
# # =====================================================

# if __name__ == "__main__":

#     import sys

#     image_path = (
#         sys.argv[1]
#         if len(sys.argv) > 1
#         else "../data/raw/shadows/Man_Modern_Window.jpeg"
#     )

#     analyzer = IlluminationAnalyzer()

#     results = analyzer.analyze(image_path)

#     print()
#     print("=" * 50)
#     print("ILLUMINATION ANALYSIS")
#     print("=" * 50)
#     print()

#     print(f"Shadow candidates : {results['shadow_candidate_pixels']}")
#     print(f"Shadow edges      : {results['shadow_edge_pixels']}")
#     print(f"Shadow lines      : {results['num_shadow_lines']}")

#     if results["shadow_angle"] is None:
#         print("Dominant shadow   : Not detected")
#     else:
#         print(f"Dominant shadow   : {results['shadow_angle']:.2f}°")

#     if results["light_angle"] is None:
#         print("Light direction   : Not detected")
#     else:
#         print(f"Light direction   : {results['light_angle']:.2f}°")

#     print(
#         f"Illumination      : "
#         f"{results['illumination_min']} - {results['illumination_max']}"
#     )

#     show_results(results)

import cv2
import numpy as np
import matplotlib.pyplot as plt


class IlluminationAnalyzer:
    """
    V2 pipeline:

        Image
          |
          v
        LAB -> L channel (illumination proxy)
          |
          v
        Local illumination estimate (large-kernel blur)
          |
          v
        Illumination discontinuity map  (L - local_mean, i.e. "how much
        darker is this pixel than its own neighborhood")
          |
          v
        Adaptive shadow threshold (local, NOT global percentile)
          |
          v
        Shape/geometry filtering  (drop thin/elongated/border-hugging
        blobs -> these are window frames, pillars, walls, not shadows)
          |
          v
        [optional] Hue-invariance filter (shadows preserve hue, material
        edges usually don't)
          |
          v
        Shadow boundary extraction -> Hough lines -> dominant direction
        -> light direction (best-effort, secondary to mask quality)

    V1 bug this fixes: "dark == shadow" (global percentile threshold)
    incorrectly classified an intrinsically dark wall as a shadow region,
    and Hough picked up window-frame edges as if they were shadow
    boundaries. Both are fixed by working in *local contrast* space
    instead of *global brightness* space, plus explicit shape filtering.
    """

    def __init__(
        self,
        # local illumination estimation
        illum_blur_size=61,
        # adaptive threshold for shadow discontinuity
        adaptive_block_size=51,
        adaptive_C=8,
        # morphology cleanup
        morph_kernel_size=5,
        min_component_area=150,
        # shape filtering (rejects thin frame/pillar edges)
        max_aspect_ratio=6.0,
        min_extent=0.15,
        max_border_touch_frac=0.6,
        # optional hue-invariance filter
        use_hue_filter=False,
        hue_tol=12,
        # edge / line detection
        canny_low=40,
        canny_high=120,
        hough_threshold=40,
        min_line_length=30,
        max_line_gap=10,
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

        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    # =====================================================
    # IMAGE PROCESSING
    # =====================================================

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        return image

    def preprocess(self, image):
        """Return (L channel from LAB, full HSV image)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        L = cv2.GaussianBlur(L, (5, 5), 0)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        return L, hsv

    # =====================================================
    # LOCAL ILLUMINATION / DISCONTINUITY MAP
    # =====================================================

    def estimate_local_illumination(self, L):
        """Large-kernel blur = 'what brightness would we expect here
        if there were no shadow' (local background estimate)."""
        k = self.illum_blur_size
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(L, (k, k), 0)

    def compute_discontinuity_map(self, L, local_illum):
        """Positive values = pixel darker than its local neighborhood.
        This is the quantity that actually corresponds to 'illumination
        suddenly decreasing here', not raw darkness."""
        diff = local_illum.astype(np.float32) - L.astype(np.float32)
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        return diff

    # =====================================================
    # SHADOW CANDIDATES (ADAPTIVE, LOCAL)
    # =====================================================

    def detect_shadow_candidates(self, L):
        """Adaptive threshold on L itself: flags pixels that are
        significantly darker than their own local neighborhood mean.
        This is local/relative, unlike a global percentile cut, so a
        uniformly dark wall (locally flat) will NOT be flagged, while a
        real shadow edge on a lit floor (local contrast) will be."""
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
    # SHAPE FILTERING (rejects window frames / pillars / wall strips)
    # =====================================================

    def filter_shadow_components(self, mask):
        """Keep only connected components that look like plausible cast
        shadows (blobby, not thin bars; not hugging the image border
        along their whole length, which is typical of walls/frames)."""
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
                # thin bar -> almost certainly a frame/pillar edge, not
                # a cast shadow blob
                continue

            extent = area / float(cw * ch)
            if extent < self.min_extent:
                # very sparse/skeleton-like region within its bbox
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
                # runs along the image edge -> likely wall/structure,
                # not a shadow cast on a surface
                continue

            keep[component_mask] = 255

        return keep

    # =====================================================
    # OPTIONAL: HUE-INVARIANCE FILTER
    # =====================================================

    def apply_hue_filter(self, mask, hsv, local_illum):
        """Cast shadows mostly reduce V while preserving H. Material
        edges (wall/floor boundary, frame edge) usually shift H/S too.
        This keeps only mask pixels whose local hue is consistent with
        the surrounding (non-masked) hue -- a weak but useful extra cue.
        Disabled by default since it needs reasonably clean color data."""
        H = hsv[:, :, 0].astype(np.float32)

        k = self.illum_blur_size
        if k % 2 == 0:
            k += 1
        H_local_mean = cv2.GaussianBlur(H, (k, k), 0)

        hue_diff = np.abs(H - H_local_mean)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # circular

        hue_ok = hue_diff <= self.hue_tol
        filtered = mask.copy()
        filtered[~hue_ok] = 0
        return filtered

    # =====================================================
    # SHADOW EDGES / LINES (unchanged in spirit, now fed a clean mask)
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
    # SHADOW / LIGHT DIRECTION (secondary, best-effort)
    # =====================================================

    def estimate_dominant_shadow_direction(self, lines):
        if len(lines) == 0:
            return None

        angles = np.array([self.line_angle(l) for l in lines])
        lengths = np.array([self.line_length(l) for l in lines])

        x = np.cos(np.deg2rad(angles))
        y = np.sin(np.deg2rad(angles))

        vx = np.average(x, weights=lengths)
        vy = np.average(y, weights=lengths)

        return np.degrees(np.arctan2(vy, vx)) % 180

    def estimate_light_direction(self, shadow_angle):
        if shadow_angle is None:
            return None
        return (shadow_angle + 180) % 360

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
        shadow_mask = self.clean_shadow_mask(shadow_mask)
        shadow_mask = self.filter_shadow_components(shadow_mask)

        if self.use_hue_filter:
            shadow_mask = self.apply_hue_filter(shadow_mask, hsv, local_illum)
            shadow_mask = self.clean_shadow_mask(shadow_mask)

        shadow_edges = self.detect_shadow_edges(shadow_mask)
        shadow_lines = self.detect_shadow_lines(shadow_edges)
        shadow_lines = self.merge_duplicate_lines(shadow_lines)

        shadow_angle = self.estimate_dominant_shadow_direction(shadow_lines)
        light_angle = self.estimate_light_direction(shadow_angle)

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

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../data/raw/shadows/Man_Modern_Window.jpeg"
    )

    analyzer = IlluminationAnalyzer()
    results = analyzer.analyze(image_path)

    print()
    print("=" * 50)
    print("ILLUMINATION ANALYSIS (v2)")
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

    show_results(results, save_path="v2_result.png")