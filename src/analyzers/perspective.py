
"""
src/analyzers/perspective.py

Perspective Analysis Module (V1.1)

Inspired by:
Hany Farid - Photo Forensics:
Perspective, Vanishing Points, and Projective Geometry

Changes from V1:
- Adaptive (median-based) Canny thresholds instead of fixed values,
  so the same analyzer behaves sensibly on both a dark high-contrast
  skyscraper and a bright low-contrast landscape.
- min_line_length / max_line_gap scaled to image diagonal instead of
  fixd pixel counts.
- Duplicate/near-collinear line merging before scoring, so one real
  edge doesn't get counted as 5-10 separate "lines".
- RANSAC-lite vanishing point estimation: picks the intersection with
  the most inlier lines, reports a confidence score, instead of a
  plain unweighted median centroid.
- Scene gating: if detected lines are short/numerous/texture-like
  (grass, foliage, gravel) rather than long structural edges, the
  module reports "not applicable" instead of forcing a fake score.
- Human-readable perspective_interpretation string.

Still left for V2:
- Full Manhattan-world orthogonal VP recovery
- Camera calibration recovery
- DBSCAN/MeanShift clustering (currently: angle-bucket + RANSAC)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


class PerspectiveAnalyzer:

    def __init__(
        self,
        hough_threshold=80,
        angle_threshold=10,
        ransac_iterations=300,
        ransac_inlier_px=6,
        min_lines_for_analysis=6,
        texture_line_density_limit=0.006,
    ):
        self.hough_threshold = hough_threshold
        self.angle_threshold = angle_threshold

        self.ransac_iterations = ransac_iterations
        self.ransac_inlier_px = ransac_inlier_px

        self.min_lines_for_analysis = min_lines_for_analysis
        # lines-per-pixel above which we suspect texture noise
        # rather than real structural edges (tunable)
        self.texture_line_density_limit = texture_line_density_limit

    # =====================================================
    # IMAGE PROCESSING
    # =====================================================

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        return image

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def auto_canny(self, image):
        """
        Otsu-based adaptive Canny thresholding. Plain median-based
        auto-thresholding (Zhu et al.) breaks down on images with a
        skewed brightness histogram -- e.g. a night skyscraper shot
        that's mostly near-black background -- because the median
        sits near zero and drags both thresholds down, flooding the
        result with noise edges. Otsu's threshold on the intensity
        histogram is more robust to that skew.
        """
        high, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low = 0.5 * high
        return cv2.Canny(image, low, high)

    def detect_edges(self, image):
        return self.auto_canny(image)

    def detect_lines(self, edges, image_shape):
        h, w = image_shape[:2]
        diag = np.hypot(h, w)

        # scale requirements to image size instead of fixed pixels.
        # 0.03 was too permissive -- let through short texture/noise
        # segments that a fixed min_line_length=100 used to filter out.
        min_line_length = max(40, int(diag * 0.07))
        max_line_gap = max(5, int(diag * 0.01))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        if lines is None:
            return np.empty((0, 4), dtype=int)

        # Different OpenCV builds have been observed to return
        # HoughLinesP output as (N, 1, 4) vs (N, 4) vs even (N,).
        # Reshaping explicitly to (-1, 4) is robust to all of them,
        # whereas lines[:, 0] silently produces garbage (e.g. just
        # the x1 column) if the array wasn't 3D to begin with.
        lines = np.asarray(lines).reshape(-1, 4)
        return lines

    # =====================================================
    # LINE DEDUPLICATION
    # =====================================================

    def merge_duplicate_lines(self, lines, angle_tol=3.0, offset_tol=None, image_diag=None):
        if offset_tol is None:
            # scale to image size; fixed 15px is too tight on a 4K
            # image and too loose on a small crop
            offset_tol = max(10.0, (image_diag or 1000) * 0.015)
        """
        Collapses near-collinear, nearby line segments (typically
        multiple Hough detections along the same real edge) into a
        single representative line. Without this, a single window
        edge on a skyscraper can register as 5-10 separate "lines"
        and quietly dominate the scoring.
        """
        if len(lines) == 0:
            return lines

        used = np.zeros(len(lines), dtype=bool)
        merged = []

        angles = np.array([self.line_angle(l) for l in lines])
        midpoints = np.array([
            ((l[0] + l[2]) / 2.0, (l[1] + l[3]) / 2.0) for l in lines
        ])

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
                    180 - abs(angles[i] - angles[j])
                )
                offset = np.linalg.norm(midpoints[i] - midpoints[j])

                if angle_diff < angle_tol and offset < offset_tol:
                    group.append(j)
                    used[j] = True

            # keep the longest segment in the group as representative
            group_lines = lines[group]
            lengths = np.hypot(
                group_lines[:, 2] - group_lines[:, 0],
                group_lines[:, 3] - group_lines[:, 1]
            )
            best = group_lines[np.argmax(lengths)]
            merged.append(best)

        return np.array(merged)

    # =====================================================
    # PROJECTIVE GEOMETRY
    # =====================================================

    def line_to_homogeneous(self, line):
        x1, y1, x2, y2 = line
        p1 = np.array([x1, y1, 1])
        p2 = np.array([x2, y2, 1])
        return np.cross(p1, p2)

    def line_angle(self, line):
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return angle % 180

    def intersection(self, l1, l2):
        point = np.cross(l1, l2)
        if abs(point[2]) < 1e-8:
            return None
        return (point[0] / point[2], point[1] / point[2])

    def point_line_distance(self, point, hom_line):
        """Perpendicular distance from a point to a homogeneous line."""
        a, b, c = hom_line
        norm = np.hypot(a, b)
        if norm < 1e-8:
            return np.inf
        x, y = point
        return abs(a * x + b * y + c) / norm

    # =====================================================
    # VANISHING POINT ESTIMATION (sequential multi-VP RANSAC)
    # =====================================================
    #
    # NOTE: earlier versions grouped lines by 2D image-angle first,
    # then searched for a vanishing point within each angle group.
    # That's wrong for converging-line scenes (railway tracks, a
    # symmetric walkway, etc.): the left and right rails have very
    # DIFFERENT image-space angles -- they only agree in that they
    # meet at the same point far away. Angle-bucketing split them
    # into separate groups and never tested them against each other,
    # so the real dominant vanishing point was never found; small
    # same-angle pairs instead produced trivial, meaningless "100%
    # confidence" 2-line groups. Real vanishing point detection has
    # to search across ALL lines regardless of angle.

    def _single_ransac_vp(self, homogeneous_lines, indices, lengths):
        """
        One RANSAC pass: sample line pairs from the given indices,
        take their intersection as a candidate VP, and keep the
        candidate whose inlier lines have the most total PIXEL LENGTH
        (not just the most inlier lines). A handful of long, clean
        structural edges (rails, roofline) should outweigh a cluster
        of short fragmented ones (tree branches, gravel texture) that
        happen to pass near a point by coincidence.
        """
        n = len(indices)
        if n < 2:
            return None, []

        pairs = list(combinations(indices, 2))
        rng = np.random.default_rng(42)
        if len(pairs) > self.ransac_iterations:
            idx = rng.choice(len(pairs), self.ransac_iterations, replace=False)
            pairs = [pairs[i] for i in idx]

        best_point = None
        best_inliers = []
        best_length = -1.0

        for i, j in pairs:
            pt = self.intersection(homogeneous_lines[i], homogeneous_lines[j])
            if pt is None:
                continue

            inliers = [
                k for k in indices
                if self.point_line_distance(pt, homogeneous_lines[k]) < self.ransac_inlier_px
            ]

            total_len = sum(lengths[k] for k in inliers)

            if total_len > best_length:
                best_length = total_len
                best_inliers = inliers
                best_point = pt

        return best_point, best_inliers

    def find_vanishing_points(self, lines, max_vps=5, min_inliers=8):
        """
        Sequentially finds up to max_vps dominant vanishing points
        across the full line set (single-point, two-point, and
        three-point perspective scenes all fall out of this the same
        way). After each VP is found, its inlier lines are removed
        before searching for the next one, so a second real vanishing
        direction (e.g. the other axis on a building facade) doesn't
        get masked by the first.
        """
        if len(lines) < 2:
            return [], np.array([])

        homogeneous_lines = [self.line_to_homogeneous(l) for l in lines]
        lengths = np.hypot(
            lines[:, 2].astype(float) - lines[:, 0],
            lines[:, 3].astype(float) - lines[:, 1]
        )
        remaining = list(range(len(homogeneous_lines)))

        vps = []

        for _ in range(max_vps):
            if len(remaining) < 2:
                break

            point, inlier_idx = self._single_ransac_vp(homogeneous_lines, remaining, lengths)

            if point is None or len(inlier_idx) < min_inliers:
                break

            vps.append({
                "point": point,
                "confidence": len(inlier_idx) / len(remaining),
                "inliers": len(inlier_idx),
                "inlier_length": float(sum(lengths[k] for k in inlier_idx)),
                "total_intersections": len(remaining),
                "line_indices": inlier_idx,
            })

            remaining = [i for i in remaining if i not in set(inlier_idx)]

        return vps, lengths

    # =====================================================
    # SCORING
    # =====================================================

    def perspective_score(self, vps, lengths):
        """
        Length-weighted coverage score: what fraction of total detected
        EDGE LENGTH (not just line count) is explained by the dominant
        vanishing point(s). A photo can have hundreds of short edges
        from tree branches or gravel texture that never converge to
        anything -- those shouldn't dilute the score the way they
        would under a plain line-count ratio. A few long, cleanly
        converging structural lines (rails, roofline, facade edges)
        should carry the weight they visually deserve.
        """
        total_length = float(np.sum(lengths)) if len(lengths) else 0.0
        if total_length == 0 or len(vps) == 0:
            return 0.0

        covered_length = sum(vp["inlier_length"] for vp in vps)
        score = covered_length / total_length
        return round(float(min(score, 1.0)), 3)

    def interpret_score(self, score, applicable, num_vps):
        if not applicable:
            return "Not applicable (scene lacks structural line geometry)"
        if num_vps == 0:
            return "Inconsistent (no reliable vanishing point found)"
        if score >= 0.5:
            return "Consistent (lines converge to confident vanishing point(s))"
        if score >= 0.25:
            return "Mixed (partial convergence; substantial unexplained structure)"
        return "Inconsistent (little reliable perspective convergence detected)"

    # =====================================================
    # VISUALIZATION
    # =====================================================

    def draw_lines(self, image, lines):
        output = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return output

    def draw_vp_groups(self, image, lines, vps):
        """Colors each vanishing point's inlier lines distinctly."""
        output = image.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        ]
        for vi, vp in enumerate(vps):
            color = colors[vi % len(colors)]
            for idx in vp["line_indices"]:
                x1, y1, x2, y2 = lines[idx]
                cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        return output

    def draw_vanishing_points(self, image, vps):
        output = image.copy()
        for vp in vps:
            if vp is None:
                continue
            x, y = map(int, vp["point"])
            h, w = image.shape[:2]
            if -w < x < 2 * w and -h < y < 2 * h:
                cv2.circle(output, (x, y), 10, (255, 0, 0), -1)
        return output

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def analyze(self, image_path):
        image = self.load_image(image_path)
        h, w = image.shape[:2]

        processed = self.preprocess(image)
        edges = self.detect_edges(processed)
        raw_lines = self.detect_lines(edges, image.shape)
        diag = float(np.hypot(h, w))
        lines = self.merge_duplicate_lines(raw_lines, image_diag=diag)

        line_density = len(lines) / (h * w)
        applicable = len(lines) >= self.min_lines_for_analysis and \
            line_density < self.texture_line_density_limit

        if applicable:
            vps, lengths = self.find_vanishing_points(
                lines,
                min_inliers=max(5, int(len(lines) * 0.02))
            )
        else:
            vps, lengths = [], np.array([])

        score = self.perspective_score(vps, lengths) if applicable else 0.0
        interpretation = self.interpret_score(score, applicable, len(vps))

        line_image = self.draw_lines(image, lines)
        group_image = self.draw_vp_groups(image, lines, vps) if vps else image.copy()
        vp_image = self.draw_vanishing_points(group_image, vps)

        return {
            "original": image,
            "edges": edges,
            "lines": line_image,
            "groups_image": group_image,
            "vanishing_points": vp_image,

            "num_lines": len(lines),
            "raw_line_count": len(raw_lines),
            "num_vanishing_points": len(vps),

            "perspective_score": score,
            "perspective_interpretation": interpretation,
            "applicable": applicable,

            "vps": vps,
        }


# =====================================================
# VISUALIZATION HELPER
# =====================================================

def show_results(results, save_path=None):
    plt.figure(figsize=(24, 6))

    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(results["edges"], cmap="gray")
    plt.title("Edges")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(cv2.cvtColor(results["lines"], cv2.COLOR_BGR2RGB))
    plt.title(f"Lines ({results['num_lines']}, raw {results['raw_line_count']})")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(cv2.cvtColor(results["groups_image"], cv2.COLOR_BGR2RGB))
    plt.title(f"VP Groups ({results['num_vanishing_points']})")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(results["vanishing_points"], cv2.COLOR_BGR2RGB))
    plt.title(f"VPs ({results['num_vanishing_points']})")
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
    import json

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../data/test_images/StockCake-Glowing_Urban_Tower-1539478-standard.jpg"
    )

    analyzer = PerspectiveAnalyzer()
    results = analyzer.analyze(image_path)

    print()
    print("=" * 50)
    print("PERSPECTIVE ANALYSIS")
    print("=" * 50)
    print()
    print("Lines detected:", results["num_lines"], f"(raw: {results['raw_line_count']})")
    print("Vanishing point clusters:", results["num_vanishing_points"])
    print("Perspective score:", results["perspective_score"])
    print("Interpretation:", results["perspective_interpretation"])
    print("\nVANISHING POINTS:\n")

    for i, vp in enumerate(results["vps"]):
        print(
            f"VP {i+1}: {100*vp['confidence']:.1f}% confidence "
            f"({vp['inliers']} inlier lines, "
            f"{vp['inlier_length']:.0f}px total length, "
            f"{vp['total_intersections']} lines available at this stage)"
        )

    show_results(results, save_path="perspective_v1_1_debug.png")