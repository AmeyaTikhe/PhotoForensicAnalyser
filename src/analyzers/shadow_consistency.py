"""
Shadow consistency analysis.

Builds on illumination.py's shadow mask. Implements the geometric
consistency test used in shadow-based photo forensics (Kee, O'Brien &
Farid style), simplified to not require full camera calibration:

    For a single light source, the 2D line drawn from
        (top of caster) -> (tip of that caster's own shadow)
    should be the same for every caster in the scene -- i.e. all such
    lines should intersect at one common point (the "shadow vanishing
    point": a finite point for a nearby lamp, or effectively at
    infinity / mutually parallel for a distant sun).

    A caster whose ray does NOT pass through the scene's consensus
    point is either lit inconsistently with the rest of the image, or
    physically/photographically implausible -- i.e. a candidate for
    tampering.

Stage B (this file): automatic caster identification
    - people        -> OpenCV HOG person detector (bbox -> base/top)
    - structures     -> near-vertical Hough lines, clustered by x
                        position (pillars, mullions, door frames)

Stage C (this file): shadow-tip association
    - for each caster base point, find the connected shadow-mask
      component nearest to it and take the mask pixel farthest from
      the base as the shadow tip

Stage D/E (this file): ray construction + RANSAC consensus point
"""

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.analyzers.illumination import IlluminationAnalyzer


# =====================================================
# STAGE B: CASTER DETECTION
# =====================================================

class CasterDetector:

    def __init__(
        self,
        hog_win_stride=(4, 4),
        hog_padding=(8, 8),
        hog_scale=1.05,
        vertical_angle_tol=10.0,   # degrees from 90 to count as "vertical"
        vertical_canny_low=50,
        vertical_canny_high=150,
        vertical_hough_threshold=80,
        vertical_min_line_length=100,
        vertical_max_line_gap=15,
        cluster_x_tol=25,          # px: merge vertical lines into one caster
    ):
        self.hog = None
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except AttributeError:
            print(
                "[CasterDetector] cv2.HOGDescriptor unavailable in this "
                "OpenCV build/install -- person detection disabled, "
                "falling back to structural (vertical-line) casters only. "
                "Fix: run 'pip uninstall opencv-python opencv-python-headless "
                "opencv-contrib-python -y && pip install opencv-python' in "
                "this environment."
            )

        self.hog_win_stride = hog_win_stride
        self.hog_padding = hog_padding
        self.hog_scale = hog_scale

        self.vertical_angle_tol = vertical_angle_tol
        self.vertical_canny_low = vertical_canny_low
        self.vertical_canny_high = vertical_canny_high
        self.vertical_hough_threshold = vertical_hough_threshold
        self.vertical_min_line_length = vertical_min_line_length
        self.vertical_max_line_gap = vertical_max_line_gap
        self.cluster_x_tol = cluster_x_tol

    # ---------------------------------------------------
    # PEOPLE
    # ---------------------------------------------------

    def detect_people(self, image):
        if self.hog is None:
            return []

        rects, weights = self.hog.detectMultiScale(
            image,
            winStride=self.hog_win_stride,
            padding=self.hog_padding,
            scale=self.hog_scale,
        )

        casters = []
        for (x, y, w, h), weight in zip(rects, weights):
            base = (x + w // 2, y + h)
            top = (x + w // 2, y)
            casters.append({
                "type": "person",
                "base": base,
                "top": top,
                "bbox": (x, y, w, h),
                "confidence": float(weight),
            })
        return casters

    def detect_people_silhouette(
        self,
        image,
        dark_thresh=60,
        min_area=800,
        min_aspect_ratio=1.4,
        max_aspect_ratio=8.0,
    ):
        """Detects standing figures as dark, tall/narrow blobs against a
        bright background -- HOG's pretrained model expects normal
        lighting and misses backlit silhouettes (e.g. a person standing
        in front of a bright window), which is a common case in this
        kind of shadow-forensics photo."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark_mask, connectivity=8
        )

        casters = []
        for label in range(1, n_labels):
            x, y, w, h, area = stats[label][:5]
            if area < min_area:
                continue

            aspect = h / max(w, 1)
            if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
                continue  # not tall/narrow enough to be a standing figure

            if y < 5:
                continue  # hugs top edge -> likely ceiling/doorway shadow

            base = (x + w // 2, y + h)
            top = (x + w // 2, y)
            casters.append({
                "type": "person_silhouette",
                "base": base,
                "top": top,
                "bbox": (int(x), int(y), int(w), int(h)),
                "confidence": None,
            })
        return casters

    def _dedupe_people(self, hog_people, silhouette_people, dist_tol=40):
        """Drop silhouette detections that are near-duplicates of a HOG
        detection (same person found twice by two different methods)."""
        kept = list(hog_people)
        for sp in silhouette_people:
            sx, sy = sp["base"]
            is_dupe = any(
                np.hypot(sx - hp["base"][0], sy - hp["base"][1]) <= dist_tol
                for hp in hog_people
            )
            if not is_dupe:
                kept.append(sp)
        return kept

    # ---------------------------------------------------
    # STRUCTURAL (pillars, mullions, frames)
    # ---------------------------------------------------

    def detect_vertical_structures(self, image, floor_top_y=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.vertical_canny_low, self.vertical_canny_high)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.vertical_hough_threshold,
            minLineLength=self.vertical_min_line_length,
            maxLineGap=self.vertical_max_line_gap,
        )

        if lines is None:
            return []

        vertical = []
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = l
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            if abs(angle - 90) <= self.vertical_angle_tol:
                # normalize so y1 < y2 (top to bottom)
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                vertical.append((x1, y1, x2, y2))

        if not vertical:
            return []

        # cluster by x position (average of the two endpoints' x)
        vertical.sort(key=lambda l: (l[0] + l[2]) / 2)

        clusters = []
        current = [vertical[0]]
        for l in vertical[1:]:
            cx = (l[0] + l[2]) / 2
            ccx = (current[-1][0] + current[-1][2]) / 2
            if abs(cx - ccx) <= self.cluster_x_tol:
                current.append(l)
            else:
                clusters.append(current)
                current = [l]
        clusters.append(current)

        casters = []
        for cluster in clusters:
            # keep the longest line in the cluster as representative
            longest = max(
                cluster, key=lambda l: np.hypot(l[2] - l[0], l[3] - l[1])
            )
            x1, y1, x2, y2 = longest

            # only count as a caster if it plausibly reaches the floor
            # region (otherwise it's a window mullion up in the air,
            # not something with a visible shadow on the floor)
            if floor_top_y is not None and y2 < floor_top_y - 15:
                continue

            casters.append({
                "type": "structure",
                "base": (int(x2), int(y2)),
                "top": (int(x1), int(y1)),
                "bbox": (
                    int(min(x1, x2)) - 6,
                    int(min(y1, y2)),
                    int(abs(x2 - x1)) + 12,
                    int(abs(y2 - y1)),
                ),
                "confidence": None,
            })

        return casters

    def detect_all(self, image, floor_top_y=None):
        hog_people = self.detect_people(image)
        silhouette_people = self.detect_people_silhouette(image)
        people = self._dedupe_people(hog_people, silhouette_people)
        structures = self.detect_vertical_structures(image, floor_top_y=floor_top_y)
        return people + structures


# =====================================================
# STAGE C: SHADOW-TIP ASSOCIATION
# =====================================================

class ShadowTipAssociator:
    """Associates each caster with the shadow-mask blob that is its own
    cast shadow.

    Candidates are resolved in two passes so no caster can claim a blob
    that's genuinely closer to a different caster's base (e.g. a pillar
    grabbing the tail end of a person's own shadow just because the
    pillar's base happens to be nearby):

    1. Label the WHOLE mask once. For every (caster, blob) pair within
       search_radius, compute distance using only the blob's pixels
       outside that caster's own bbox (so a caster can't claim its own
       silhouette/edge as its shadow). A blob is "owned" by whichever
       caster is nearest to it.
    2. Each caster then ranks only the blobs it owns by angular
       agreement with a local direction prior (rejecting neighborhoods
       dominated by repeating texture rather than one real shadow),
       tie-broken toward longer, better-constrained shadows.
    """

    def __init__(
        self,
        search_radius=80,
        min_shadow_length=25,
        max_angle_deviation=35.0,
        local_angle_radius_mult=2.5,
        min_local_pixels=200,
        min_local_elongation=1.3,
        min_local_dominance_ratio=2.0,
        length_bias=0.05,
        self_exclusion_margin=10,
    ):
        self.search_radius = search_radius
        self.min_shadow_length = min_shadow_length
        self.max_angle_deviation = max_angle_deviation
        self.local_angle_radius_mult = local_angle_radius_mult
        self.min_local_pixels = min_local_pixels
        self.min_local_elongation = min_local_elongation
        self.min_local_dominance_ratio = min_local_dominance_ratio
        self.length_bias = length_bias
        self.self_exclusion_margin = self_exclusion_margin

    @staticmethod
    def _angle_diff_mod180(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)

    def _local_expected_angle(self, shadow_mask, base_point, fallback_angle):
        bx, by = base_point
        r = int(self.search_radius * self.local_angle_radius_mult)
        h, w = shadow_mask.shape

        x0, x1 = max(0, bx - r), min(w, bx + r)
        y0, y1 = max(0, by - r), min(h, by + r)
        patch = shadow_mask[y0:y1, x0:x1]

        if np.count_nonzero(patch) < self.min_local_pixels:
            return fallback_angle

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            patch, connectivity=8
        )
        if n_labels <= 1:
            return fallback_angle

        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1]

        # one component must clearly dominate, or this patch is texture
        # (many similarly-sized fragments), not a single real shadow
        top_area = areas[order[0]]
        second_area = areas[order[1]] if len(order) > 1 else 0
        if top_area < self.min_local_pixels:
            return fallback_angle
        if second_area > 0 and (top_area / second_area) < self.min_local_dominance_ratio:
            return fallback_angle

        dominant_label = order[0] + 1
        ys, xs = np.nonzero(labels == dominant_label)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        _, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
        elongation = eigenvalues[0, 0] / (eigenvalues[1, 0] + 1e-6)
        if elongation < self.min_local_elongation:
            return fallback_angle

        vx, vy = eigenvectors[0]
        return np.degrees(np.arctan2(vy, vx)) % 180

    def _candidate(self, xs, ys, label, caster):
        bx, by = caster["base"]
        x, y, w, h = caster.get("bbox", (0, 0, 0, 0))
        m = self.self_exclusion_margin
        outside = ~((xs >= x - m) & (xs < x + w + m) & (ys >= y - m) & (ys < y + h + m))
        xs, ys = xs[outside], ys[outside]
        if len(xs) == 0:
            return None

        dists = np.hypot(xs - bx, ys - by)
        dmin = float(dists.min())
        if dmin > self.search_radius:
            return None

        tip_idx = np.argmax(dists)
        tip = (int(xs[tip_idx]), int(ys[tip_idx]))
        length = float(dists[tip_idx])
        if length < self.min_shadow_length:
            return None

        angle = np.degrees(np.arctan2(tip[1] - by, tip[0] - bx)) % 180
        return {"tip": tip, "length": length, "angle": angle, "dmin": dmin, "label": label}

    def _choose_tip(self, shadow_mask, caster, candidates, expected_angle):
        if not candidates:
            return None, None, {"local_angle": None, "candidates": []}

        if expected_angle is None:
            best = min(candidates, key=lambda c: c["dmin"])
            for c in candidates:
                c["passed"] = c is best
            return best["tip"], best["label"], {"local_angle": None, "candidates": candidates}

        local_angle = self._local_expected_angle(shadow_mask, caster["base"], expected_angle)
        for c in candidates:
            c["deviation"] = self._angle_diff_mod180(c["angle"], local_angle)
            c["passed"] = c["deviation"] <= self.max_angle_deviation

        scored = [(c["deviation"], c) for c in candidates if c["passed"]]
        debug = {"local_angle": local_angle, "candidates": candidates}
        if not scored:
            return None, None, debug

        # angular agreement first, longer candidates win close calls
        scored.sort(key=lambda t: t[0] - self.length_bias * t[1]["length"])
        best = scored[0][1]
        for c in candidates:
            c["chosen"] = c is best
        return best["tip"], best["label"], debug

    def associate_all(self, shadow_mask, casters, expected_angle=None):
        """Returns a list of (tip, label, debug), one per caster, in order."""
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            shadow_mask, connectivity=8
        )

        bids = {}  # label -> list of (dmin, caster_idx, candidate)
        for label in range(1, n_labels):
            ys, xs = np.where(labels == label)
            for i, caster in enumerate(casters):
                cand = self._candidate(xs, ys, label, caster)
                if cand is not None:
                    bids.setdefault(label, []).append((cand["dmin"], i, cand))

        owned = {i: [] for i in range(len(casters))}
        for label, label_bids in bids.items():
            _, winner_i, winner_cand = min(label_bids, key=lambda b: b[0])
            owned[winner_i].append(winner_cand)

        return [
            self._choose_tip(shadow_mask, caster, owned[i], expected_angle)
            for i, caster in enumerate(casters)
        ]


# =====================================================
# STAGE D/E: RAY CONSTRUCTION + CONSENSUS
# =====================================================

class ShadowConsistencyChecker:

    def __init__(self, inlier_dist_px=40, ransac_iters=500, random_seed=0):
        self.inlier_dist_px = inlier_dist_px
        self.ransac_iters = ransac_iters
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def _line_from_points(p1, p2):
        """Return (a, b, c) for line a*x + b*y + c = 0 through p1, p2."""
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = -(a * x1 + b * y1)
        norm = np.hypot(a, b)
        if norm < 1e-9:
            return None
        return a / norm, b / norm, c / norm

    @staticmethod
    def _point_line_distance(point, line):
        a, b, c = line
        x, y = point
        return abs(a * x + b * y + c)

    @staticmethod
    def _intersect(line1, line2):
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            return None  # parallel
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return (x, y)

    def find_consensus(self, rays):
        """rays: list of dicts with 'top' and 'tip' points (image coords).
        Returns (consensus_point_or_None, inlier_flags, per_ray_distance)."""
        lines = []
        valid_idx = []
        for i, ray in enumerate(rays):
            line = self._line_from_points(ray["top"], ray["tip"])
            if line is not None:
                lines.append(line)
                valid_idx.append(i)

        n = len(lines)
        if n < 2:
            return None, [False] * len(rays), [None] * len(rays)

        best_inliers = None
        best_point = None

        for _ in range(self.ransac_iters):
            i, j = self.rng.choice(n, size=2, replace=False)
            pt = self._intersect(lines[i], lines[j])
            if pt is None:
                continue

            inliers = 0
            for line in lines:
                d = self._point_line_distance(pt, line)
                if d <= self.inlier_dist_px:
                    inliers += 1

            if best_inliers is None or inliers > best_inliers:
                best_inliers = inliers
                best_point = pt

        if best_point is None:
            return None, [False] * len(rays), [None] * len(rays)

        # refine: least-squares point minimizing sum of squared line
        # distances over the inlier set
        A = []
        Bv = []
        for line in lines:
            d = self._point_line_distance(best_point, line)
            if d <= self.inlier_dist_px:
                a, b, c = line
                A.append([a, b])
                Bv.append([-c])
        if len(A) >= 2:
            A = np.array(A)
            Bv = np.array(Bv)
            sol, *_ = np.linalg.lstsq(A, Bv, rcond=None)
            refined_point = (float(sol[0, 0]), float(sol[1, 0]))
        else:
            refined_point = best_point

        inlier_flags = [False] * len(rays)
        distances = [None] * len(rays)
        for idx, line in zip(valid_idx, lines):
            d = self._point_line_distance(refined_point, line)
            distances[idx] = d
            inlier_flags[idx] = d <= self.inlier_dist_px

        return refined_point, inlier_flags, distances


# =====================================================
# FULL PIPELINE
# =====================================================

def analyze_shadow_consistency(image_path, illum_kwargs=None, verbose=True):
    illum_kwargs = illum_kwargs or {}

    illum_analyzer = IlluminationAnalyzer(**illum_kwargs)
    illum_results = illum_analyzer.analyze(image_path)

    image = illum_results["original"]
    shadow_mask = illum_results["shadow_mask"]

    floor_top_y = int(image.shape[0] * illum_analyzer.floor_roi_top_frac)

    caster_detector = CasterDetector()
    casters = caster_detector.detect_all(image, floor_top_y=floor_top_y)

    # use the whole-image dominant shadow angle (from illumination.py)
    # as a directional prior to reject implausible tip candidates
    expected_angle = illum_results["shadow_angle"]

    tip_associator = ShadowTipAssociator()
    tip_results = tip_associator.associate_all(shadow_mask, casters, expected_angle=expected_angle)
    rays = [
        {**caster, "tip": tip, "_tip_label": label, "debug": debug}
        for caster, (tip, label, debug) in zip(casters, tip_results)
    ]

    valid_rays = [r for r in rays if r["tip"] is not None]
    for r in rays:
        r.pop("_tip_label", None)

    MIN_CASTERS_FOR_VERDICT = 4

    checker = ShadowConsistencyChecker()
    if len(valid_rays) >= MIN_CASTERS_FOR_VERDICT:
        consensus_point, inlier_flags, distances = checker.find_consensus(valid_rays)
    else:
        consensus_point, inlier_flags, distances = None, [None] * len(valid_rays), [None] * len(valid_rays)

    for r, inlier, dist in zip(valid_rays, inlier_flags, distances):
        r["is_inlier"] = inlier
        r["consensus_distance"] = dist

    insufficient_sample = len(valid_rays) < MIN_CASTERS_FOR_VERDICT

    if verbose:
        print()
        print("=" * 50)
        print("SHADOW CONSISTENCY ANALYSIS")
        print("=" * 50)
        print(f"Casters detected      : {len(casters)}")
        print(f"Casters w/ shadow tip : {len(valid_rays)}")
        print(f"Expected angle prior  : {expected_angle}")
        if insufficient_sample:
            print(
                f"VERDICT               : INSUFFICIENT SAMPLE "
                f"(need >= {MIN_CASTERS_FOR_VERDICT} valid casters, "
                f"got {len(valid_rays)}) -- no consistency verdict given"
            )
        else:
            print(f"Consensus point       : {consensus_point}")
        for r in valid_rays:
            if r["is_inlier"] is None:
                status = "(no verdict)"
            else:
                status = "INLIER" if r["is_inlier"] else "OUTLIER"
            print(
                f"  [{r['type']:9s}] base={r['base']} top={r['top']} "
                f"tip={r['tip']}  dist={r['consensus_distance']}  {status}"
            )

    return {
        "image": image,
        "shadow_mask": shadow_mask,
        "casters": casters,
        "rays": valid_rays,
        "all_rays": rays,
        "consensus_point": consensus_point,
        "insufficient_sample": insufficient_sample,
    }


def _render_result_image(result):
    image = result["image"].copy()

    for r in result["rays"]:
        if r["is_inlier"] is None:
            color = (0, 165, 255)
        else:
            color = (0, 200, 0) if r["is_inlier"] else (0, 0, 255)
        cv2.line(image, r["top"], r["tip"], color, 2)
        cv2.circle(image, r["base"], 5, (255, 200, 0), -1)
        cv2.circle(image, r["top"], 5, (255, 0, 255), -1)
        cv2.circle(image, r["tip"], 5, color, -1)

    cp = result["consensus_point"]
    if cp is not None:
        cx = max(-2000, min(4000, int(cp[0])))
        cy = max(-2000, min(4000, int(cp[1])))
        cv2.drawMarker(
            image, (cx, cy), (0, 255, 255),
            markerType=cv2.MARKER_STAR, markerSize=30, thickness=3
        )
    return image


def _render_debug_panel(result):
    """Shows every caster's excluded bbox and every candidate tip it
    considered, not just the one it picked: green = chosen, orange =
    passed the angle filter but lost, red = failed the angle filter."""
    image = result["image"].copy()

    for r in result["all_rays"]:
        x, y, w, h = r.get("bbox", (0, 0, 0, 0))
        if w and h:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.circle(image, r["base"], 5, (255, 200, 0), -1)
        cv2.circle(image, r["top"], 5, (255, 0, 255), -1)
        for c in r["debug"]["candidates"]:
            if c.get("chosen"):
                color = (0, 200, 0)
            elif c.get("passed"):
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(image, c["tip"], 4, color, -1)
    return image


def draw_consistency_result(result, save_path=None):
    image = _render_result_image(result)
    if save_path:
        cv2.imwrite(save_path, image)
    else:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Shadow Consistency")
        plt.axis("off")
        plt.show()
    return image


def plot_diagnostics(result, save_path=None):
    """Mask + all candidate tips per caster + final result, side by side,
    plus a text summary of why each caster ended up with the tip it did."""
    shadow_mask = result["shadow_mask"]
    debug_panel = _render_debug_panel(result)
    final_panel = _render_result_image(result)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(shadow_mask, cmap="gray")
    axes[0].set_title("Shadow mask")
    axes[1].imshow(cv2.cvtColor(debug_panel, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Excluded bboxes + all candidate tips")
    axes[2].imshow(cv2.cvtColor(final_panel, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Final rays")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    print()
    print("=" * 50)
    print("PER-CASTER TIP DEBUG")
    print("=" * 50)
    for r in result["all_rays"]:
        n = len(r["debug"]["candidates"])
        la = r["debug"]["local_angle"]
        la_str = f"{la:.1f}" if la is not None else "n/a (fallback)"
        print(f"[{r['type']:16s}] base={r['base']} candidates={n} local_angle={la_str}")
        for c in r["debug"]["candidates"]:
            dev = c.get("deviation")
            dev_str = f"{dev:.1f}" if dev is not None else "n/a"
            tag = "CHOSEN" if c.get("chosen") else ("passed" if c.get("passed") else "rejected")
            print(
                f"    tip={c['tip']} len={c['length']:.0f} "
                f"angle={c['angle']:.1f} dev={dev_str}  {tag}"
            )
        if not r["debug"]["candidates"]:
            print("    (no candidates within search_radius / min_shadow_length)")

    return debug_panel, final_panel


if __name__ == "__main__":
    image_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "data/raw/shadows/Man_Modern_Window.jpeg"
    )

    result = analyze_shadow_consistency(image_path)
    draw_consistency_result(result, save_path="shadow_consistency_result.png")
    plot_diagnostics(result, save_path="shadow_consistency_debug.png")