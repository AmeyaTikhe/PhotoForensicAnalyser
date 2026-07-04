# src/analyzers/perspective.py

"""
Perspective Analysis Module

Inspired by:
Hany Farid - Photo Forensics:
Perspective, Vanishing Points, and Projective Geometry

V1 Features:
- Canny edge detection
- Hough line detection
- Homogeneous line equations
- Parallel line grouping
- Pairwise intersections
- Vanishing point estimation
- Perspective consistency scoring
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


class PerspectiveAnalyzer:

    def __init__(
        self,
        canny_low=50,
        canny_high=150,
        hough_threshold=100,
        min_line_length=100,
        max_line_gap=10,
        angle_threshold=15,
        min_group_size=15,
        max_groups=3
    ):

        self.canny_low = canny_low
        self.canny_high = canny_high

        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        self.angle_threshold = angle_threshold
        self.min_group_size = min_group_size
        self.max_groups = max_groups


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

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        return blur


    def detect_edges(self, image):

        return cv2.Canny(
            image,
            self.canny_low,
            self.canny_high
        )


    # def detect_lines(self, edges):

    #     lines = cv2.HoughLinesP(
    #         edges,
    #         rho=1,
    #         theta=np.pi / 180,
    #         threshold=self.hough_threshold,
    #         minLineLength=self.min_line_length,
    #         maxLineGap=self.max_line_gap
    #     )

    #     if lines is None:
    #         return []

    #     return lines[:, 0]
    
    def detect_lines(self, edges):

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return []

        print("LINES SHAPE:", lines.shape)
        print("FIRST LINE:", lines[0])

        # Convert to plain tuples
        processed_lines = []

        for line in lines:

            x1, y1, x2, y2 = line

            processed_lines.append(
                (x1, y1, x2, y2)
            )

        return processed_lines


    # =====================================================
    # PROJECTIVE GEOMETRY
    # =====================================================

    def line_to_homogeneous(self, line):

        x1, y1, x2, y2 = line

        p1 = np.array([x1, y1, 1])
        p2 = np.array([x2, y2, 1])

        # Cross product gives:
        # ax + by + c = 0
        l = np.cross(p1, p2)

        return l


    def line_angle(self, line):

        x1, y1, x2, y2 = line

        angle = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

        return angle % 180


    def intersection(self, l1, l2):

        point = np.cross(l1, l2)

        if abs(point[2]) < 1e-8:
            return None

        x = point[0] / point[2]
        y = point[1] / point[2]

        return (x, y)


    # =====================================================
    # PARALLEL LINE GROUPING
    # =====================================================

    def group_parallel_lines(self, lines):

        groups = []

        for line in lines:

            angle = self.line_angle(line)

            assigned = False

            for group in groups:

                group_angle = group["angle"]

                # if abs(angle - group_angle) < self.angle_threshold:
                
                diff = min(

                    abs(angle - group_angle),

                    180 - abs(angle - group_angle)

                )

                if diff < self.angle_threshold:

                    group["lines"].append(line)

                    assigned = True
                    break

            if not assigned:

                groups.append({

                    "angle": angle,
                    "lines": [line]

                })
        
        print("\nGROUP STATISTICS:\n")

        for i, group in enumerate(groups):

            print(
                f"Group {i+1}: "
                f"Angle={group['angle']:.2f}°, "
                f"Lines={len(group['lines'])}"
            )
            
        return groups


    # =====================================================
    # VANISHING POINT ESTIMATION
    # =====================================================

    def estimate_vanishing_point(self, lines):

        if len(lines) < 2:
            return None

        homogeneous_lines = [

            self.line_to_homogeneous(line)
            for line in lines

        ]

        intersections = []

        for l1, l2 in combinations(
            homogeneous_lines,
            2
        ):

            point = self.intersection(l1, l2)

            if point is not None:

                intersections.append(point)

        if len(intersections) == 0:
            return None

        intersections = np.array(intersections)

        # Simple centroid estimate
        vp_x = np.median(intersections[:, 0])
        vp_y = np.median(intersections[:, 1])

        return (vp_x, vp_y)


    # =====================================================
    # SCORING
    # =====================================================

    # def perspective_score(self, groups):

    #     if len(groups) == 0:
    #         return 0.0

    #     line_counts = [

    #         len(group["lines"])
    #         for group in groups

    #     ]

    #     dominant = max(line_counts)

    #     total = sum(line_counts)

    #     score = dominant / total

    #     return round(score, 3)
    
    def perspective_score(self, groups):

        if len(groups) < 2:
            return 0.0

        total_lines = sum(
            len(g["lines"])
            for g in groups
        )

        dominant_lines = total_lines

        score = dominant_lines / total_lines

        return round(score, 3)


    # =====================================================
    # VISUALIZATION
    # =====================================================

    def draw_lines(self, image, lines):

        output = image.copy()

        for line in lines:

            x1, y1, x2, y2 = line

            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        return output


    def draw_vanishing_point(self, image, point):

        if point is None:
            return image

        output = image.copy()

        x, y = map(int, point)

        cv2.circle(
            output,
            (x, y),
            12,
            (255, 0, 0),
            -1
        )

        return output


    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def analyze(self, image_path):

        image = self.load_image(image_path)

        processed = self.preprocess(image)

        edges = self.detect_edges(processed)

        lines = self.detect_lines(edges)

        groups = self.group_parallel_lines(lines)
        
        # Remove small groups
        groups = [

            g for g in groups

            if len(g["lines"]) >= self.min_group_size

        ]

        # Sort by size
        groups.sort(
            key=lambda g: len(g["lines"]),
            reverse=True
        )

        # Keep only top groups
        groups = groups[:self.max_groups]

        vanishing_points = []

        for group in groups:

            if len(group["lines"]) >= 2:

                vp = self.estimate_vanishing_point(
                    group["lines"]
                )

                if vp is not None:

                    vanishing_points.append(vp)

        score = self.perspective_score(groups)

        line_image = self.draw_lines(
            image,
            lines
        )

        vp_image = line_image.copy()

        for vp in vanishing_points:

            vp_image = self.draw_vanishing_point(
                vp_image,
                vp
            )

        return {

            "original": image,

            "edges": edges,

            "lines": line_image,

            "vanishing_points": vp_image,

            "num_lines": len(lines),

            "num_groups": len(groups),

            "perspective_score": score,

            "groups": groups,

            "vps": vanishing_points

        }


# =====================================================
# VISUALIZATION HELPER
# =====================================================

def show_results(results):

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 4, 1)

    plt.imshow(
        cv2.cvtColor(
            results["original"],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Original")
    plt.axis("off")


    plt.subplot(1, 4, 2)

    plt.imshow(
        results["edges"],
        cmap="gray"
    )

    plt.title("Edges")
    plt.axis("off")


    plt.subplot(1, 4, 3)

    plt.imshow(
        cv2.cvtColor(
            results["lines"],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"Lines ({results['num_lines']})"
    )

    plt.axis("off")


    plt.subplot(1, 4, 4)

    plt.imshow(
        cv2.cvtColor(
            results["vanishing_points"],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"VPs ({len(results['vps'])})"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()