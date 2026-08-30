"""
PhotoForensics
Main Driver
"""

IMAGE_PATH = "data/raw/shadows/Man_Modern_Window.jpeg"

# Select ONE analyzer
RUN_PERSPECTIVE = False
RUN_LIGHTING = False  # broken
RUN_ILLUMINATION = True
RUN_SHADOW_CONSISTENCY = False
RUN_NOISE = False


if RUN_PERSPECTIVE:
    from src.analyzers.perspective import PerspectiveAnalyzer, show_results

    analyzer = PerspectiveAnalyzer()
    results = analyzer.analyze(IMAGE_PATH)

    print("\n" + "=" * 50)
    print("PERSPECTIVE ANALYSIS")
    print("=" * 50)
    print("Lines detected:", results["num_lines"])
    print("Vanishing point clusters:", results["num_vanishing_points"])
    print("Perspective score:", results["perspective_score"])
    print("Interpretation:", results["perspective_interpretation"])

    print("\nVANISHING POINTS\n")
    for i, vp in enumerate(results["vps"]):
        print(
            f"VP {i+1}: "
            f"{100*vp['confidence']:.1f}% "
            f"({vp['inliers']}/{vp['total_intersections']})"
        )

    show_results(results)


if RUN_LIGHTING:
    from src.analyzers.lighting import LightingAnalyzer, show_results

    analyzer = LightingAnalyzer()
    results = analyzer.analyze(IMAGE_PATH)

    print("\n" + "=" * 50)
    print("LIGHTING ANALYSIS")
    print("=" * 50)
    print("Lighting score:", results["lighting_score"])
    print(results["lighting_interpretation"])

    print("\nStatistics")
    print("Mean brightness:", results["brightness_mean"])
    print("Brightness std :", results["brightness_std"])
    print("Mean illumination:", results["illumination_mean"])
    print("Illumination std :", results["illumination_std"])
    print("Bright fraction:", results["bright_fraction"])

    show_results(results)


if RUN_ILLUMINATION:
    from src.analyzers.illumination_v3 import IlluminationAnalyzer, show_results

    analyzer = IlluminationAnalyzer()
    results = analyzer.analyze(IMAGE_PATH)

    print("\n" + "=" * 50)
    print("ILLUMINATION ANALYSIS")
    print("=" * 50)
    print(f"Shadow candidates       : {results['shadow_candidate_pixels']}")
    print(f"Shadow edge pixels      : {results['shadow_edge_pixels']}")
    print(f"Illumination range      : {results['illumination_min']} - {results['illumination_max']}")

    show_results(results)


if RUN_SHADOW_CONSISTENCY:
    from src.analyzers.shadow_consistency import (
        analyze_shadow_consistency,
        draw_consistency_result,
    )

    result = analyze_shadow_consistency(IMAGE_PATH)

    print("\n" + "=" * 50)
    print("SHADOW CONSISTENCY -- SUMMARY")
    print("=" * 50)
    if result["insufficient_sample"]:
        print(
            f"Verdict: INSUFFICIENT SAMPLE "
            f"({len(result['rays'])} valid caster(s) found)"
        )
    else:
        print(f"Consensus light point: {result['consensus_point']}")
        n_inliers = sum(1 for r in result["rays"] if r["is_inlier"])
        print(f"Inliers: {n_inliers}/{len(result['rays'])}")

    draw_consistency_result(result)


if RUN_NOISE:
    from src.analyzers.noise import analyze, show_results

    results = analyze(IMAGE_PATH)

    print("\n" + "=" * 50)
    print("NOISE ANALYSIS")
    print("=" * 50)
    print(f"Version : {results['version']}")
    print(f"Mean    : {results['stats']['mean']:.4f}")
    print(f"Std     : {results['stats']['std']:.4f}")
    print(f"Var     : {results['stats']['var']:.4f}")
    print(f"Range   : {results['stats']['min']:.2f} to {results['stats']['max']:.2f}")

    if "suspicious_boxes" in results:
        print(f"Suspicious regions (statistical): {len(results['suspicious_boxes'])}")
    if "ml_boxes" in results:
        print(f"Suspicious regions (ML-assisted): {len(results['ml_boxes'])}")

    show_results(results)