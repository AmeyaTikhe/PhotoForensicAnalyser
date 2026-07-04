from src.analyzers.perspective import (
    PerspectiveAnalyzer,
    show_results
)


IMAGE_PATH = (
    "data/raw/buildings/"
    "Glowing_Urban_Tower.jpg"
)


analyzer = PerspectiveAnalyzer()

results = analyzer.analyze(
    IMAGE_PATH
)


print()

print("=" * 50)

print("PERSPECTIVE ANALYSIS")

print("=" * 50)

print()

print("Lines detected:",
      results["num_lines"])

print("Parallel groups:",
      results["num_groups"])

print("Vanishing points:",
      len(results["vps"]))

print("Perspective score:",
      results["perspective_score"])

print()

show_results(results)