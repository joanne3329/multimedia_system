from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
RESULT_DIR = Path("results/part1")


def cubic_bezier(control_points, t_values):
    """Use the cubic Bezier formula to calculate curve points."""
    b0, b1, b2, b3 = np.asarray(control_points, dtype=float)
    t = np.asarray(t_values, dtype=float)[:, None]

    return (
        (1 - t) ** 3 * b0
        + 3 * t * (1 - t) ** 2 * b1
        + 3 * t**2 * (1 - t) * b2
        + t**3 * b3
    )


def split_bezier_midpoint(control_points):
    """Split one cubic Bezier curve at t = 0.5 by midpoint construction."""
    b0, b1, b2, b3 = np.asarray(control_points, dtype=float)

    m01 = (b0 + b1) / 2
    m12 = (b1 + b2) / 2
    m23 = (b2 + b3) / 2

    m012 = (m01 + m12) / 2
    m123 = (m12 + m23) / 2

    center = (m012 + m123) / 2

    left = np.array([b0, m01, m012, center])
    right = np.array([center, m123, m23, b3])
    return left, right


def midpoint_polyline(control_points, depth):
    """
    Recursively split a Bezier curve and return the endpoints of sub-curves.

    The returned points form the polyline approximation required in Part 1.
    """
    if depth == 0:
        control_points = np.asarray(control_points, dtype=float)
        return np.array([control_points[0], control_points[-1]])

    left, right = split_bezier_midpoint(control_points)
    left_points = midpoint_polyline(left, depth - 1)
    right_points = midpoint_polyline(right, depth - 1)

    # The last point of left_points is the first point of right_points.
    return np.vstack([left_points[:-1], right_points])


def distance_point_to_segment(points, start, end):
    """Calculate each point's distance to one line segment."""
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    segment = end - start
    segment_length_sq = np.dot(segment, segment)

    if segment_length_sq == 0:
        return np.linalg.norm(points - start, axis=1)

    projection = np.dot(points - start, segment) / segment_length_sq
    projection = np.clip(projection, 0, 1)
    closest = start + projection[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def max_gap_formula_to_polyline(formula_points, polyline_points):
    """
    For every point on the formula curve, find its nearest polyline segment.

    This follows the assignment direction: A-to-B distance, not B-to-A.
    """
    min_distances = np.full(len(formula_points), np.inf)

    for i in range(len(polyline_points) - 1):
        distances = distance_point_to_segment(
            formula_points,
            polyline_points[i],
            polyline_points[i + 1],
        )
        min_distances = np.minimum(min_distances, distances)

    return float(np.max(min_distances))


def plot_curve_comparison(control_points, formula_points, depth):
    """Draw the formula curve and a shallow midpoint polyline together."""
    polyline = midpoint_polyline(control_points, depth)
    control_points = np.asarray(control_points, dtype=float)

    plt.figure(figsize=(8, 6))
    plt.plot(
        formula_points[:, 0],
        formula_points[:, 1],
        label="A: cubic formula",
        linewidth=2.5,
    )
    plt.plot(
        polyline[:, 0],
        polyline[:, 1],
        "o--",
        label=f"B: midpoint construction, depth={depth}",
        linewidth=1.5,
        markersize=4,
    )
    plt.plot(
        control_points[:, 0],
        control_points[:, 1],
        "k:",
        label="control polygon",
        linewidth=1,
    )
    plt.scatter(control_points[:, 0], control_points[:, 1], color="black", s=25)
    plt.title("Part 1.1: Two Roads to the Same Bezier Curve")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "part1_1_curve_comparison.png", dpi=160)
    plt.close()


def plot_convergence(depths, errors):
    """Draw how midpoint approximation error changes with depth."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(depths, errors, "o-", linewidth=2)
    plt.axhline(0.5, color="red", linestyle="--", label="0.5 px")
    plt.yscale("log")
    plt.title("Part 1.2: Convergence of Midpoint Construction")
    plt.xlabel("subdivision depth")
    plt.ylabel("max gap from formula curve to polyline (px)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "part1_2_convergence.png", dpi=160)
    plt.close()


def draw_cubic_segment(control_points, label=None):
    """Draw one cubic segment using our own formula sampler."""
    t_values = np.linspace(0, 1, 400)
    points = cubic_bezier(control_points, t_values)
    plt.plot(points[:, 0], points[:, 1], linewidth=2.2, label=label)


def draw_direction_lines(points_before_join, join, points_after_join):
    """Draw the two direction lines around a join point."""
    xs = [points_before_join[0], join[0], points_after_join[0]]
    ys = [points_before_join[1], join[1], points_after_join[1]]
    plt.plot(xs[:2], ys[:2], "k--", linewidth=1)
    plt.plot(xs[1:], ys[1:], "k--", linewidth=1)
    plt.scatter(xs, ys, color="black", s=20)


def plot_join_examples():
    """Create the smooth/corner and equal/unequal direction-line examples."""
    plt.figure(figsize=(12, 5))

    # Three connected cubic segments: join1 is smooth, join2 is a corner.
    segment1 = np.array([[60, 420], [130, 260], [230, 250], [300, 330]])
    segment2 = np.array([[300, 330], [370, 410], [430, 410], [500, 330]])
    segment3 = np.array([[500, 330], [500, 150], [660, 170], [720, 300]])

    plt.subplot(1, 2, 1)
    draw_cubic_segment(segment1, "segment 1")
    draw_cubic_segment(segment2, "segment 2")
    draw_cubic_segment(segment3, "segment 3")
    draw_direction_lines(segment1[2], segment1[3], segment2[1])
    draw_direction_lines(segment2[2], segment2[3], segment3[1])
    plt.title("Three Bezier Segments: Smooth Join and Corner")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.25)
    plt.legend()

    # Same join location and collinear direction points.
    # Left side is equal length; right side is unequal length.
    equal_left = np.array([[80, 360], [170, 240], [230, 300], [300, 300]])
    equal_right = np.array([[300, 300], [370, 300], [430, 360], [520, 240]])

    unequal_left = np.array([[80, 540], [170, 420], [230, 480], [300, 480]])
    unequal_right = np.array([[300, 480], [440, 480], [470, 540], [520, 420]])

    plt.subplot(1, 2, 2)
    draw_cubic_segment(equal_left, "equal length")
    draw_cubic_segment(equal_right)
    draw_direction_lines(equal_left[2], equal_left[3], equal_right[1])

    draw_cubic_segment(unequal_left, "unequal length")
    draw_cubic_segment(unequal_right)
    draw_direction_lines(unequal_left[2], unequal_left[3], unequal_right[1])

    plt.text(540, 300, "equal", va="center")
    plt.text(540, 480, "unequal", va="center")
    plt.title("Collinear Direction Lines: Equal vs Unequal Length")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "part1_3_join_examples.png", dpi=160)
    plt.close()


def write_report(errors_by_depth, first_depth_below_half):
    """Write the numbers and short answers requested by Part 1."""
    report_path = RESULT_DIR / "part1_report.txt"
    lines = []

    lines.append("Part 1: Two Roads to the Same Curve")
    lines.append("")
    lines.append("題目重點：")
    lines.append(
        "A 方法使用三次 Bezier 公式直接算 C(t)；B 方法每次用 t=0.5 的中點建構法把曲線切成兩段，"
        "遞迴切到指定深度後，用子曲線端點連成折線近似原曲線。"
    )
    lines.append("")
    lines.append("Part 1.2 convergence results:")
    lines.append("depth, max_gap_px, ratio_to_previous_depth")

    previous_error = None
    for depth, error in errors_by_depth:
        if previous_error is None:
            ratio = "N/A"
        else:
            ratio = f"{previous_error / error:.3f}"
        lines.append(f"{depth}, {error:.6f}, {ratio}")
        previous_error = error

    lines.append("")
    lines.append(f"First depth with max gap below 0.5 px: {first_depth_below_half}")
    lines.append("")
    lines.append("Error rule:")
    lines.append(
        "When the subdivision depth increases by one, each segment is roughly halved, "
        "so the maximum chord error usually decreases by about 4 times. "
        "The deeper depths follow this rule more closely; shallow depths can deviate because the polyline is still too coarse."
    )
    lines.append("")
    lines.append("One advantage of midpoint construction:")
    lines.append(
        "It can subdivide adaptively: flatter parts can stop early, while more curved parts can keep splitting. "
        "The formula method asks us to choose the number of t samples in advance."
    )
    lines.append("")
    lines.append("Joins:")
    lines.append(
        "A corner is a corner because the two direction lines around the join point are not on the same straight line."
    )
    lines.append(
        "If the two direction points are collinear with the join point but the two direction-line lengths are unequal, "
        "the join still looks smooth because the tangent direction is continuous."
    )
    lines.append(
        "Equal length is not required for visual smoothness; it mainly gives a more uniform parameter speed through the join."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_part1():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # These four points create a clearly curved shape across most of the canvas.
    control_points = np.array(
        [
            [80, 500],
            [180, 80],
            [620, 80],
            [720, 500],
        ],
        dtype=float,
    )

    # Fine sampling: the curve is under 0.1 px between neighboring samples.
    t_values = np.linspace(0, 1, 20001)
    formula_points = cubic_bezier(control_points, t_values)

    plot_curve_comparison(control_points, formula_points, depth=3)

    errors_by_depth = []
    first_depth_below_half = None

    for depth in range(0, 13):
        polyline = midpoint_polyline(control_points, depth)
        error = max_gap_formula_to_polyline(formula_points, polyline)
        errors_by_depth.append((depth, error))

        if error < 0.5 and first_depth_below_half is None:
            first_depth_below_half = depth

    plot_convergence(
        [depth for depth, _ in errors_by_depth],
        [error for _, error in errors_by_depth],
    )
    plot_join_examples()
    write_report(errors_by_depth, first_depth_below_half)

    print("Part 1 finished.")
    print(f"Results saved in: {RESULT_DIR}")

