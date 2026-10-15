from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from part1 import midpoint_polyline


CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
RESULT_DIR = Path("results/part2")


def close_path(points):
    """Ensure that the final point is the same as the first point."""

    points = np.asarray(
        points,
        dtype=float
    )

    if not np.allclose(
        points[0],
        points[-1]
    ):
        points = np.vstack([
            points,
            points[0]
        ])

    return points


def scanline_fill(
    subpaths,
    width,
    height
):
    """
    Rasterize closed subpaths using the non-zero winding rule.

    Image coordinates are used, so y increases downward.

    Sign convention:
    - Downward edge: +1
    - Upward edge: -1

    Pixel centers are located at (x + 0.5, y + 0.5).
    """

    image = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    closed_subpaths = [
        close_path(path)
        for path in subpaths
    ]

    # Process one pixel row at a time.
    for row in range(height):
        scan_y = row + 0.5
        crossings = []

        # Find all intersections between the current
        # scanline and all subpaths.
        for path in closed_subpaths:
            for p0, p1 in zip(
                path[:-1],
                path[1:]
            ):
                x0, y0 = p0
                x1, y1 = p1

                # A horizontal edge does not cross the scanline.
                if y0 == y1:
                    continue

                # The half-open interval prevents a shared
                # vertex from being counted twice.
                if (
                    min(y0, y1)
                    <= scan_y
                    < max(y0, y1)
                ):
                    t = (
                        (scan_y - y0)
                        / (y1 - y0)
                    )

                    cross_x = (
                        x0 + t * (x1 - x0)
                    )

                    # Winding-number sign convention.
                    if y1 > y0:
                        change = 1
                    else:
                        change = -1

                    crossings.append(
                        (cross_x, change)
                    )

        # Walk through crossings from left to right.
        crossings.sort(
            key=lambda item: item[0]
        )

        winding = 0

        for index, (
            cross_x,
            change
        ) in enumerate(crossings):

            winding += change

            if index + 1 == len(crossings):
                continue

            # Non-zero means that this span is inside.
            if winding == 0:
                continue

            next_x = crossings[
                index + 1
            ][0]

            start_column = max(
                0,
                int(
                    np.ceil(
                        cross_x - 0.5
                    )
                )
            )

            end_column = min(
                width,
                int(
                    np.ceil(
                        next_x - 0.5
                    )
                )
            )

            # Write directly into our own pixel array.
            if start_column < end_column:
                image[
                    row,
                    start_column:end_column
                ] = 255

    return image


def winding_number_at_point(
    point,
    subpaths
):
    """
    Calculate the winding number at one point using the
    same sign convention as scanline_fill().
    """

    point_x, point_y = np.asarray(
        point,
        dtype=float
    )

    crossings = []

    for subpath in subpaths:
        path = close_path(subpath)

        for p0, p1 in zip(
            path[:-1],
            path[1:]
        ):
            x0, y0 = p0
            x1, y1 = p1

            if y0 == y1:
                continue

            if (
                min(y0, y1)
                <= point_y
                < max(y0, y1)
            ):
                t = (
                    (point_y - y0)
                    / (y1 - y0)
                )

                cross_x = (
                    x0 + t * (x1 - x0)
                )

                if y1 > y0:
                    change = 1
                else:
                    change = -1

                crossings.append(
                    (cross_x, change)
                )

    # Accumulate crossings from the left side
    # to the test point.
    return sum(
        change
        for cross_x, change in crossings
        if cross_x <= point_x
    )


def make_five_point_star(
    center,
    radius,
    rotation_degrees=-90
):
    """
    Create one self-intersecting five-pointed star path.

    Only five outer vertices are created. The path visits
    every other vertex in the order:

    0 -> 2 -> 4 -> 1 -> 3 -> 0
    """

    center = np.asarray(
        center,
        dtype=float
    )

    angles = np.deg2rad(
        rotation_degrees
        + np.arange(5) * 72
    )

    vertices = np.column_stack([
        center[0]
        + radius * np.cos(angles),

        center[1]
        + radius * np.sin(angles)
    ])

    visiting_order = [
        0,
        2,
        4,
        1,
        3,
        0
    ]

    return vertices[visiting_order]


def make_circle_bezier_segments(
    center,
    radius
):
    """
    Describe a circle using four cubic Bézier segments.

    The two direction control points are placed
    0.5523 * radius along the tangent from each endpoint.
    """

    cx, cy = center
    k = 0.5523 * radius

    top = np.array(
        [cx, cy - radius],
        dtype=float
    )

    right = np.array(
        [cx + radius, cy],
        dtype=float
    )

    bottom = np.array(
        [cx, cy + radius],
        dtype=float
    )

    left = np.array(
        [cx - radius, cy],
        dtype=float
    )

    return [
        np.array([
            top,
            [cx + k, cy - radius],
            [cx + radius, cy - k],
            right
        ]),

        np.array([
            right,
            [cx + radius, cy + k],
            [cx + k, cy + radius],
            bottom
        ]),

        np.array([
            bottom,
            [cx - k, cy + radius],
            [cx - radius, cy + k],
            left
        ]),

        np.array([
            left,
            [cx - radius, cy - k],
            [cx - k, cy - radius],
            top
        ])
    ]


def flatten_bezier_segments(
    segments,
    depth=6
):
    """
    Flatten connected Bézier segments using the midpoint
    construction implemented in Part 1.
    """

    pieces = []

    for index, segment in enumerate(
        segments
    ):
        polyline = midpoint_polyline(
            segment,
            depth
        )

        # Do not repeat the shared endpoint.
        if index == 0:
            pieces.append(polyline)
        else:
            pieces.append(
                polyline[1:]
            )

    complete_path = np.vstack(
        pieces
    )

    return close_path(
        complete_path
    )


def save_star_result(
    star_image,
    center_winding
):
    """
    Save the filled star.

    Matplotlib only displays the pixel array produced by
    scanline_fill(). It does not draw or fill the shape.
    """

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        star_image,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        interpolation="nearest"
    )

    plt.title(
        "Part 2.2: Non-zero Fill of a "
        "Five-pointed Star\n"
        f"Center winding number = "
        f"{center_winding}"
    )

    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.xlim(0, CANVAS_WIDTH)
    plt.ylim(CANVAS_HEIGHT, 0)
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "part2_2_star_nonzero_fill.png",
        dpi=160
    )

    plt.close()


def save_circle_comparison(
    same_image,
    opposite_image,
    same_winding,
    opposite_winding
):
    """
    Save the two circle-direction results next to each other.

    Matplotlib only displays the two pixel arrays.
    """

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].imshow(
        same_image,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        interpolation="nearest"
    )

    axes[0].set_title(
        "Inner circle: same direction\n"
        f"Center winding = "
        f"{same_winding}"
    )

    axes[1].imshow(
        opposite_image,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        interpolation="nearest"
    )

    axes[1].set_title(
        "Inner circle: opposite direction\n"
        f"Center winding = "
        f"{opposite_winding}"
    )

    for axis in axes:
        axis.set_xlabel("x (px)")
        axis.set_ylabel("y (px)")
        axis.set_xlim(
            0,
            CANVAS_WIDTH
        )
        axis.set_ylim(
            CANVAS_HEIGHT,
            0
        )

    figure.suptitle(
        "Part 2.3: Path Direction Changes "
        "the Non-zero Fill"
    )

    figure.tight_layout()

    figure.savefig(
        RESULT_DIR
        / "part2_3_circle_directions.png",
        dpi=160
    )

    plt.close(figure)


def write_part2_report(
    star_winding,
    same_winding,
    opposite_winding
):
    """Write the results and answers required in Part 2."""

    lines = [
        "Part 2: Inside and Outside",
        "",
        "Sign convention:",
        (
            "In image coordinates, y increases downward. "
            "A downward edge adds +1 and an upward edge "
            "adds -1."
        ),
        (
            "The scanline is evaluated at each pixel center. "
            "A span is filled whenever the running winding "
            "number is non-zero."
        ),
        "",
        "Part 2.2 - five-pointed star:",
        (
            "Winding number at the center: "
            f"{star_winding}"
        ),
        (
            "The center is filled because its winding "
            "number is non-zero."
        ),
        (
            "This differs from filling alternate spans "
            "with the even-odd rule."
        ),
        "",
        "Part 2.3 - two circle subpaths:",
        (
            "Center winding number when both circles "
            "have the same direction: "
            f"{same_winding}"
        ),
        (
            "Center winding number when the inner "
            "circle is reversed: "
            f"{opposite_winding}"
        ),
        (
            "When both circles have the same direction, "
            "their winding contributions add, so the "
            "center is filled."
        ),
        (
            "When the inner circle is reversed, its "
            "contribution cancels the outer circle, so "
            "the center becomes a hole."
        ),
        (
            "No point is moved. Only the visiting order "
            "of the inner-circle points is reversed."
        ),
        (
            "Therefore, a font must store the inner and "
            "outer contours of the letter O in opposite "
            "directions so that the non-zero winding "
            "rule leaves the center unfilled."
        )
    ]

    report_path = (
        RESULT_DIR
        / "part2_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def run_part2():
    """Run all experiments required in Part 2."""

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Part 2.2: self-intersecting five-pointed star
    # ---------------------------------------------------------

    star_center = np.array([
        400.0,
        300.0
    ])

    star_path = make_five_point_star(
        star_center,
        radius=230
    )

    star_image = scanline_fill(
        [star_path],
        CANVAS_WIDTH,
        CANVAS_HEIGHT
    )

    star_winding = winding_number_at_point(
        star_center,
        [star_path]
    )

    save_star_result(
        star_image,
        star_winding
    )

    # ---------------------------------------------------------
    # Part 2.3: outer and inner circle subpaths
    # ---------------------------------------------------------

    circle_center = np.array([
        400.0,
        300.0
    ])

    outer_segments = (
        make_circle_bezier_segments(
            circle_center,
            radius=230
        )
    )

    inner_segments = (
        make_circle_bezier_segments(
            circle_center,
            radius=105
        )
    )

    # Flatten both circles using Part 1's midpoint method.
    outer_path = flatten_bezier_segments(
        outer_segments,
        depth=6
    )

    inner_path = flatten_bezier_segments(
        inner_segments,
        depth=6
    )

    same_direction_paths = [
        outer_path,
        inner_path
    ]

    # Only reverse the order of the inner-circle points.
    # Their positions remain unchanged.
    opposite_direction_paths = [
        outer_path,
        inner_path[::-1].copy()
    ]

    same_image = scanline_fill(
        same_direction_paths,
        CANVAS_WIDTH,
        CANVAS_HEIGHT
    )

    opposite_image = scanline_fill(
        opposite_direction_paths,
        CANVAS_WIDTH,
        CANVAS_HEIGHT
    )

    same_winding = winding_number_at_point(
        circle_center,
        same_direction_paths
    )

    opposite_winding = winding_number_at_point(
        circle_center,
        opposite_direction_paths
    )

    save_circle_comparison(
        same_image,
        opposite_image,
        same_winding,
        opposite_winding
    )

    write_part2_report(
        star_winding,
        same_winding,
        opposite_winding
    )

    print("Part 2 finished.")

    print(
        "Star center winding number:",
        star_winding
    )

    print(
        "Same-direction circle center "
        "winding number:",
        same_winding
    )

    print(
        "Opposite-direction circle center "
        "winding number:",
        opposite_winding
    )

    print(
        f"Results saved in: {RESULT_DIR}"
    )


if __name__ == "__main__":
    run_part2()