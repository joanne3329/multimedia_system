from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from part2 import make_five_point_star, scanline_fill


CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
RESULT_DIR = Path("results/part3")


def render_supersampled(
    subpaths,
    width,
    height,
    k
):
    """
    Render the paths on a canvas enlarged by k.

    After rasterization, every k-by-k block is averaged
    into one output pixel. No built-in anti-aliasing or
    smoothing function is used.
    """

    scaled_paths = [
        np.asarray(path, dtype=float) * k
        for path in subpaths
    ]

    high_resolution = scanline_fill(
        scaled_paths,
        width * k,
        height * k
    )

    high_resolution = (
        high_resolution.astype(float)
        / 255.0
    )

    output = high_resolution.reshape(
        height,
        k,
        width,
        k
    ).mean(axis=(1, 3))

    return output


def save_star_supersampling(images):
    """Save the star rendered at k = 1, 2, 4, and 8."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9)
    )

    for axis, k in zip(
        axes.ravel(),
        [1, 2, 4, 8]
    ):
        # imshow only displays our pixel array.
        # nearest prevents display-time smoothing.
        axis.imshow(
            images[k],
            cmap="gray",
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="nearest"
        )

        axis.set_title(
            f"k = {k}"
        )

        axis.set_xlim(
            100,
            700
        )

        axis.set_ylim(
            560,
            40
        )

        axis.set_xlabel(
            "x (px)"
        )

        axis.set_ylabel(
            "y (px)"
        )

    figure.suptitle(
        "Part 3.I: Area Coverage by Supersampling"
    )

    figure.tight_layout()

    figure.savefig(
        RESULT_DIR
        / "part3_1_star_supersampling.png",
        dpi=160
    )

    plt.close(figure)


def choose_measurement_row(star_path):
    """
    Find the shallowest star edge and choose a pixel row
    that crosses it near one endpoint.
    """

    best_edge = None
    best_slope = np.inf

    for p0, p1 in zip(
        star_path[:-1],
        star_path[1:]
    ):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]

        if abs(dx) < 1e-12:
            continue

        slope = abs(
            dy / dx
        )

        if slope < best_slope:
            best_slope = slope
            best_edge = (
                p0,
                p1
            )

    p0, p1 = best_edge

    # Select a point near one endpoint of the edge.
    near_end = (
        p0 + 0.18 * (p1 - p0)
    )

    row = int(
        np.clip(
            np.floor(
                near_end[1]
            ),
            0,
            CANVAS_HEIGHT - 1
        )
    )

    # Calculate where the edge crosses the center
    # of the selected pixel row.
    edge_x_at_row = (
        p0[0]
        + (
            (row + 0.5 - p0[1])
            / (p1[1] - p0[1])
        )
        * (p1[0] - p0[0])
    )

    # Only plot a small region around the selected
    # edge so that other crossings are excluded.
    start_x = max(
        0,
        int(edge_x_at_row) - 40
    )

    end_x = min(
        CANVAS_WIDTH,
        int(edge_x_at_row) + 41
    )

    return (
        row,
        start_x,
        end_x,
        p0,
        p1
    )


def save_row_profile(
    images,
    row,
    start_x,
    end_x
):
    """
    Compare pixel values along the selected row
    for k=1 and k=8.
    """

    x_values = np.arange(
        start_x,
        end_x
    )

    plt.figure(
        figsize=(9, 4.8)
    )

    # These functions are only used to plot the
    # measurements required by the assignment.
    plt.step(
        x_values,
        images[1][
            row,
            start_x:end_x
        ],
        where="mid",
        linewidth=1.6,
        label="k = 1"
    )

    plt.plot(
        x_values,
        images[8][
            row,
            start_x:end_x
        ],
        linewidth=2,
        label="k = 8"
    )

    plt.title(
        "Part 3.II: Pixel Values "
        f"Along Row y = {row}"
    )

    plt.xlabel(
        "x coordinate (px)"
    )

    plt.ylabel(
        "pixel value / area coverage"
    )

    plt.ylim(
        -0.05,
        1.05
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "part3_2_row_profile.png",
        dpi=160
    )

    plt.close()


def save_render_times(render_times):
    """Plot the measured render time for every k value."""

    k_values = np.array([
        1,
        2,
        4,
        8
    ])

    times = np.array([
        render_times[k]
        for k in k_values
    ])

    plt.figure(
        figsize=(7, 4.5)
    )

    # Plotting measurements is allowed by the assignment.
    plt.plot(
        k_values,
        times,
        "o-",
        linewidth=2
    )

    plt.xlabel(
        "supersampling factor k"
    )

    plt.ylabel(
        "render time (seconds)"
    )

    plt.title(
        "Part 3.II: Render Time on "
        f"{CANVAS_WIDTH} x "
        f"{CANVAS_HEIGHT} Canvas"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "part3_2_render_times.png",
        dpi=160
    )

    plt.close()


def make_wheel(
    center,
    radius,
    wedge_count=48
):
    """
    Create a wheel using separate triangular subpaths.

    Each wedge and each gap occupy one half of an
    angular slot, so the gaps are as wide as the wedges.
    """

    center = np.asarray(
        center,
        dtype=float
    )

    slot_angle = (
        2 * np.pi
        / wedge_count
    )

    wedge_angle = (
        slot_angle / 2
    )

    subpaths = []

    for index in range(
        wedge_count
    ):
        start_angle = (
            -np.pi / 2
            + index * slot_angle
        )

        end_angle = (
            start_angle
            + wedge_angle
        )

        p1 = (
            center
            + radius
            * np.array([
                np.cos(start_angle),
                np.sin(start_angle)
            ])
        )

        p2 = (
            center
            + radius
            * np.array([
                np.cos(end_angle),
                np.sin(end_angle)
            ])
        )

        triangle = np.array([
            center,
            p1,
            p2,
            center
        ])

        subpaths.append(
            triangle
        )

    return subpaths


def save_wheel_comparison(
    wheel_k1,
    wheel_k8,
    wedge_count
):
    """Save the wheel results for k=1 and k=8."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5)
    )

    axes[0].imshow(
        wheel_k1,
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="nearest"
    )

    axes[0].set_title(
        "k = 1: aliasing near the center"
    )

    axes[1].imshow(
        wheel_k8,
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="nearest"
    )

    axes[1].set_title(
        "k = 8: filtered area coverage"
    )

    for axis in axes:
        axis.set_xlim(
            120,
            680
        )

        axis.set_ylim(
            580,
            20
        )

        axis.set_xlabel(
            "x (px)"
        )

        axis.set_ylabel(
            "y (px)"
        )

    figure.suptitle(
        "Part 3.III: Wheel with "
        f"{wedge_count} Wedges"
    )

    figure.tight_layout()

    figure.savefig(
        RESULT_DIR
        / "part3_3_wheel_aliasing.png",
        dpi=160
    )

    plt.close(figure)


def write_part3_report(
    row,
    edge_start,
    edge_end,
    render_times,
    wedge_count
):
    """Write all measurements and answers for Part 3."""

    ratio_1_to_2 = (
        render_times[2]
        / render_times[1]
    )

    ratio_2_to_4 = (
        render_times[4]
        / render_times[2]
    )

    ratio_4_to_8 = (
        render_times[8]
        / render_times[4]
    )

    lines = [
        "Part 3: Landing on the Pixel Grid",
        "",
        "Part 3.I - area coverage:",
        (
            "The five-pointed star was rotated by "
            "5 degrees so that no edge is exactly "
            "horizontal or vertical."
        ),
        (
            "For each k, the path and canvas were "
            "scaled by k and filled using the Part 2 "
            "scanline algorithm."
        ),
        (
            "Each k-by-k block was then averaged "
            "into one output pixel. No built-in "
            "anti-aliasing or smoothing function was used."
        ),
        "",
        "Part 3.II - measurement:",
        f"Selected pixel row: y = {row}",
        (
            "Shallow edge endpoints: "
            f"{edge_start.tolist()} to "
            f"{edge_end.tolist()}"
        ),
        (
            "At k=1, the profile contains an abrupt "
            "binary step with values of only 0 or 1."
        ),
        (
            "At k=8, intermediate coverage values appear, "
            "producing a smoother transition along the edge."
        ),
        (
            f"Canvas size: {CANVAS_WIDTH} x "
            f"{CANVAS_HEIGHT} pixels"
        ),
        "",
        "Render times:"
    ]

    for k in [
        1,
        2,
        4,
        8
    ]:
        lines.append(
            f"k={k}: "
            f"{render_times[k]:.6f} seconds"
        )

    lines.extend([
        "",
        "Measured growth when k doubles:",
        (
            "k=1 to k=2: "
            f"{ratio_1_to_2:.3f} times"
        ),
        (
            "k=2 to k=4: "
            f"{ratio_2_to_4:.3f} times"
        ),
        (
            "k=4 to k=8: "
            f"{ratio_4_to_8:.3f} times"
        ),
        (
            "The enlarged image contains k^2 times as "
            "many sample positions. Therefore, doubling "
            "k produces four times as many high-resolution "
            "pixels."
        ),
        (
            "The measured wall-clock time does not "
            "necessarily increase by exactly four times "
            "because NumPy processes row slices and block "
            "averaging using vector operations."
        ),
        "",
        "Part 3.III - where the grid loses:",
        (
            f"The wheel contains {wedge_count} wedge "
            "subpaths, with gaps as wide as the wedges."
        ),
        (
            "(a) At k=1, the center shows aliasing, "
            "including false and unstable radial patterns "
            "rather than only jagged edges."
        ),
        (
            "(b) Moving toward the center, the wedges "
            "and gaps become narrower, so their spatial "
            "frequency increases."
        ),
        (
            "Once the spatial frequency exceeds the "
            "Nyquist limit of the final pixel grid, the "
            "Sampling Theorem says that the detail cannot "
            "be represented or recovered."
        ),
        (
            "Increasing k cleans the outer edges, but the "
            "center approaches uniform gray instead of "
            "recovering the individual wedges."
        ),
        (
            "(c) A low-pass anti-aliasing filter is "
            "relevant here."
        ),
        (
            "Averaging every k-by-k block already applies "
            "a box low-pass filter before downsampling."
        ),
        (
            "It suppresses misleading high-frequency "
            "aliases and produces average area coverage, "
            "but it cannot recover details that the final "
            "pixel grid cannot represent."
        )
    ])

    report_path = (
        RESULT_DIR
        / "part3_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def run_part3():
    """Run every experiment required in Part 3."""

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Part 3.I: rotated star and supersampling
    # ---------------------------------------------------------

    star_center = np.array([
        400.0,
        300.0
    ])

    # The Part 2 star used -90 degrees.
    # Adding 5 degrees gives -85 degrees.
    rotated_star = make_five_point_star(
        star_center,
        radius=230,
        rotation_degrees=-85
    )

    star_images = {}
    render_times = {}

    for k in [
        1,
        2,
        4,
        8
    ]:
        start_time = perf_counter()

        star_images[k] = (
            render_supersampled(
                [rotated_star],
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
                k
            )
        )

        render_times[k] = (
            perf_counter()
            - start_time
        )

    save_star_supersampling(
        star_images
    )

    # ---------------------------------------------------------
    # Part 3.II: row profile and execution times
    # ---------------------------------------------------------

    (
        row,
        start_x,
        end_x,
        edge_start,
        edge_end
    ) = choose_measurement_row(
        rotated_star
    )

    save_row_profile(
        star_images,
        row,
        start_x,
        end_x
    )

    save_render_times(
        render_times
    )

    # ---------------------------------------------------------
    # Part 3.III: high-frequency wheel
    # ---------------------------------------------------------

    wedge_count = 48

    wheel_paths = make_wheel(
        star_center,
        radius=250,
        wedge_count=wedge_count
    )

    wheel_k1 = render_supersampled(
        wheel_paths,
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
        1
    )

    wheel_k8 = render_supersampled(
        wheel_paths,
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
        8
    )

    save_wheel_comparison(
        wheel_k1,
        wheel_k8,
        wedge_count
    )

    write_part3_report(
        row,
        edge_start,
        edge_end,
        render_times,
        wedge_count
    )

    print("Part 3 finished.")
    print(
        f"Measured row: y = {row}"
    )

    for k in [
        1,
        2,
        4,
        8
    ]:
        print(
            f"k={k}: "
            f"{render_times[k]:.6f} seconds"
        )

    print(
        f"Results saved in: {RESULT_DIR}"
    )


if __name__ == "__main__":
    run_part3()