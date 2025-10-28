from typing import Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
import numpy as np


def plot_points(
    fig: Figure,
    points: np.ndarray,
    i_param: Optional[np.ndarray] = None,
    size: float = 10.0,
) -> Tuple[PathCollection, Axes]:
    """
    Scatter plot for 2D or 3D point data with a coloring.

    :param fig: Matplotlib Figure object where the scatter plot will be drawn.
    :param points: Array of shape (n_points, 2) or (n_points, 3) containing the coordinates of each point.
    :param i_param: Optional array of intrinsic parameters to color the plot with. (default: None).
    :param size: Marker size for each point in the scatter plot. (default: 10).

    :return: The PathCollection object for the scatter plot.
    """
    dim = points.shape[1]

    if dim not in (2, 3):
        raise ValueError(
            f"Points dimension {points.shape[1]}, expected dimension 2 or 3!"
        )

    projection = "3d" if dim == 3 else None
    ax = fig.add_subplot(111, projection=projection)

    if dim == 2:
        sc = ax.scatter(points[:, 0], points[:, 1], c=i_param, s=size)
    else:  # 3D
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=i_param)

    if i_param is not None:
        fig.colorbar(sc, ax=ax, label="intrinsic parameter")

    return sc, ax
