from typing import Optional, Sequence, Tuple
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
import numpy as np


def plot_points(
    gr: Figure | Axes,
    points: np.ndarray,
    i_param: Optional[np.ndarray] = None,
    use_dims: Optional[Sequence[int]] = None,
    size: float = 10.0,
) -> Tuple[PathCollection, Axes]:
    """
    Scatter plot for 2D or 3D point data with a coloring.

    :param fig: Matplotlib Figure object where the scatter plot will be drawn.
    :param points: Array of shape (n_points, 2) or (n_points, 3) containing the coordinates of each point.
    :param i_param: Optional array of intrinsic parameters to color the plot with. (default: None).
    :param use_dims: Optional sequence of dimensions to plot. If None, plots all dimensions. (default: None).
    :param size: Marker size for each point in the scatter plot. (default: 10).

    :return: A Tuple of PathCollection and Axes objects for the scatter plot.
    """
    if use_dims is None:
        use_dims = range(points.shape[1])

    if len(use_dims) not in (2, 3):
        raise ValueError(
            f"Points dimension {len(use_dims)}, expected dimension 2 or 3!"
        )

    projection = "3d" if len(use_dims) == 3 else None
    ax = gr.add_subplot(111, projection=projection) if isinstance(gr, Figure) else gr

    point_arrs = (
        points[:, use_dims[0]],
        points[:, use_dims[1]],
        size if len(use_dims) == 2 else points[:, use_dims[2]],
    )

    sc = ax.scatter(*point_arrs, c=i_param)

    if i_param is not None and isinstance(gr, Figure):
        gr.colorbar(sc, ax=ax, label="intrinsic parameter")

    return sc, ax
