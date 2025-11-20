import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def composite_trajectory(figs, labels=None, mark_endpoints=True, title='Composite Trajectories', show=False):
    """
    Take a list of matplotlib Figure objects (each containing a single 3D Axes
    with one trajectory line from twobody_ODE) and draw them onto a new figure.

    Args:
        figs   : list[Figure]
        labels : list[str] or None (optional label per figure)
        mark_endpoints : bool
        title  : str

    Returns:
        new_fig : matplotlib.figure.Figure
    """

    new_fig = plt.figure()
    new_ax = new_fig.add_subplot(111, projection='3d')

    handles = []
    for i, f in enumerate(figs):
        src_ax = f.axes[0]  # assume single 3D axes
        # copy each line from source figure
        for j, line in enumerate(src_ax.get_lines()):
            # Line3D supports get_data_3d()
            try:
                x, y, z = line.get_data_3d()
            except Exception:
                # fallback to 2D 
                x, y = line.get_data()
                z = np.zeros_like(x)

            lbl = labels[i] if (labels and j == 0) else None
            h, = new_ax.plot(x, y, z,
                             linewidth=line.get_linewidth(),
                             linestyle=line.get_linestyle(),
                             marker=None,
                             label=lbl)
            handles.append(h)

            if mark_endpoints and x.size > 0:
                new_ax.scatter(x[0],  y[0],  z[0],  s=30)
                new_ax.scatter(x[-1], y[-1], z[-1], s=30)

    new_ax.set_xlabel('X (km)')
    new_ax.set_ylabel('Y (km)')
    new_ax.set_zlabel('Z (km)')
    new_ax.set_title(title)

    # legend if any labels provided
    if labels:
        # keep only the first handle per label
        used = {}
        uniq_handles = []
        for h in handles:
            if h.get_label() and h.get_label() not in used:
                used[h.get_label()] = True
                uniq_handles.append(h)
        if uniq_handles:
            new_ax.legend(handles=uniq_handles, loc='best')
    new_ax.set_box_aspect([1, 1, 1])
    # show plot
    for fig in figs:
        plt.close(fig)
    if show == True:
        plt.show()

    return new_fig
