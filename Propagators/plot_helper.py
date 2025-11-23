import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
chatgpt was used in partial creation of this file as my matplotlib animation skills need some work. i will fix it soon!

'''

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
                x = np.asarray(x)
                y = np.asarray(y)
                z = np.zeros_like(x)

            # ensure arrays for downstream ops
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)

            lbl = labels[i] if (labels and j == 0) else None
            h, = new_ax.plot(x, y, z,
                             linewidth=line.get_linewidth(),
                             linestyle=line.get_linestyle(),
                             marker=None,
                             label=lbl)
            handles.append(h)

            if mark_endpoints and len(x) > 0:
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

def _get_xyz_from_line(line):
    if hasattr(line, "get_data_3d"):
        x, y, z = line.get_data_3d()
        return np.asarray(x), np.asarray(y), np.asarray(z)
    if hasattr(line, "_verts3d"):
        x, y, z = line._verts3d
        return np.asarray(x), np.asarray(y), np.asarray(z)
    x = np.asarray(line.get_xdata())
    y = np.asarray(line.get_ydata())
    z = np.asarray(line.get_zdata()) if hasattr(line, "get_zdata") else np.asarray(line.get_3d_properties())
    return x, y, z

def animate_composite_figure(
    composite_fig,
    animate_mask=None,        # legacy: index-based; ignored if label_order is provided
    order=None,               # legacy: index-based; ignored if label_order is provided
    label_order=None,         # preferred: list of labels to animate in sequence
    fps=30,
    duration=12,
    background_alpha=0.25,    # alpha for non-active/parking
    active_alpha=1.0,         # alpha for the active line
    save_path=None,           # ".mp4" or ".gif"
    show=True
):
    """
    Animate only *selected* lines from an existing composite 3D figure.
    Animated lines are HIDDEN until their turn, then drawn progressively.
    Non-animated (parking) lines are visible the whole time (dim).

    Prefer label_order=['leg1 label', 'leg2 label', ...].
    """
    assert composite_fig.axes, "Composite figure has no axes."
    ax = composite_fig.axes[0]
    if not isinstance(ax, Axes3D):
        raise ValueError("Composite figure must be 3D (projection='3d').")

    # Grab labeled 3D lines (the ones you set via composite_trajectory labels)
    all_lines = [ln for ln in ax.get_lines() if isinstance(ln, Line3D)]
    labeled = [(ln.get_label(), ln) for ln in all_lines if ln.get_label() not in (None, "", "_nolegend_")]
    if not labeled:
        raise ValueError("No labeled 3D lines found to animate. Ensure composite_trajectory provided labels.")

    # Map label -> line & data
    label2line = {}
    label2xyz  = {}
    for lab, ln in labeled:
        x, y, z = _get_xyz_from_line(ln)
        label2line[lab] = ln
        label2xyz[lab]  = (x, y, z)

    # Determine animation sequence (labels)
    if label_order is None:
        # Use current labeled order
        label_order = [lab for lab, _ in labeled]

    # Active set = will be animated; any labeled line NOT in label_order is treated as parking (static)
    animated_labels = [lab for lab in label_order if lab in label2line]
    parking_labels  = [lab for lab, _ in labeled if lab not in animated_labels]

    if not animated_labels:
        # Nothing to animate; just show figure
        if show:
            plt.show()
        return composite_fig, None

    # Precompute lengths
    Ns = {lab: len(label2xyz[lab][0]) for lab in animated_labels}

    # Allocate frames proportional to length so draw speed feels consistent
    frames_total = max(1, int(round(duration * fps)))
    lengths = np.array([Ns[lab] for lab in animated_labels], dtype=float)
    if lengths.sum() == 0:
        lengths[:] = 1.0
    weights = lengths / lengths.sum()
    frames_per = np.maximum(1, np.round(weights * frames_total).astype(int))
    diff = frames_total - int(frames_per.sum())
    if diff != 0:
        frames_per[0] += diff
    cum = np.cumsum(frames_per)

    def which_segment(f):
        return int(np.searchsorted(cum, f, side='right'))

    def local_index(f, seg):
        start = 0 if seg == 0 else cum[seg - 1]
        end   = cum[seg]
        span  = max(1, end - start)
        frac  = (f - start + 1) / span
        Nseg  = Ns[animated_labels[seg]]
        return int(np.clip(np.round(frac * Nseg), 1, Nseg))

    # ---------- INIT: parking visible; animated HIDDEN until their turn ----------
    base_elev, base_azim = ax.elev, ax.azim

    for lab in parking_labels:
        ln = label2line[lab]
        x, y, z = label2xyz[lab]
        ln.set_alpha(background_alpha)
        ln.set_data_3d(x, y, z)  # fully visible

    for i, lab in enumerate(animated_labels):
        ln = label2line[lab]
        x, y, z = label2xyz[lab]
        if i == 0:
            # first animated segment starts at an empty path (revealed by update)
            ln.set_alpha(active_alpha)
            ln.set_data_3d([], [], [])
        else:
            # all future segments: completely hidden & empty
            ln.set_alpha(0.0)
            ln.set_data_3d([], [], [])

    def init():
        ax.view_init(elev=base_elev, azim=base_azim)
        return [ln for _, ln in labeled]

    def update(f):
        seg = which_segment(f)
        seg = min(seg, len(animated_labels) - 1)

        # Completed segments: show full path, dimmed
        for k in range(seg):
            lab = animated_labels[k]
            ln  = label2line[lab]
            x, y, z = label2xyz[lab]
            ln.set_alpha(background_alpha)
            ln.set_data_3d(x, y, z)

        # Current segment: reveal (if hidden) and draw partial
        curr_lab = animated_labels[seg]
        curr_ln  = label2line[curr_lab]
        x, y, z  = label2xyz[curr_lab]
        n = local_index(f, seg)
        curr_ln.set_alpha(active_alpha)
        curr_ln.set_data_3d(x[:n], y[:n], z[:n])

        # Future segments: keep completely hidden & empty
        for k in range(seg + 1, len(animated_labels)):
            lab = animated_labels[k]
            ln  = label2line[lab]
            ln.set_alpha(0.0)
            ln.set_data_3d([], [], [])

        return [ln for _, ln in labeled]

    anim = FuncAnimation(
        composite_fig, update, init_func=init,
        frames=frames_per.sum(), interval=1000.0 / float(fps), blit=False
    )

    if save_path:
        try:
            if save_path.lower().endswith(".mp4"):
                try:
                    writer = FFMpegWriter(fps=fps, bitrate=1800)
                    anim.save(save_path, writer=writer)
                except Exception:
                    # fallback to GIF if ffmpeg missing
                    writer = PillowWriter(fps=fps)
                    anim.save(save_path.replace(".mp4", ".gif"), writer=writer)
            elif save_path.lower().endswith(".gif"):
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            else:
                raise ValueError("save_path must end with .mp4 or .gif")
        except Exception as e:
            print(f"Could not save animation to '{save_path}': {e}")

    if show:
        plt.show()

    return composite_fig, anim
