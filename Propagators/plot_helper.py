import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3D

from MathHelpers.constants import AU  # km


"""
notes to self:
- composite_trajectory: merge per-leg twobody_ODE figures into one figure (backwards compatible)
- animate_composite_figure: animate lines already in a composite fig (backwards compatible)
- animate_transfer_samples: new: animate a stitched transfer + planet markers, minimal clutter
- big visual fixes: plot in AU + force equal xyz data scaling + set a consistent view angle
"""


# ----------------------------
# tiny utilities
# ----------------------------

def _get_xyz_from_line(line):
    if hasattr(line, "get_data_3d"):
        x, y, z = line.get_data_3d()
        return np.asarray(x), np.asarray(y), np.asarray(z)
    if hasattr(line, "_verts3d"):
        x, y, z = line._verts3d
        return np.asarray(x), np.asarray(y), np.asarray(z)

    x = np.asarray(line.get_xdata())
    y = np.asarray(line.get_ydata())
    if hasattr(line, "get_zdata"):
        z = np.asarray(line.get_zdata())
    else:
        z = np.asarray(line.get_3d_properties())
    return x, y, z


def _set_equal_xyz_limits(ax, X, Y, Z, pad=0.05):
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    xmin, xmax = float(np.min(X)), float(np.max(X))
    ymin, ymax = float(np.min(Y)), float(np.max(Y))
    zmin, zmax = float(np.min(Z)), float(np.max(Z))

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    rx = 0.5 * (xmax - xmin)
    ry = 0.5 * (ymax - ymin)
    rz = 0.5 * (zmax - zmin)

    r = max(rx, ry, rz)
    if r <= 0.0:
        r = 1.0

    r *= (1.0 + float(pad))

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def _scatter_set_xyz(scatter, x, y, z):
    # matplotlib 3D scatter update
    scatter._offsets3d = (np.asarray([x]), np.asarray([y]), np.asarray([z]))


# ----------------------------
# backwards compatible: composite plot
# ----------------------------

def composite_trajectory(
    figs,
    labels=None,
    mark_endpoints=True,
    title="Composite Trajectories",
    show=False,
    # new optional knobs (defaults are safe; old callers won't break)
    use_au=True,
    show_sun=True,
    view_elev=20,
    view_azim=-60,
    **kwargs
):
    """
    take a list of matplotlib Figure objects (each assumed to contain a single 3D axes
    with trajectory line(s) from twobody_ODE) and draw them onto a new figure.

    args kept for backwards compatibility:
      figs, labels, mark_endpoints, title, show

    new optional args (safe defaults):
      use_au: plot in AU instead of km
      show_sun: draw origin marker
      view_elev/view_azim: camera angle
    """
    _ = kwargs  # ignore unknown stuff from older scripts

    new_fig = plt.figure()
    new_ax = new_fig.add_subplot(111, projection="3d")

    au_km = float(AU)
    all_X, all_Y, all_Z = [], [], []
    handles = []

    for i, f in enumerate(figs):
        if not f.axes:
            continue
        src_ax = f.axes[0]

        for j, line in enumerate(src_ax.get_lines()):
            try:
                x, y, z = line.get_data_3d()
            except Exception:
                x, y = line.get_data()
                x = np.asarray(x)
                y = np.asarray(y)
                z = np.zeros_like(x)

            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)

            if use_au:
                x = x / au_km
                y = y / au_km
                z = z / au_km

            lbl = labels[i] if (labels and i < len(labels) and j == 0) else None

            h, = new_ax.plot(
                x, y, z,
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                marker=None,
                label=lbl
            )
            handles.append(h)

            if mark_endpoints and len(x) > 0:
                new_ax.scatter(x[0], y[0], z[0], s=30)
                new_ax.scatter(x[-1], y[-1], z[-1], s=30)

            if len(x) > 0:
                all_X.append(x)
                all_Y.append(y)
                all_Z.append(z)

    if show_sun:
        new_ax.scatter([0.0], [0.0], [0.0], s=60, marker="o")

    unit = "AU" if use_au else "km"
    new_ax.set_xlabel(f"X ({unit})")
    new_ax.set_ylabel(f"Y ({unit})")
    new_ax.set_zlabel(f"Z ({unit})")
    new_ax.set_title(title)

    if labels:
        used = {}
        uniq_handles = []
        for h in handles:
            lab = h.get_label()
            if lab and lab not in ("_nolegend_", "") and lab not in used:
                used[lab] = True
                uniq_handles.append(h)
        if uniq_handles:
            new_ax.legend(handles=uniq_handles, loc="best")

    if all_X:
        X = np.concatenate(all_X)
        Y = np.concatenate(all_Y)
        Z = np.concatenate(all_Z)
        _set_equal_xyz_limits(new_ax, X, Y, Z, pad=0.05)

    new_ax.set_box_aspect([1, 1, 1])
    new_ax.view_init(elev=float(view_elev), azim=float(view_azim))

    for fig in figs:
        plt.close(fig)

    if show:
        plt.show()

    return new_fig


# ----------------------------
# backwards compatible: animate labeled lines inside a composite fig
# ----------------------------

def animate_composite_figure(
    composite_fig,
    animate_mask=None,        # legacy
    order=None,               # legacy
    label_order=None,         # preferred
    fps=30,
    duration=12,
    background_alpha=0.25,
    active_alpha=1.0,
    save_path=None,
    show=True
):
    """
    animate only selected lines from an existing composite 3D figure.
    (this is your existing behavior; kept so other scripts don't break)
    """
    assert composite_fig.axes, "composite figure has no axes"
    ax = composite_fig.axes[0]
    if not isinstance(ax, Axes3D):
        raise ValueError("composite figure must be 3D (projection='3d')")

    all_lines = [ln for ln in ax.get_lines() if isinstance(ln, Line3D)]
    labeled = [(ln.get_label(), ln) for ln in all_lines if ln.get_label() not in (None, "", "_nolegend_")]
    if not labeled:
        raise ValueError("no labeled 3D lines found to animate; pass labels into composite_trajectory")

    label2line = {}
    label2xyz = {}
    for lab, ln in labeled:
        x, y, z = _get_xyz_from_line(ln)
        label2line[lab] = ln
        label2xyz[lab] = (x, y, z)

    if label_order is None:
        label_order = [lab for lab, _ in labeled]

    animated_labels = [lab for lab in label_order if lab in label2line]
    parking_labels = [lab for lab, _ in labeled if lab not in animated_labels]

    if not animated_labels:
        if show:
            plt.show()
        return composite_fig, None

    Ns = {lab: len(label2xyz[lab][0]) for lab in animated_labels}

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
        return int(np.searchsorted(cum, f, side="right"))

    def local_index(f, seg):
        start = 0 if seg == 0 else cum[seg - 1]
        end = cum[seg]
        span = max(1, end - start)
        frac = (f - start + 1) / span
        Nseg = Ns[animated_labels[seg]]
        return int(np.clip(np.round(frac * Nseg), 1, Nseg))

    base_elev, base_azim = ax.elev, ax.azim

    for lab in parking_labels:
        ln = label2line[lab]
        x, y, z = label2xyz[lab]
        ln.set_alpha(background_alpha)
        ln.set_data_3d(x, y, z)

    for i, lab in enumerate(animated_labels):
        ln = label2line[lab]
        if i == 0:
            ln.set_alpha(active_alpha)
            ln.set_data_3d([], [], [])
        else:
            ln.set_alpha(0.0)
            ln.set_data_3d([], [], [])

    def init():
        ax.view_init(elev=base_elev, azim=base_azim)
        return [ln for _, ln in labeled]

    def update(f):
        seg = which_segment(f)
        seg = min(seg, len(animated_labels) - 1)

        for k in range(seg):
            lab = animated_labels[k]
            ln = label2line[lab]
            x, y, z = label2xyz[lab]
            ln.set_alpha(background_alpha)
            ln.set_data_3d(x, y, z)

        curr_lab = animated_labels[seg]
        curr_ln = label2line[curr_lab]
        x, y, z = label2xyz[curr_lab]
        n = local_index(f, seg)
        curr_ln.set_alpha(active_alpha)
        curr_ln.set_data_3d(x[:n], y[:n], z[:n])

        for k in range(seg + 1, len(animated_labels)):
            lab = animated_labels[k]
            ln = label2line[lab]
            ln.set_alpha(0.0)
            ln.set_data_3d([], [], [])

        return [ln for _, ln in labeled]

    anim = FuncAnimation(
        composite_fig,
        update,
        init_func=init,
        frames=int(frames_per.sum()),
        interval=1000.0 / float(fps),
        blit=False
    )

    if save_path:
        try:
            if save_path.lower().endswith(".mp4"):
                try:
                    writer = FFMpegWriter(fps=fps, bitrate=1800)
                    anim.save(save_path, writer=writer)
                except Exception:
                    writer = PillowWriter(fps=fps)
                    anim.save(save_path.replace(".mp4", ".gif"), writer=writer)
            elif save_path.lower().endswith(".gif"):
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            else:
                raise ValueError("save_path must end with .mp4 or .gif")
        except Exception as e:
            print(f"could not save animation to '{save_path}': {e}")

    if show:
        plt.show()

    return composite_fig, anim


# ----------------------------
# new: animate stitched transfer + planet markers (not crowded)
# ----------------------------

def animate_transfer_samples(
    t_sec,
    sc_r_km,
    planet_r_km_by_name=None,
    title="Transfer Animation",
    use_au=True,
    show_sun=True,
    view_elev=20,
    view_azim=-60,
    trail_points=200,
    fps=30,
    save_path=None,
    show=True,
):
    """
    animate a transfer using pre-sampled positions.
    this stays simple: spacecraft dot + short trail + planet dots.

    inputs:
      t_sec: (N,) seconds from mission start (monotonic)
      sc_r_km: (N,3) spacecraft position in km, heliocentric
      planet_r_km_by_name: dict[str -> (N,3)] planet positions in km sampled on same t grid
      title: plot title
      use_au: convert display to AU
      show_sun: origin marker
      view_elev/view_azim: camera
      trail_points: how many recent points to show behind the spacecraft
      fps: animation fps (used for saving)
      save_path: optional .mp4 or .gif
      show: plt.show()

    outputs:
      (fig, anim)
    """
    t_sec = np.asarray(t_sec, dtype=float).reshape(-1)
    sc_r_km = np.asarray(sc_r_km, dtype=float)
    if sc_r_km.shape[1] != 3:
        raise ValueError("sc_r_km must be shape (N,3)")

    N = len(t_sec)
    if sc_r_km.shape[0] != N:
        raise ValueError("t_sec and sc_r_km must have same length")

    planet_r_km_by_name = planet_r_km_by_name or {}

    au_km = float(AU)
    scale = au_km if use_au else 1.0
    unit = "AU" if use_au else "km"

    # scaled arrays for display
    sc = sc_r_km / scale
    planets = {k: (np.asarray(v, dtype=float) / scale) for k, v in planet_r_km_by_name.items()}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # base geometry: draw full trajectory faintly so you always know where you're headed
    ax.plot(sc[:, 0], sc[:, 1], sc[:, 2], linewidth=1.0, alpha=0.25)

    if show_sun:
        ax.scatter([0.0], [0.0], [0.0], s=60, marker="o")

    # spacecraft artists
    trail_line, = ax.plot([], [], [], linewidth=2.0)   # trailing segment
    sc_dot = ax.scatter([sc[0, 0]], [sc[0, 1]], [sc[0, 2]], s=40)

    # planet artists: one dot per planet, no trails (keeps it clean)
    planet_dots = {}
    for name, arr in planets.items():
        planet_dots[name] = ax.scatter([arr[0, 0]], [arr[0, 1]], [arr[0, 2]], s=30)

    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit})")
    ax.set_title(title)

    # axis limits based on spacecraft + planets, equal scaled
    X = [sc[:, 0]]
    Y = [sc[:, 1]]
    Z = [sc[:, 2]]
    for arr in planets.values():
        if arr.shape[0] == N and arr.shape[1] == 3:
            X.append(arr[:, 0])
            Y.append(arr[:, 1])
            Z.append(arr[:, 2])

    _set_equal_xyz_limits(ax, np.concatenate(X), np.concatenate(Y), np.concatenate(Z), pad=0.05)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    # annotation with time
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        trail_line.set_data_3d([], [], [])
        _scatter_set_xyz(sc_dot, sc[0, 0], sc[0, 1], sc[0, 2])
        for name, arr in planets.items():
            _scatter_set_xyz(planet_dots[name], arr[0, 0], arr[0, 1], arr[0, 2])
        time_text.set_text("")
        return []

    def update(i):
        i = int(i)
        i = max(0, min(N - 1, i))

        # trail window
        j0 = max(0, i - int(trail_points))
        trail = sc[j0:i + 1]

        trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        _scatter_set_xyz(sc_dot, sc[i, 0], sc[i, 1], sc[i, 2])

        for name, arr in planets.items():
            if arr.shape[0] == N:
                _scatter_set_xyz(planet_dots[name], arr[i, 0], arr[i, 1], arr[i, 2])

        # show elapsed days (nice mental scale)
        days = t_sec[i] / 86400.0
        time_text.set_text(f"t = {days:.1f} days")
        return []

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=N,
        interval=1000.0 / 30.0,
        blit=False
    )

    if save_path:
        try:
            if save_path.lower().endswith(".mp4"):
                try:
                    writer = FFMpegWriter(fps=int(fps), bitrate=1800)
                    anim.save(save_path, writer=writer)
                except Exception:
                    writer = PillowWriter(fps=int(fps))
                    anim.save(save_path.replace(".mp4", ".gif"), writer=writer)
            elif save_path.lower().endswith(".gif"):
                writer = PillowWriter(fps=int(fps))
                anim.save(save_path, writer=writer)
            else:
                raise ValueError("save_path must end with .mp4 or .gif")
        except Exception as e:
            print(f"could not save animation to '{save_path}': {e}")

    if show:
        plt.show()

    return fig, anim
