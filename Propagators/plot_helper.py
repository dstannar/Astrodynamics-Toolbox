import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3D

from MathHelpers.constants import AU, JDaysInSecs  # AU in km
from matplotlib.lines import Line2D


"""
- composite_trajectory: merge per-leg twobody_ODE figures into one figure (backwards compatible)
- animate_composite_figure: animate labeled lines already in a composite fig (backwards compatible)
- stitch helpers: take per-leg solve_ivp outputs and build one continuous (t, r) array
- animate_transfer_samples: animate stitched transfer + planet dots, minimal clutter
"""


# ----------------------------
# small utilities
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
    scatter._offsets3d = (np.asarray([x]), np.asarray([y]), np.asarray([z]))


def _text3d_set_xyz(text_obj, x, y, z):
    # matplotlib 3D text position update
    text_obj.set_position((float(x), float(y)))
    text_obj.set_3d_properties(float(z), zdir='z')


# ----------------------------
# stitching helpers (new)
# ----------------------------

def stitch_leg_solutions(leg_solutions, leg_tofs_sec=None):
    '''
    Build a single continuous trajectory from per-leg solve_ivp outputs.

    Inputs:
        leg_solutions : list
            each element may be:
              - a scipy solve_ivp "solution" object, OR
              - a dict with key "solution" containing that object
        leg_tofs_sec : list[float] or None
            optional sanity check: expected leg duration in seconds

    Outputs:
        t_sec : (N,) ndarray
            stitched time in seconds from start of leg 1
        r_km : (N,3) ndarray
            stitched spacecraft position in km
    '''
    t_all = []
    r_all = []
    t_offset = 0.0

    for i, item in enumerate(leg_solutions):
        sol = item["solution"] if isinstance(item, dict) and "solution" in item else item

        if sol is None or not hasattr(sol, "t") or not hasattr(sol, "y"):
            raise TypeError("each leg solution must be solve_ivp output (or dict with key 'solution')")

        t = np.asarray(sol.t, dtype=float).reshape(-1)
        y = np.asarray(sol.y, dtype=float)

        if y.shape[0] < 3:
            raise ValueError("solve_ivp solution.y must have at least 3 rows for position")

        r = y[:3, :].T  # (N,3)

        # optional duration sanity check
        if leg_tofs_sec is not None:
            exp_dt = float(leg_tofs_sec[i])
            got_dt = float(t[-1] - t[0])
            if abs(got_dt - exp_dt) > 1.0e-3:
                print(f"[plot_helper] warn: leg {i+1} dt mismatch: got {got_dt:.6f} s vs exp {exp_dt:.6f} s")

        # shift time so legs are continuous
        t_all.append(t_offset + (t - t[0]))
        r_all.append(r)

        t_offset = float(t_all[-1][-1])

    t_sec = np.concatenate(t_all)
    r_km = np.vstack(r_all)
    return t_sec, r_km


def sample_planets_on_grid(t_sec, JD0, planet_names, planet_id_map, ephem_func):
    '''
    Sample planets at the same time grid as an animation.

    Inputs:
        t_sec : (N,) ndarray
            seconds from JD0
        JD0 : float
            start epoch as absolute JD (days)
        planet_names : list[str]
            names used as keys in planet_id_map
        planet_id_map : dict[str -> int]
            e.g. {"Earth":3, "Venus":2, ...}
        ephem_func : callable
            ephem_func(planet_id:int, JD:float) -> (r_km(3,), v_kms(3,))

    Outputs:
        planet_r_km_by_name : dict[str -> (N,3) ndarray]
    '''
    t_sec = np.asarray(t_sec, dtype=float).reshape(-1)
    planet_r = {}

    for name in planet_names:
        pid = int(planet_id_map[name])
        rr = np.zeros((len(t_sec), 3), dtype=float)

        for k, tk in enumerate(t_sec):
            jd = float(JD0 + tk / float(JDaysInSecs))
            r_km, _ = ephem_func(pid, jd)
            rr[k, :] = np.asarray(r_km, dtype=float).reshape(3)

        planet_r[name] = rr

    return planet_r


# ----------------------------
# backwards compatible: composite plot
# ----------------------------

def composite_trajectory(
    figs,
    labels=None,
    mark_endpoints=True,
    title="Composite Trajectories",
    show=False,
    use_au=True,
    show_sun=True,
    view_elev=20,
    view_azim=-60,
    **kwargs
):
    '''
    Merge per-leg 3D figures into one composite 3D figure.

    Backwards compatible with older scripts.
    '''
    _ = kwargs  # ignore extra args from older callers

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
    fps=60,
    duration=12,
    background_alpha=0.25,    # alpha for non-active/parking
    active_alpha=1.0,         # alpha for the active line
    save_path=None,           # ".mp4" or ".gif"
    moving_points=None,       # optional: list of moving marker specs (see below)
    show=True
):
    """
    Animate only *selected* lines from an existing composite 3D figure.
    Animated lines are HIDDEN until their turn, then drawn progressively.
    Non-animated (parking) lines are visible the whole time (dim).

    Prefer label_order=['leg1 label', 'leg2 label', ...].

    moving_points (optional)
    ------------------------
    A list of dicts, each like:
      {
        "label": "Earth",
        "xyz": (x, y, z),     # arrays of equal length, in plot units
        "color": "tab:blue",
        "marker": "o",
        "markersize": 6,
        "alpha": 1.0,
      }
    These markers move throughout the full animation timeline (frames_total).
    """
    assert composite_fig.axes, "Composite figure has no axes."
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

    # Optional moving point artists
    point_artists = []
    point_specs = []
    if moving_points:
        for spec in moving_points:
            if not isinstance(spec, dict) or "xyz" not in spec:
                continue
            x, y, z = spec["xyz"]
            x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
            if len(x) == 0 or len(x) != len(y) or len(x) != len(z):
                continue
            pt, = ax.plot(
                [], [], [],
                linestyle="None",
                marker=spec.get("marker", "o"),
                markersize=float(spec.get("markersize", 6)),
                color=spec.get("color", "k"),
                alpha=float(spec.get("alpha", 1.0)),
                label=spec.get("label", None),
            )
            point_artists.append(pt)
            point_specs.append((x, y, z))

    def _point_index_for_frame(f, N):
        if frames_total <= 1 or N <= 1:
            return 0
        return int(np.clip(np.round((float(f) / float(frames_total - 1)) * float(N - 1)), 0, N - 1))

    def init():
        ax.view_init(elev=base_elev, azim=base_azim)
        # initialize moving points to frame 0
        for pt, (x, y, z) in zip(point_artists, point_specs):
            i0 = _point_index_for_frame(0, len(x))
            pt.set_data_3d([x[i0]], [y[i0]], [z[i0]])
        return [ln for _, ln in labeled] + point_artists

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

        # Update moving points for this global frame f
        for pt, (x, y, z) in zip(point_artists, point_specs):
            ii = _point_index_for_frame(f, len(x))
            pt.set_data_3d([x[ii]], [y[ii]], [z[ii]])

        return [ln for _, ln in labeled] + point_artists

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
# new: animate stitched transfer + planet markers (+ labels)
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
    stride=2,
    max_frames=None,
    label_spacecraft=False,
    label_planets=None,
    label_fontsize=10,
    label_offset_frac=0.015,
    # NEW:
    planet_colors=None,     # dict like {"Earth":"C0", "Venus":"C1", ...}
    spacecraft_color="k",   # any matplotlib color
    sun_color="gold",
    legend=True,
    legend_loc="upper right",
):
    '''
    Animate a transfer using pre-sampled positions.
    Clean view: spacecraft dot + short trail + planet dots.

    Added:
      - fixed colors for planet dots + spacecraft dot
      - legend with matching dot colors
    '''
    t_sec = np.asarray(t_sec, dtype=float).reshape(-1)
    sc_r_km = np.asarray(sc_r_km, dtype=float)

    if sc_r_km.ndim != 2 or sc_r_km.shape[1] != 3:
        raise ValueError("sc_r_km must be shape (N,3)")
    if sc_r_km.shape[0] != len(t_sec):
        raise ValueError("t_sec and sc_r_km must have same length")

    if stride is None or int(stride) < 1:
        stride = 1
    stride = int(stride)

    idx = np.arange(0, len(t_sec), stride, dtype=int)
    if max_frames is not None:
        idx = idx[:int(max_frames)]

    t_sec = t_sec[idx]
    sc_r_km = sc_r_km[idx, :]

    planet_r_km_by_name = planet_r_km_by_name or {}
    label_planets = label_planets or []

    au_km = float(AU)
    scale = au_km if use_au else 1.0
    unit = "AU" if use_au else "km"

    sc = sc_r_km / scale
    planets = {k: (np.asarray(v, dtype=float)[idx, :] / scale) for k, v in planet_r_km_by_name.items()}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # faint full path so you always see the “shape”
    ax.plot(sc[:, 0], sc[:, 1], sc[:, 2], linewidth=1.0, alpha=0.25)

    sun_dot = None
    if show_sun:
        sun_dot = ax.scatter([0.0], [0.0], [0.0], s=70, marker="o", color=sun_color)

    trail_line, = ax.plot([], [], [], linewidth=2.0)

    # spacecraft dot (explicit color)
    sc_dot = ax.scatter([sc[0, 0]], [sc[0, 1]], [sc[0, 2]], s=55, color=spacecraft_color)

    # choose colors for planets
    # - if user provides planet_colors dict, use it
    # - else auto-assign from matplotlib color cycle
    planet_colors = planet_colors or {}
    cycle = plt.rcParams.get("axes.prop_cycle", None)
    cycle_colors = cycle.by_key().get("color", []) if cycle is not None else []
    if not cycle_colors:
        cycle_colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

    # stable deterministic assignment order
    planet_names_sorted = list(planets.keys())
    planet_color_map = {}
    for i_name, name in enumerate(planet_names_sorted):
        if name in planet_colors:
            planet_color_map[name] = planet_colors[name]
        else:
            planet_color_map[name] = cycle_colors[i_name % len(cycle_colors)]

    planet_dots = {}
    for name, arr in planets.items():
        planet_dots[name] = ax.scatter(
            [arr[0, 0]], [arr[0, 1]], [arr[0, 2]],
            s=40,
            color=planet_color_map.get(name, "C0")
        )

    # limits from sc + planets, equal aspect
    X = [sc[:, 0]]
    Y = [sc[:, 1]]
    Z = [sc[:, 2]]
    for arr in planets.values():
        if arr.shape == sc.shape:
            X.append(arr[:, 0])
            Y.append(arr[:, 1])
            Z.append(arr[:, 2])

    _set_equal_xyz_limits(ax, np.concatenate(X), np.concatenate(Y), np.concatenate(Z), pad=0.05)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit})")
    ax.set_title(title)

    # legend (proxy artists so legend markers match dot colors reliably)
    if legend:
        handles = []

        if show_sun:
            handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                  markerfacecolor=sun_color, markeredgecolor='none',
                                  markersize=8, label="Sun"))

        handles.append(Line2D([0], [0], marker='o', linestyle='None',
                              markerfacecolor=spacecraft_color, markeredgecolor='none',
                              markersize=8, label="Spacecraft"))

        for name in planet_names_sorted:
            c = planet_color_map[name]
            handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                  markerfacecolor=c, markeredgecolor='none',
                                  markersize=8, label=name))

        ax.legend(handles=handles, loc=legend_loc)

    # label offsets: small fraction of plot span
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0]))
    d = float(label_offset_frac) * float(span)

    # text labels
    sc_text = None
    if label_spacecraft:
        sc_text = ax.text(sc[0, 0] + d, sc[0, 1] + d, sc[0, 2] + d, "SC", fontsize=label_fontsize)

    planet_text = {}
    for name in label_planets:
        if name in planets:
            arr = planets[name]
            planet_text[name] = ax.text(arr[0, 0] + d, arr[0, 1] + d, arr[0, 2] + d, name, fontsize=label_fontsize)

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    N = len(t_sec)

    def init():
        trail_line.set_data_3d([], [], [])
        _scatter_set_xyz(sc_dot, sc[0, 0], sc[0, 1], sc[0, 2])

        if sc_text is not None:
            _text3d_set_xyz(sc_text, sc[0, 0] + d, sc[0, 1] + d, sc[0, 2] + d)

        for name, arr in planets.items():
            _scatter_set_xyz(planet_dots[name], arr[0, 0], arr[0, 1], arr[0, 2])
            if name in planet_text:
                _text3d_set_xyz(planet_text[name], arr[0, 0] + d, arr[0, 1] + d, arr[0, 2] + d)

        time_text.set_text("")
        return []

    def update(i):
        i = int(i)
        i = max(0, min(N - 1, i))

        j0 = max(0, i - int(trail_points))
        trail = sc[j0:i + 1]
        trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        _scatter_set_xyz(sc_dot, sc[i, 0], sc[i, 1], sc[i, 2])

        if sc_text is not None:
            _text3d_set_xyz(sc_text, sc[i, 0] + d, sc[i, 1] + d, sc[i, 2] + d)

        for name, arr in planets.items():
            if arr.shape == sc.shape:
                _scatter_set_xyz(planet_dots[name], arr[i, 0], arr[i, 1], arr[i, 2])
                if name in planet_text:
                    _text3d_set_xyz(planet_text[name], arr[i, 0] + d, arr[i, 1] + d, arr[i, 2] + d)

        days = t_sec[i] / 86400.0
        time_text.set_text(f"t = {days:.1f} days")
        return []

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=N,
        interval=1000.0 / float(fps),
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
