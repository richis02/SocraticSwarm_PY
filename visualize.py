import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import config

FILE = "agents.txt"          # prova anche "agents_c.txt" (swarm compromesso)
FRAME_STRIDE = 4             # un frame ogni 4 passi (0.2 s con dt = 0.05)
TAIL = 60                    # lunghezza scia, in passi

TIME_RE = re.compile(r"---TIME:\s*([\d.]+)---")
POS_RE = re.compile(r"id=(\S+)\s*---\s*pos=\[([^\]]*)\]")


def load(filename):
    """Ritorna ids, times (T,), traj (N, T, 3)."""
    data, times = {}, []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = TIME_RE.match(line)
            if m:
                times.append(float(m.group(1)))
                continue
            m = POS_RE.search(line)
            if m:
                data.setdefault(m.group(1), []).append(np.array(m.group(2).split(), dtype=float))
    ids = sorted(data, key=lambda s: int(s.split("_")[1]))
    return ids, np.array(times), np.array([data[i] for i in ids])


def draw_obstacles_2d(ax, ix, iz):
    for c, r in getattr(config, "OBSTACLES", []):
        ax.add_patch(plt.Circle((c[ix], c[iz]), r, color="gray", alpha=0.35))


def plot_static(ids, traj):
    cols = plt.cm.tab10(np.arange(len(ids)) % 10)
    fig = plt.figure(figsize=(16, 5.5))

    # vista dall'alto: x-z (y è la quota)
    ax = fig.add_subplot(1, 3, 1)
    for k, i in enumerate(ids):
        ax.plot(traj[k, :, 0], traj[k, :, 2], color=cols[k], lw=1.2, label=i)
        ax.plot(*traj[k, 0, [0, 2]], "o", color=cols[k], ms=5)       # partenza
        ax.plot(*traj[k, -1, [0, 2]], "s", color=cols[k], ms=6)      # arrivo
    draw_obstacles_2d(ax, 0, 2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_title("Vista dall'alto (x-z)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)

    # vista laterale: x-y
    ax = fig.add_subplot(1, 3, 2)
    for k in range(len(ids)):
        ax.plot(traj[k, :, 0], traj[k, :, 1], color=cols[k], lw=1.2)
    ax.axhline(config.Y_MIN, color="k", ls="--", lw=0.8)
    ax.axhline(config.Y_MAX, color="k", ls="--", lw=0.8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("Vista laterale (x-y)")
    ax.grid(alpha=0.3)

    # 3D, con y verticale
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    for k in range(len(ids)):
        ax.plot(traj[k, :, 0], traj[k, :, 2], traj[k, :, 1], color=cols[k], lw=1.2)
    for c, r in getattr(config, "OBSTACLES", []):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        ax.plot_surface(c[0] + r * np.cos(u) * np.sin(v), c[2] + r * np.sin(u) * np.sin(v),
                        c[1] + r * np.cos(v), color="gray", alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (quota)")
    ax.set_title("3D")
    fig.tight_layout()
    fig.savefig("traiettorie.png", dpi=150)


def animate(ids, times, traj, outfile="animazione.gif"):
    cols = plt.cm.tab10(np.arange(len(ids)) % 10)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1)
    pad = 20
    ax.set_xlim(traj[..., 0].min() - pad, traj[..., 0].max() + pad)
    ax.set_ylim(traj[..., 2].min() - pad, traj[..., 2].max() + pad)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    draw_obstacles_2d(ax, 0, 2)

    d_safe = config.MIN_DISTANCE * config.SAFETY_FACTOR
    pts = [ax.plot([], [], "o", color=cols[k], ms=7)[0] for k in range(len(ids))]
    tails = [ax.plot([], [], "-", color=cols[k], lw=1, alpha=0.6)[0] for k in range(len(ids))]
    halos = [plt.Circle((0, 0), d_safe / 2, fill=False, color=cols[k], alpha=0.4) for k in range(len(ids))]
    for h in halos:
        ax.add_patch(h)                       # cerchio di raggio d_safe/2: se due si toccano, d = d_safe
    title = ax.set_title("")

    frames = range(0, traj.shape[1], FRAME_STRIDE)

    def update(f):
        for k in range(len(ids)):
            x, z = traj[k, f, 0], traj[k, f, 2]
            pts[k].set_data([x], [z])
            halos[k].center = (x, z)
            s = max(0, f - TAIL)
            tails[k].set_data(traj[k, s:f + 1, 0], traj[k, s:f + 1, 2])
        title.set_text(f"t = {times[f]:.1f} s")
        return pts + tails + halos + [title]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(outfile, writer=PillowWriter(fps=15))
    plt.close(fig)


def plot_metrics(csv="metrics.csv"):
    import pandas as pd
    m = pd.read_csv(csv)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(m.time, m.d_min, label="distanza minima")
    axs[0].axhline(config.MIN_DISTANCE, color="r", ls="--", label="MIN_DISTANCE")
    axs[0].axhline(config.MIN_DISTANCE * config.SAFETY_FACTOR, color="orange", ls="--", label="barriera CBF")
    axs[0].set_ylabel("[m]"); axs[0].legend(); axs[0].set_yscale("log")
    axs[1].plot(m.time, m.max_slack); axs[1].set_ylabel("slack massimo")
    axs[2].plot(m.time, m.mean_dist_target); axs[2].set_ylabel("dist. media dal bersaglio [m]")
    axs[2].set_xlabel("t [s]")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("metriche.png", dpi=150)


if __name__ == "__main__":
    ids, times, traj = load(FILE)
    print(f"{len(ids)} droni, {len(times)} istanti")
    plot_static(ids, traj)
    plot_metrics()
    animate(ids, times, traj)
    plt.show()