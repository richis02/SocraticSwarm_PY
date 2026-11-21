import numpy as np
from qpsolvers import solve_qp
import config


def _hocbf_row(p, v, p_o, v_o, a_o, d_safe):
    """Vincolo  A @ a_i >= b  per h = |p - p_o|^2 - d_safe^2 (grado relativo 2)."""
    a1, a2 = config.CBF_ALPHA1, config.CBF_ALPHA2
    dp, dv = p - p_o, v - v_o
    h = dp @ dp - d_safe ** 2
    h_dot = 2.0 * dp @ dv
    A = 2.0 * dp
    b = -(2.0 * dv @ dv - 2.0 * dp @ a_o + (a1 + a2) * h_dot + a1 * a2 * h)
    # normalizzo la riga: migliora il condizionamento e rende lo slack
    # confrontabile con un'accelerazione
    n = np.linalg.norm(A)
    if n < 1e-9:
        return None
    return A / n, b / n


def cbf_filter(p, v, a_nom, neighbors, obstacles):
    """
    neighbors: lista di (p_j, v_j, a_j) stimati al tempo corrente
    obstacles: lista di (centro, raggio)
    Ritorna (a_sicura, slack_max).
    """
    d_safe = config.MIN_DISTANCE * config.SAFETY_FACTOR
    zero = np.zeros(3)
    rows = []

    for p_j, v_j, a_j in neighbors:
        r = _hocbf_row(p, v, p_j, v_j, a_j, d_safe)
        if r is not None:
            rows.append(r)
    for c, rad in obstacles:
        r = _hocbf_row(p, v, c, zero, zero, rad + config.OBSTACLE_MARGIN)
        if r is not None:
            rows.append(r)

    # box inscritto nella sfera |a| <= MAX_ACCEL: così il clipping a valle non fa nulla
    a_lim = config.MAX_ACCEL / np.sqrt(3.0)
    a_nom = np.clip(a_nom, -a_lim, a_lim)
    if not rows:
        return a_nom, 0.0

    A = np.array([r[0] for r in rows])
    b = np.array([r[1] for r in rows])
    m = len(rows)

    # variabili: [a (3), slack (m)]
    P = np.diag([1.0] * 3 + [config.CBF_SLACK_WEIGHT] * m)
    q = np.concatenate([-a_nom, np.zeros(m)])
    G = np.hstack([-A, -np.eye(m)])          # -A a - s <= -b
    h = -b
    lb = np.concatenate([np.full(3, -a_lim), np.zeros(m)])
    ub = np.concatenate([np.full(3, a_lim), np.full(m, np.inf)])

    sol = solve_qp(P, q, G, h, lb=lb, ub=ub, solver="osqp")
    if sol is None:                           # fallback: frena
        return np.clip(-2.0 * v, -a_lim, a_lim), np.inf
    return sol[:3], float(sol[3:].max())