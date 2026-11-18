import numpy as np
from scipy.optimize import differential_evolution
import config

C_PENALTY = 0.5

def cohesion_cost(p, neighbors, r_c=config.COMMUNICATION_RANGE, alpha=0.8, z=config.N_NEIGHBORS):
    S = p.shape[0]
    if len(neighbors) == 0:
        return np.zeros(S)
    d = np.linalg.norm(p[:, None, :] - neighbors[None, :, :], axis=2)
    d = np.sort(d, axis=1)[:, :z]
    f = np.where(d < 0.75 * r_c, np.abs(2.0 * d / r_c - 1.0), C_PENALTY)
    fade = alpha ** np.arange(1, d.shape[1] + 1)
    return np.clip((fade * f).sum(axis=1) / d.shape[1], 0.0, 1.0)


def safety_cost(p, y_min=config.Y_MIN, y_max=config.Y_MAX):
    y = p[:, 1]
    c = np.where(y < y_min, (y / y_min - 1.0) ** 2, ((y - y_min) / y_max) ** 2)
    return np.clip(c, 0.0, 1.0)


def goal_cost(p, p_goal, c_dist=150.0):
    d = np.linalg.norm(p - p_goal, axis=1)
    return np.clip((2.0 / np.pi) * np.arctan(d / c_dist), 0.0, 1.0)


def predict(p0, v0, a, t):
    pos = p0 + v0 * t + 0.5 * a * t ** 2
    vel = v0 + a * t
    return pos, vel


def to_accel(u):
    n = np.linalg.norm(u, axis=1, keepdims=True)
    return u * (config.MAX_ACCEL / np.maximum(n, config.MAX_ACCEL))


def objective(u, p0, v0, neighbors_traj, p_goal, weights, T):
    w_eta, w_y, w_g = weights
    a = to_accel(u.T * config.MAX_ACCEL)               
    pT, vT = predict(p0, v0, a, T)

    N_T = neighbors_traj[-1][1]
    cost = w_eta * cohesion_cost(pT, N_T) + w_y * safety_cost(pT)

    if p_goal is not None and w_g > 0:
        # punto in cui il drone si fermerebbe se frenasse al massimo a fine orizzonte
        speed = np.linalg.norm(vT, axis=1, keepdims=True)
        p_stop = pT + config.BRAKE_GAIN * vT * speed / (2.0 * config.MAX_ACCEL)
        cost = cost + w_g * goal_cost(p_stop, p_goal)

    cost = cost + 10.0 * np.maximum(0.0, np.linalg.norm(vT, axis=1) - config.MAX_VELOCITY)

    # vincolo anti-collisione
    d_safe = config.MIN_DISTANCE * config.SAFETY_FACTOR
    viol = np.zeros(a.shape[0])
    for t_k, N_k in neighbors_traj:
        if len(N_k) == 0:
            continue
        p_k, _ = predict(p0, v0, a, t_k)
        dmin = np.linalg.norm(p_k[:, None, :] - N_k[None, :, :], axis=2).min(axis=1)
        viol += np.maximum(0.0, d_safe - dmin)
    cost = cost + np.where(viol > 0, 1e3 + 1e3 * viol, 0.0)   # penalita' graduale
    return cost


def optimize_acceleration(p0, v0, neighbors_traj, p_goal, weights, T=config.HORIZON):
    res = differential_evolution(
        func=objective,
        bounds=[(-1.0, 1.0)] * 3,
        args=(p0, v0, neighbors_traj, p_goal, weights, T),
        strategy='rand1bin',
        popsize=config.DE_POPSIZE,
        maxiter=config.DE_MAXITER,
        tol=0.0, atol=0.0,           
        polish=False,
        vectorized=True, updating='deferred',
        x0=np.zeros(3),              
        rng=config.SEED,
    )
    return to_accel(res.x[None, :] * config.MAX_ACCEL)[0]