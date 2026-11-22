import numpy as np
import config
from agent import Agent
from config_pos import offset
from message import Message
from copy import deepcopy
import pandas as pd

# --- Parametri della valutazione ---
SIM_DURATION = 60.0
OFFSET_SCALE = 2.0    # prova 2.0 poi 1.5 e 1.0
COLLISION_DISTANCE = 2.0        # sotto questa distanza si considera "schianto"
SLACK_TOL = 1e-3                # slack sopra cui il filtro ha dovuto violare un vincolo
SCENARIO = "crossing"          # "formation" | "crossing"
USE_FORMATION_OFFSET = True   # per il crossing conviene False

OUTPUT_FILES = ["agents.txt", "agents_c.txt", "delta.txt", "metrics.csv", "summary.txt"]
METRICS_HEADER = ("time,d_min,pair_min,n_viol,max_slack,n_fallback,"
                  "mean_effort,max_effort,formation_err,mean_dist_target,max_delta")


def reset_outputs():
    """Svuota tutti i file di output, così ogni esecuzione parte da zero."""
    for fname in OUTPUT_FILES:
        open(fname, "w", encoding="utf-8").close()
    with open("metrics.csv", "w", encoding="utf-8") as f:
        print(METRICS_HEADER, file=f)


def save_recap(agents, num_agents, time, filename):
    with open(filename, "a", encoding="utf-8") as file:
        print(f"---TIME: {time:.2f}---", file=file)
        for i in range(num_agents):
            pos = agents[f"Agent_{i}"].position
            id = agents[f"Agent_{i}"].agent_id
            print(f"Agent --- id={id} --- pos={pos}", file=file)
        print(f"---END TIME: {time}---", file=file)


def positions(agents, ids):
    return np.array([agents[i].position for i in ids])


def run_swarm_asi_evaluation():
    reset_outputs()

    num_agents = 10
    delta = 1.0
    rng = np.random.default_rng(seed=config.SEED)
    time = 0.0

    goal_pos = np.array([300.0, config.Y_MIN, 300.0])

    agents, goals = {}, []
    base_pos = np.array([0.0, config.Y_MIN, 0.0])
    offsets = OFFSET_SCALE * np.array([offset[i] for i in range(num_agents)], dtype=float)
    for i in range(num_agents):
        if SCENARIO == "crossing" and i >= num_agents // 2:
            start, goal_i = goal_pos + offset[i], base_pos     # gruppo B: va nel verso opposto
        else:
            start, goal_i = base_pos + offset[i], goal_pos
        form = offsets[i] if USE_FORMATION_OFFSET else np.zeros(3)
        agents[f"Agent_{i}"] = Agent(f"Agent_{i}", start, formation_offset=form)
        agents[f"Agent_{i}"].update_goal(goal_i)
        goals.append(goal_i)
    ids = list(agents.keys())

    target_rel = offsets - offsets.mean(axis=0)
    targets = np.array(goals) + (offsets if USE_FORMATION_OFFSET else 0.0)
    iu = np.triu_indices(num_agents, k=1)

    print("---START---")
    save_recap(agents=agents, num_agents=num_agents, time=time, filename="agents.txt")
    agents_compromised = deepcopy(agents)
    save_recap(agents=agents_compromised, num_agents=num_agents, time=time, filename="agents_c.txt")

    # --- accumulatori per il riepilogo finale ---
    g_dmin, g_dmin_time, g_dmin_pair = np.inf, 0.0, ""
    steps_viol = 0          # passi con almeno una violazione di MIN_DISTANCE
    pair_collisions = 0     # coppie-passo sotto COLLISION_DISTANCE
    steps_slack = 0         # passi in cui almeno un drone ha usato lo slack
    tot_fallback = 0        # totale attivazioni del fallback "frena"
    g_max_slack = 0.0
    g_max_effort = 0.0

    step_count = 0
    total_steps = int(round(SIM_DURATION / config.DT))
    while step_count < total_steps:
        step_count += 1
        time = round(step_count * config.DT, 10)
        print(f'--- TIME: {time:.2f} ---')

        # 1) scambio messaggi
        for i in range(num_agents):
            m = Message(
                agents[f"Agent_{i}"].agent_id,
                agents[f"Agent_{i}"].position,
                agents[f"Agent_{i}"].velocity,
                agents[f"Agent_{i}"].acceleration,
                time)
            m.send(all_agents=agents)

        # 2) passo di tutti i droni
        for i in range(num_agents):
            agents[f"Agent_{i}"].step(time)

        # 3) metriche, calcolate DOPO che tutti si sono mossi (stesso istante per tutti)
        P = positions(agents, ids)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)[iu]
        k = int(np.argmin(D))
        d_min = float(D[k])
        pair_min = f"{ids[iu[0][k]]}-{ids[iu[1][k]]}"
        
        if step_count % 100 == 0 and time >= 30:
            d_each = np.linalg.norm(P - targets, axis=1)
            v_each = np.linalg.norm([agents[i].velocity for i in ids], axis=1)
            print(f"t={time:5.1f} dist={np.round(d_each, 1)} vel={np.round(v_each, 1)}")

        viol = np.where(D < config.MIN_DISTANCE)[0]
        for kk in viol:
            a_id, b_id = ids[iu[0][kk]], ids[iu[1][kk]]
            if D[kk] < COLLISION_DISTANCE:
                pair_collisions += 1
                print(f'!!!ATTENZIONE SCHIANTO TRA AGENTE: {a_id} E AGENTE: {b_id} al tempo {time:.2f}')
            else:
                print(f"VIOLATO PARAMETRO D_MIN tra AGENTE: {a_id} e AGENTE: {b_id} al tempo {time:.2f}")

        slacks = np.array([agents[i].last_slack for i in ids])
        efforts = np.array([agents[i].last_filter_effort for i in ids])
        n_fallback = int(np.isinf(slacks).sum())
        finite = slacks[np.isfinite(slacks)]
        max_slack = float(finite.max()) if finite.size else 0.0

        rel = P - P.mean(axis=0)
        formation_err = float(np.linalg.norm(rel - target_rel, axis=1).mean())
        mean_dist_target = float(np.linalg.norm(P - targets, axis=1).mean())

        # aggiornamento accumulatori
        if d_min < g_dmin:
            g_dmin, g_dmin_time, g_dmin_pair = d_min, time, pair_min
        steps_viol += int(len(viol) > 0)
        steps_slack += int(max_slack > SLACK_TOL or n_fallback > 0)
        tot_fallback += n_fallback
        g_max_slack = max(g_max_slack, max_slack)
        g_max_effort = max(g_max_effort, float(efforts.max()))

        # 4) swarm "compromesso" (test di sensibilità a una perturbazione)
        if step_count == 10:
            gaussian_vector = rng.normal(0, 1, 3)
            norma = np.linalg.norm(gaussian_vector)
            shift_vector = (gaussian_vector / norma) * delta
            agents_compromised["Agent_0"].position += shift_vector

        for i in range(num_agents):
            m = Message(
                agents_compromised[f"Agent_{i}"].agent_id,
                agents_compromised[f"Agent_{i}"].position,
                agents_compromised[f"Agent_{i}"].velocity,
                agents_compromised[f"Agent_{i}"].acceleration,
                time)
            m.send(all_agents=agents_compromised)

        for i in range(num_agents):
            agents_compromised[f"Agent_{i}"].step(time)

        max_delta = float("nan")
        if step_count >= 10:
            deltas = []
            with open('delta.txt', 'a', encoding="utf-8") as file:
                print(f"STEP: {step_count}", file=file)
                for id, data in agents.items():
                    delta_c = np.linalg.norm(data.position - agents_compromised[id].position)
                    deltas.append(delta_c)
                    print(f"AGENTE: {id}. DELTA={delta_c}", file=file)
            max_delta = float(max(deltas))

        save_recap(agents=agents, num_agents=num_agents, time=time, filename='agents.txt')
        save_recap(agents=agents_compromised, num_agents=num_agents, time=time, filename='agents_c.txt')

        dists = np.linalg.norm(positions(agents, ids) - targets, axis=1)
        print("Distanza di ogni drone dal proprio slot:", np.round(dists, 2))

        with open("metrics.csv", "a", encoding="utf-8") as f:
            print(f"{time:.2f},{d_min:.4f},{pair_min},{len(viol)},{max_slack:.6f},{n_fallback},"
                  f"{efforts.mean():.4f},{efforts.max():.4f},{formation_err:.4f},"
                  f"{mean_dist_target:.4f},{max_delta:.6f}", file=f)

    # --- riepilogo finale ---
    summary = [
        "========== RIEPILOGO ==========",
        f"Passi simulati:               {total_steps}  (T = {SIM_DURATION:.1f} s, dt = {config.DT})",
        f"Distanza minima tra droni:    {g_dmin:.3f} m  ({g_dmin_pair}, t = {g_dmin_time:.2f} s)",
        f"MIN_DISTANCE richiesta:       {config.MIN_DISTANCE:.3f} m",
        f"Passi con violazione D_MIN:   {steps_viol} / {total_steps}",
        f"Coppie-passo di 'schianto':   {pair_collisions}",
        f"Passi con slack > {SLACK_TOL}:    {steps_slack} / {total_steps}  (max slack = {g_max_slack:.4f})",
        f"Attivazioni fallback 'frena': {tot_fallback}",
        f"Max intervento del filtro:    {g_max_effort:.3f} m/s^2  (MAX_ACCEL = {config.MAX_ACCEL})",
        f"Distanza media dal bersaglio: {mean_dist_target:.2f} m (a fine simulazione)",
        f"Errore di formazione finale:  {formation_err:.2f} m",
    ]
    text = "\n".join(summary)
    print(text)
    with open("summary.txt", "w", encoding="utf-8") as f:
        print(text, file=f)


if __name__ == "__main__":
    run_swarm_asi_evaluation()
    m = pd.read_csv("metrics.csv")

    # 1) quando lo slack è attivo
    s = m[m.max_slack > 1e-3]
    print("slack attivo tra t =", s.time.min(), "e t =", s.time.max())
    print(s.time.round(0).value_counts().sort_index())      # passi con slack per secondo

    # 2) la distanza dal bersaglio sta ancora calando?
    print(m[["time", "mean_dist_target"]].iloc[::100])