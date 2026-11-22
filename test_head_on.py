import numpy as np
import config
from agent import Agent
from message import Message

y = config.Y_MIN
agents = {
    "A": Agent("A", [0.0, y, 0.0]),
    "B": Agent("B", [100.0, y, 0.5]),   # 0.5 m di scarto laterale: evita la simmetria perfetta
}
agents["A"].update_goal([100.0, y, 0.0])
agents["B"].update_goal([0.0, y, 0.5])

dmin, emax = np.inf, 0.0
fb = 0
first_act = None
for k in range(1, int(60.0 / config.DT) + 1):
    t = k * config.DT
    for a in agents.values():
        Message(a.agent_id, a.position, a.velocity, a.acceleration, t).send(agents)
    for a in agents.values():
        a.step(t)
    dmin = min(dmin, np.linalg.norm(agents["A"].position - agents["B"].position))
    emax = max(emax, max(a.last_filter_effort for a in agents.values()))
    fb += sum(int(np.isinf(a.last_slack)) for a in agents.values())

    if first_act is None and max(a.last_filter_effort for a in agents.values()) > 0.01:
        d = np.linalg.norm(agents["A"].position - agents["B"].position)
        first_act = (t, d)

print("primo intervento (t, distanza):", first_act)

print(f"fallback = {fb}")

print(f"d_min = {dmin:.2f} m (richiesta {config.MIN_DISTANCE * config.SAFETY_FACTOR:.2f} m)")
print(f"max intervento filtro = {emax:.3f} m/s^2")
print(f"Velocità di A: {agents['A'].velocity} --- Accelerazione di A: {agents['A'].acceleration}")

for k, a in agents.items():
    print(k, "pos finale:", np.round(a.position, 2), "| dist dal goal:",
          round(float(np.linalg.norm(a.position - a.current_goal)), 2))