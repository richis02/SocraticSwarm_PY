import numpy as np
import config
from cbf import cbf_filter

class Agent:
    def __init__(self, agent_id, initial_position, formation_offset=(0,0,0)):
        self.agent_id = agent_id
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)

        self.formation_offset = np.array(formation_offset, dtype=float)
        
        self.current_goal = None
        
        # Knowledge Base: memorizza le informazioni note sugli altri agenti {id: {'position': np.array, 'timestamp': float}}
        self.knowledge_base = {}


    def update_goal(self, goal_position):
        self.current_goal = None if goal_position is None else np.array(goal_position, dtype=float)

    def update_knowledge(self, sender_id, position, velocity, acceleration, timestamp):
        if sender_id != self.agent_id:
            self.knowledge_base[sender_id] = {
                'position': np.array(position, dtype=float),
                'velocity': np.array(velocity, dtype=float),
                'acceleration': np.array(acceleration, dtype=float),
                'timestamp': timestamp
            }

    def get_neighbors_now(self, current_time):
        """Stato stimato dei vicini ADESSO (compensa il ritardo del messaggio)."""
        out = []
        for d in self.knowledge_base.values():
            tau = max(0.0, current_time - d['timestamp'])
            a = d['acceleration']
            v = d['velocity'] + a * tau
            p = d['position'] + d['velocity'] * tau + 0.5 * a * tau ** 2
            dist = np.linalg.norm(p - self.position)
            if dist < config.CBF_NEIGHBOR_RANGE:
                out.append((dist, p, v, a))
        out.sort(key=lambda x: x[0])
        return [(p, v, a) for _, p, v, a in out[:config.CBF_MAX_NEIGHBORS]]

    def get_nearby_obstacles(self):
        return [(c, r) for c, r in config.OBSTACLES
                if np.linalg.norm(self.position - c) < r + config.CBF_NEIGHBOR_RANGE]

    def nominal_acceleration(self):
        if self.current_goal is None:
            return -config.K_VEL * self.velocity          # frena
        e = (self.current_goal + self.formation_offset) - self.position
        v_des = config.K_POS * e
        n = np.linalg.norm(v_des)
        if n > config.V_CRUISE:
            v_des *= config.V_CRUISE / n
        return config.K_VEL * (v_des - self.velocity)

    def decide_next_action(self, current_time, weights=None):
        a, _ = cbf_filter(
            p=self.position, v=self.velocity,
            a_nom=self.nominal_acceleration(),
            neighbors=self.get_neighbors_now(current_time),
            obstacles=self.get_nearby_obstacles(),
        )
        return a


    def step(self, time, dt=config.DT):
        a = self.decide_next_action(current_time=time)
        
        a_mag = np.linalg.norm(a)
        if a_mag > config.MAX_ACCEL:
            a = a * (config.MAX_ACCEL / a_mag)

        self.acceleration = a

        new_vel = self.velocity + a * dt
        v_mag = np.linalg.norm(new_vel)
        if v_mag > config.MAX_VELOCITY:
            new_vel = new_vel * (config.MAX_VELOCITY / v_mag)

        # Aggiorna la posizione finale
        self.position = self.position + self.velocity * dt + 0.5 * a * dt ** 2
        self.velocity = new_vel