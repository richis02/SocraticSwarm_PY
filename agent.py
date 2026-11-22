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
        self.swirl_gain = 0.0
        
        self.current_goal = None
        
        # Knowledge Base: memorizza le informazioni note sugli altri agenti {id: {'position': np.array, 'timestamp': float}}
        self.knowledge_base = {}

        #per log
        self.last_slack = 0.0
        self.last_filter_effort = 0.0


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

    def _update_swirl_gain(self, dt=config.DT):
        a_lim = config.MAX_ACCEL / np.sqrt(3.0)
        inst = np.clip(self.last_filter_effort / (0.5 * a_lim), 0.0, 1.0)
        self.swirl_gain = max(inst, self.swirl_gain * np.exp(-dt / config.SWIRL_TAU))

    def swirl_velocity(self, to_goal_dir, neighbors, obstacles):
        """Velocità laterale che rompe la simmetria, perpendicolare a chi mi blocca."""
        up = np.asarray(config.SWIRL_UP, dtype=float)
        d_safe = config.MIN_DISTANCE * config.SAFETY_FACTOR
        threats = [(p_j, d_safe) for p_j, _, _ in neighbors]
        threats += [(c, r + config.OBSTACLE_MARGIN) for c, r in obstacles]

        swirl = np.zeros(3)
        for c, clearance in threats:
            d_vec = c - self.position
            dist = np.linalg.norm(d_vec)
            if dist < 1e-6:
                continue
            d_hat = d_vec / dist
            if d_hat @ to_goal_dir < config.SWIRL_CONE:
                continue                                  # non è sulla mia strada
            t = np.cross(up, d_hat)
            n = np.linalg.norm(t)
            if n < 1e-3:                                  # congiungente parallela a "su"
                t = np.cross(np.array([1.0, 0.0, 0.0]), d_hat)
                n = np.linalg.norm(t)
            w = min(1.0, clearance / dist)                # più vicino = più forte
            swirl += w * t / n

        n = np.linalg.norm(swirl)
        if n > 1.0:
            swirl /= n
        return config.K_SWIRL * self.swirl_gain * swirl

    def _desired_velocity(self, e):
        """Velocità desiderata verso il bersaglio, con frenata compatibile con A_BRAKE."""
        d = np.linalg.norm(e)
        if d < 1e-9:
            return np.zeros(3)
        speed = min(config.V_CRUISE,
                    np.sqrt(2.0 * config.A_BRAKE * d),   # lontano: profilo di frenata
                    config.K_POS * d)                     # vicino: lineare, evita il chattering
        return speed * e / d

    def nominal_acceleration(self, neighbors=(), obstacles=()):
        if self.current_goal is None:
            return -config.K_VEL * self.velocity          # frena
        e = (self.current_goal + self.formation_offset) - self.position
        e_norm = np.linalg.norm(e)
        v_des = self._desired_velocity(e)
        if self.swirl_gain > 1e-3 and e_norm > 1e-6:
            v_des = v_des + self.swirl_velocity(e / e_norm, neighbors, obstacles)
            n = np.linalg.norm(v_des)
            if n > config.V_CRUISE:
                v_des *= config.V_CRUISE / n
        return config.K_VEL * (v_des - self.velocity)

    def decide_next_action(self, current_time, weights=None):
        neighbors = self.get_neighbors_now(current_time)
        obstacles = self.get_nearby_obstacles()
        self._update_swirl_gain()
        a_nom = self.nominal_acceleration(neighbors, obstacles)
        a, slack = cbf_filter(
            p=self.position, v=self.velocity, a_nom=a_nom,
            neighbors=neighbors, obstacles=obstacles,
        )
        a_lim = config.MAX_ACCEL / np.sqrt(3.0)
        self.last_slack = slack
        self.last_filter_effort = float(np.linalg.norm(a - np.clip(a_nom, -a_lim, a_lim)))
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