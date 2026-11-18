import numpy as np
import config
from costs import optimize_acceleration

class Agent:
    def __init__(self, agent_id, initial_position):
        self.agent_id = agent_id
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)
        
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

    def get_neighbors_trajectories(self, current_time, T=config.HORIZON, n_sub=4):
        out = []
        for k in range(1, n_sub + 1):
            t_k = T * k / n_sub
            pts = []
            for data in self.knowledge_base.values():
                tau = max(0.0, current_time - data['timestamp']) + t_k
                pts.append(data['position'] + data['velocity'] * tau
                           + 0.5 * data['acceleration'] * tau ** 2)
            out.append((t_k, np.array(pts).reshape(-1, 3)))
        return out

    def decide_next_action(self, current_time, weights=None):
        if weights is None:
            weights = (config.INITIAL_WEIGHTS['w_eta'], 
                       config.INITIAL_WEIGHTS['w_y'], 
                       config.INITIAL_WEIGHTS['w_g'])
            
        n_traj = self.get_neighbors_trajectories(current_time=current_time)

        acc = optimize_acceleration(
            p0=self.position,
            v0=self.velocity,
            neighbors_traj=n_traj,
            p_goal=self.current_goal,
            weights=weights
        )
        
        return acc


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