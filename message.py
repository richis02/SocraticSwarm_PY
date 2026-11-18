import agent

class Message:
    def __init__(self, agent, position, velocity, acceleration, timestamp):
        self.agent = agent
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.timestamp = timestamp

    def send(self, all_agents):
        for a in all_agents.values():
            if a.agent_id != self.agent:
                a.update_knowledge(
                    self.agent, 
                    self.position, 
                    self.velocity, 
                    self.acceleration, 
                    self.timestamp)