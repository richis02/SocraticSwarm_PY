import numpy as np
import config
from agent import Agent
from config_pos import offset
from message import Message
from copy import deepcopy

def save_recap(agents, num_agents, time, filename):
    with open(filename, "a", encoding="utf-8") as file:
        print(f"---TIME: {time}---", file=file)
        for i in range(num_agents):
            pos = agents[f"Agent_{i}"].position
            id = agents[f"Agent_{i}"].agent_id
            print(f"Agent --- id={id} --- pos={pos}", file=file)
        print(f"---END TIME: {time}---", file=file)


def run_swarm_asi_evaluation():
    num_agents = 10
    delta_values = np.linspace(0.0, 3.0, 30)
    time = 0.0
    asi_results = {}

    goal_pos = np.array([200.0, config.Y_MAX, 200.0])

    agents = {}
    base_pos = np.array([0.0, 0.0, 0.0])
    for i in range(num_agents):
        agents[f"Agent_{i}"] = Agent(f"Agent_{i}", base_pos + offset[i])
        agents[f"Agent_{i}"].update_goal(goal_pos)
    print("---START---")
    #save_recap(agents=agents, num_agents=num_agents, time=time, filename="results.txt")

    step = 0
    steps_per_checkpoint = round(0.5 / config.DT)
    while time < 10.0:
        step += 1
        time = step * config.DT
        for i in range(num_agents):
            m = Message(
                agents[f"Agent_{i}"].agent_id, 
                agents[f"Agent_{i}"].position,
                agents[f"Agent_{i}"].velocity,
                agents[f"Agent_{i}"].acceleration,
                time)
            m.send(all_agents=agents)

            agents[f"Agent_{i}"].step(time)

            agent = agents[f"Agent_{i}"]
            for neighbor_id, neighbor in agents.items():
                if neighbor_id == agent.agent_id:
                    continue
                distance = np.linalg.norm(agent.position - neighbor.position)
                if distance < config.MIN_DISTANCE:
                    print("VIOLATO PARAMETRO D_MIN")

        if step % steps_per_checkpoint == 0:
            for delta in delta_values:
                print(f"---TIME: {time:.2f} --- DELTA: {delta} --- POS: {agents['Agent_0'].position} ---")
                agents_compromised = deepcopy(agents)
                gaussian_vector = np.random.normal(0, 1, 3)
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

                    agents_compromised[f"Agent_{i}"].step(time)

                # CALCOLO ASI su un vicino
                pos_perturbed = agents_compromised["Agent_1"].position
                id_neighbor = 'Agent_1'
                pos_nominal = agents[id_neighbor].position
                control_diff = np.linalg.norm(pos_perturbed - pos_nominal)
                delta_norm = np.linalg.norm(shift_vector)
                sensitivity_ratio = 0
                if delta_norm != 0:
                    sensitivity_ratio = control_diff / delta_norm 

                #save_recap(agents=agents_compromised, num_agents=num_agents, time=time, filename=f"results_compromised_{delta}.txt")

                key = (tuple(agents['Agent_0'].position), delta)
                asi_results[key] = np.mean(sensitivity_ratio) 

            #save_recap(agents=agents, num_agents=num_agents, time=time, filename="results.txt")

    with open("asi_results.txt", "a", encoding="utf-8") as file:
        for key, value in asi_results.items():
            print(f"{key}: {value}", file=file)


if __name__ == "__main__":
    run_swarm_asi_evaluation()
