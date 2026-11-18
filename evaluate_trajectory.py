import numpy as np
import config
from agent import Agent
from config_pos import offset
from message import Message
from copy import deepcopy

def save_recap(agents, num_agents, time, filename):
    with open(filename, "a", encoding="utf-8") as file:
        print(f"---TIME: {time:.2f}---", file=file)
        for i in range(num_agents):
            pos = agents[f"Agent_{i}"].position
            id = agents[f"Agent_{i}"].agent_id
            print(f"Agent --- id={id} --- pos={pos}", file=file)
        print(f"---END TIME: {time}---", file=file)


def run_swarm_asi_evaluation():
    num_agents = 10
    delta = 1.0
    rng = np.random.default_rng(seed=config.SEED)
    time = 0.0

    goal_pos = np.array([300.0, config.Y_MAX, 300.0])

    agents = {}
    base_pos = np.array([0.0, config.Y_MIN, 0.0])
    for i in range(num_agents):
        agents[f"Agent_{i}"] = Agent(f"Agent_{i}", base_pos + offset[i])
        agents[f"Agent_{i}"].update_goal(goal_pos)
    print("---START---")
    save_recap(agents=agents, num_agents=num_agents, time=time, filename="agents.txt")
    agents_compromised = deepcopy(agents)
    save_recap(agents=agents_compromised, num_agents=num_agents, time=time, filename="agents_c.txt")

    step_count = 0
    total_steps = int(round(40.0 / config.DT))
    while step_count < total_steps:
        step_count += 1
        time = round(step_count * config.DT, 10)
        print(f'--- TIME: {time:.2f} ---')
        for i in range(num_agents):
            m = Message(
                agents[f"Agent_{i}"].agent_id, 
                agents[f"Agent_{i}"].position,
                agents[f"Agent_{i}"].velocity,
                agents[f"Agent_{i}"].acceleration,
                time)
            m.send(all_agents=agents)

        for i in range(num_agents):
            agents[f"Agent_{i}"].step(time)

            agent = agents[f"Agent_{i}"]
            for neighbor_id, neighbor in agents.items():
                if neighbor_id == agent.agent_id:
                    continue
                distance = np.linalg.norm(agent.position - neighbor.position)
                if distance < config.MIN_DISTANCE:
                    if distance < 2.0:
                        print(f'!!!ATTENZIONE SCHIANTO TRA AGENTE: {agent.agent_id} E AGENTE: {neighbor_id} al tempo {time:.2f}')
                    else:
                        print(f"VIOLATO PARAMETRO D_MIN tra AGENTE: {agent.agent_id} e AGENTE: {neighbor_id} al tempo {time:.2f}")

        # print(f"---TIME: {time} --- DELTA: {delta} --- POS: {agents['Agent_0'].position} ---")

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

        if (step_count >= 10): 
            with open('delta.txt', 'a', encoding="utf-8") as file:
                print(f"STEP: {step_count}", file=file)
                for id, data in agents.items():
                    delta_c = np.linalg.norm(data.position - agents_compromised[id].position)
                    print(f"AGENTE: {id}. DELTA={delta_c}", file=file)


        save_recap(agents=agents, num_agents=num_agents, time=time, filename='agents.txt')
        save_recap(agents=agents_compromised, num_agents=num_agents, time=time, filename='agents_c.txt')



if __name__ == "__main__":
    run_swarm_asi_evaluation()
