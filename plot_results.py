import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def read_file(filename):
    # I tuoi dati di esempio (incollati qui per comodità)
    with open(filename, "r", encoding="utf-8") as file:
        dati_testo = file.read()

    # 1. Parsing dei dati con le RegEx
    pattern_time = r'--TIME:\s*([\d\.]+)---'
    pattern_agent = r'Agent\s+---\s+id=(\w+)\s+---\s+pos=\[\s*([-\d\.e\+\s]+)\]'

    agent_trajectories = {}
    current_time = None

    for line in dati_testo.strip().split('\n'):
        t_match = re.search(pattern_time, line)
        if t_match:
            current_time = float(t_match.group(1))
            continue
            
        a_match = re.search(pattern_agent, line)
        if a_match and current_time is not None:
            agent_id = a_match.group(1)
            coords = [float(c) for c in a_match.group(2).split()]
            
            if agent_id not in agent_trajectories:
                agent_trajectories[agent_id] = {'time': [], 'x': [], 'y': [], 'z': []}
                
            agent_trajectories[agent_id]['time'].append(current_time)
            agent_trajectories[agent_id]['x'].append(coords[0])
            agent_trajectories[agent_id]['y'].append(coords[1])
            agent_trajectories[agent_id]['z'].append(coords[2])
    return agent_trajectories

def plot_results():
    agent_trajectories = read_file("agents.txt")
    agent_trajectories_c = read_file("agents_c.txt")
    # 2. Creazione del grafico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for agent_id, data in agent_trajectories.items():
        ax.plot(data['x'], data['y'], data['z'], marker='o', label=agent_id)

    ax.set_xlabel('Asse X')
    ax.set_ylabel('Asse Y')
    ax.set_zlabel('Asse Z')
    ax.set_title('Traiettorie 3D degli Agenti nel Tempo')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()


    fig = go.Figure()

    for agent_id, data in agent_trajectories.items():
        fig.add_trace(go.Scatter3d(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            mode='lines+markers',
            name=agent_id,
            text=[f"Time: {t}" for t in data['time']] # Mostra il tempo al passaggio del mouse
        ))
    for agent_id, data in agent_trajectories_c.items():
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='lines+markers',
                name=f"{agent_id}_c",
                text=[f"Time: {t}" for t in data['time']] # Mostra il tempo al passaggio del mouse
            ))

    fig.update_layout(
        title='Traiettorie 3D Interattive',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )

    fig.show()

if __name__ == "__main__":
    plot_results()