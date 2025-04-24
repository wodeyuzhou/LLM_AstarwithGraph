import json
import networkx as nx

def graph_to_prompt(graph_json, start, goal, n_points):
    G = nx.node_link_graph(graph_json)
    prompt_nodes = ""
    prompt_nodes += " Available Nodes:\n"
    for node in G.nodes:
        prompt_nodes += f"- Node ID : {node} \n "

    prompt_nodes += "\n Available Paths:\n"
    for u, v, attr in G.edges(data=True):
        cost = attr.get('weight', 'Unknown')
        prompt_nodes += f"- From {u} to {v} → Cost : {cost}\n"

    prompt_nodes += f"\nStart: {start}\n"
    prompt_nodes += f"Destination: {goal}\n\n"
    
    prompt = f"""
### System:
You are an AI specialized in route planning and recommendations.

1. When a user provides graph data (edge list, adjacency list, etc.), parse it to understand the nodes and edges.  
2. Compute the most efficient route (by cost) from the user’s specified start to destination.  
3. Recommend exactly {n_points} waypoint(s) that will enhance the optimal route.

**Hard Constraints:**  
- Exclude the start node and the destination node from waypoint(s) candidates.  
- **Output only:** a Python-style list of unquoted integers, e.g. [6803250], [6806130, ...].  
- The list must contain exactly {n_points} integers, in order.  
- Do **not** output the path itself or pretend the path as waypoint(s).

### Examples (do not edit)

----------Examples 1 start------------
**User**  
Available Nodes:  
 - Node ID: 10
 - Node ID: 20
 - Node ID: 30
 - Node ID: 40
 - Node ID: 50
 - Node ID: 60
 - Node ID: 70

Available Paths:
 - From 40 to 20 → Cost: 1
 - From 20 to 30 → Cost: 3
 - From 40 to 30 → Cost: 2
 - From 10 to 30 → Cost: 4
 - From 30 to 10 → Cost: 1
 - From 50 to 10 → Cost: 2
 - From 50 to 20 → Cost: 5
 - From 50 to 40 → Cost: 3
 - From 60 to 50 → Cost: 4
 - From 60 to 30 → Cost: 2
 - From 60 to 20 → Cost: 6
 - From 60 to 70 → Cost: 1
 - From 70 to 10 → Cost: 3
 - From 70 to 30 → Cost: 4
 - From 70 to 40 → Cost: 2

Start: 60  
Destination: 20  

Please compute the optimal path, and recommend number of 2 waypoint.

**Assistant**  
[70, 40]
----------Examples 1 end------------

----------Examples 2 start------------
**User**  
Available Nodes:  
 - Node ID: 10  
 - Node ID: 20  
 - Node ID: 30  
 - Node ID: 40  

Available Paths:  
 - From 40 to 20 → Cost: 1  
 - From 20 to 30 → Cost: 3  
 - From 40 to 30 → Cost: 2  
 - From 10 to 30 → Cost: 4  
 - From 30 to 10 → Cost: 1  

Start: 40  
Destination: 10  

Please compute the optimal path, and recommend number of 1 waypoint.

**Assistant**  
[30]
----------Examples 2 end------------

→ Do NOT generate or complete any more Examples sections.

### User’s Task:
{prompt_nodes}

### Assistant:
"""

    return prompt