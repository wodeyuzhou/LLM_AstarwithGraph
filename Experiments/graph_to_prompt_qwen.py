import json
import networkx as nx

import networkx as nx
import json, textwrap


n_points = 7
start, goal = (10, 20)

system_prompt = textwrap.dedent(f"""
You are a robot that determines the shortest route connecting the start node and the goal node in a given graph.\
You are must returns **only** a Python-style list of exactly {n_points} integer waypoint IDs.

RULES:
• Exclude start node({start}) and goal node({goal}).
• Do not reveal or explain the path.
• The answer is INVALID if the list length ≠ {n_points}.
FORMAT: [id1, id2, ...]  # no quotes, no extra text
""")
print(system_prompt)

def graph_to_prompt_opt(G, start, goal, n_points):
    # ---------- 1. SYSTEM  ----------
    system_prompt = textwrap.dedent(f"""
    You are an assistant that returns **only** a Python-style list of exactly {n_points} integer waypoint IDs.
    RULES:
      • Exclude start ({start}) and goal ({goal}).
      • Do not reveal or explain the path.
      • The answer is INVALID if the list length ≠ {n_points}.
    FORMAT: [id1, id2, ...]  # no quotes, no extra text
    """)

    # ---------- 2. FEW-SHOT EXAMPLES ----------
    examples = [
        # 1-waypoint
        ("""Available Nodes: 1 2 3
            Available Paths: 1-2(1) 2-3(1)
            Compute the optimal path 1→3 with 1 waypoint.""",
         "[2]"),
        # 2-waypoints
        ("""Available Nodes: 10 20 30 40
            Available Paths: 10-20(1) 20-30(1) 30-40(1)
            Compute the optimal path 10→40 with 2 waypoints.""",
         "[20, 30]"),
        # 3-waypoints
        ("""Available Nodes: 5 6 7 8 9
            Available Paths: 25-6(1) 6-7(1) 7-8(1) 8-9(1)
            Compute the optimal path 5→9 with 3 waypoints.""",
         "[6, 7, 8]")
    ]

    # ---------- 3. USER GRAPH ----------
    graph_txt = ["Available Nodes:"]
    graph_txt += [f" - {n}" for n in G.nodes]
    graph_txt += ["", "Available Paths:"]
    for u, v, d in G.edges(data=True):
        graph_txt.append(f" - {u}→{v} (cost {d.get('weight','?')})")
    graph_txt.append(
        f"\nCompute the optimal path from {start} to {goal} with {n_points} waypoint(s)."
    )
    graph_prompt = "\n".join(graph_txt)

    # ---------- 4. BUILD MESSAGE LIST ----------
    messages = [{"role": "system", "content": system_prompt}]
    for u, a in examples:
        messages.extend([
            {"role": "user", "content": textwrap.dedent(u)},
            {"role": "assistant", "content": a}
        ])
    messages.append({"role": "user", "content": graph_prompt})
    return messages


def graph_to_prompt(G, start, goal, n_points):

    # system prompt

    system_prompt = f"""
You are an Assistant specialized in optimal route planning and recommendations.

1. When a user provides graph data (edge list, adjacency list, etc.), parse it to understand the nodes and edges.  
2. Compute the most efficient route (by cost) from the user’s specified start to destination.  
3. Recommend exactly {n_points} waypoint(s) that will enhance the optimal route.

**Hard Constraints:**  
- **The start node ({start}) and the destination node ({goal}) must NEVER appear in the waypoint list.** 
- **Output only:** a Python-style list of unquoted integers, e.g. [6803250], [6806130, ...].  
- The list must contain exactly {n_points} integers in path planning order.
- Do **not** output the path itself or pretend the path as waypoint(s).
- Do **not** output the path itself under any circumstance.  
- Do **not** pretend the path as waypoints.  
- You must output **only** a **Python-style list** of node IDs.  
- The waypoints array must contain exactly {n_points} node IDs.
"""
    example1_user = """ 
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

Compute the optimal path from start node 60 to destination node 20 and recommend 1 waypoint.
"""

    example1_assistant = """
[70, 40]
"""
    example2_user = """
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

Compute the optimal path from start node 40 to destination node 10 and recommend 1 waypoint.
"""
    example2_assistant = """
[30]
"""
    user_prompt = " Available Nodes:\n"
    for node in G.nodes:
        user_prompt += f"- Node ID: {node}\n "

    user_prompt += "\n Available Paths:\n"
    for u, v, attr in G.edges(data=True):
        cost = attr.get('weight', 'Unknown')
        user_prompt += f"- From {u} to {v} → Cost: {cost}\n"

    user_prompt += f"""
Compute the optimal path from start node {start} to destination node {goal} and recommend {n_points} waypoint.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example1_user},
        {"role": "assistant", "content": example1_assistant},
        {"role": "user", "content": example2_user},
        {"role": "assistant", "content": example2_assistant},
        {"role": "user", "content": user_prompt},
    ]

    return messages


def graph_to_prompt_CoT(graph_json, start, goal, n_points):
    G = nx.node_link_graph(graph_json)
    prompt_nodes = ""
    prompt_nodes += " Available Nodes:\n"
    for node in G.nodes:
        prompt_nodes += f"- Node ID : {node} \n "

    prompt_nodes += "\n Available Paths:\n"
    for u, v, attr in G.edges(data=True):
        cost = attr.get('weight', 'Unknown')
        prompt_nodes += f"- From {u} to {v} → Cost : {cost}\n"
    
    prompt = f"""
### System:
You are an AI specialized in route planning and recommendations.

**Hidden-CoT Guidelines (do NOT reveal):**
1. When a user gives graph data, construct an internal graph object and use an optimal-path algorithm (e.g. Dijkstra) to find the minimum-cost route from **Start** to **Destination**.
2. From that optimal route choose exactly **{n_points}** intermediate node(s) (excluding start & destination) that improve travel efficiency or user experience.  
   • Prefer nodes that lie on, or cause only minimal detour from, the optimal path.  
3. Think **step-by-step internally**, but **never include your thoughts, the full path, or cost details in the final answer**.  

**Public Output Rules (must follow):**
- Output only a Python-style list of unquoted integers, e.g. `[6803250]`, `[6806130, 6806144]`.  
- The list must contain **exactly {n_points} ID(s)**, ordered as waypoints should be visited.  
- Do **NOT** output the complete path, costs, or any explanation.  
- Do **NOT** “pretend” the full path as waypoints.  
- Keep total output ≤ 200 tokens.

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

→ Do NOT add or modify any Examples.

### User’s Task:
{prompt_nodes}

### Assistant:

"""
    return prompt