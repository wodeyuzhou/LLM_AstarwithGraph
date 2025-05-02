import json
import textwrap
from networkx.readwrite import json_graph

def few_shot(G, start, goal, n_points):
    
    user_prompt = "Graph: \nAvailable Nodes:\n"
    for node in G.nodes:
        user_prompt += f"- Node ID: {node}\n "

    user_prompt += "\nAvailable Paths:\n"
    for u, v, attr in G.edges(data=True):
        cost = attr.get('weight', 'Unknown')
        user_prompt += f"- From {u} to {v} → Cost: {cost}\n"

    user_prompt += f"""Recommend {n_points} waypoint after compute the cheapest rount from start node {start} to goal node {goal}."""

    text = textwrap.dedent(f"""<|im_start|>system
    You are a path-finding robot that computes the cheapest route in a directed graph.  
    Your task is to return exactly **{n_points}** intermediate waypoint nodes (in order) that should appear between the start node `{start}` and goal node `{goal}` for the most cost-efficient route.
    RULES:
    • *\Exclude the start node ({start}) and the goal node ({goal}) from the output.
    • The output must contain **exactly {n_points} integer node IDs** in Python list format, and **only that**.
    • Avoid loops (e.g., [30, 10, 30]) or revisiting the same node twice.
    • Prefer the lowest cumulative cost route (cost-prioritized reasoning, like A*).
    • If a path becomes clearly worse than a better one, discard it early (prune it).
    FORMAT: [id1, id2, ...]  # unquoted integer node IDs, in route order only<|im_end|>
    <|im_start|>user
    GRAPH:
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
    Recommend 1 waypoint after compute the cheapest rount from start node 40 to goal node 10.<|im_end|>
    <|im_start|>assistant
    [30]<|im_end|>
    <|im_start|>user
    GRAPH:
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
    Recommend 2 waypoint after compute the cheapest rount from start node 60 to goal node 20.<|im_end|>
    <|im_start|>assistant
    [70, 40]<|im_end|>
    <|im_start|>user
    {user_prompt}<|im_end|>
    <|im_start|>assistant""")

    return text

def cot(G, start, goal, n_points):
    
    user_prompt = "Graph: \nAvailable Nodes:\n"
    for node in G.nodes:
        user_prompt += f"- Node ID: {node}\n "

    user_prompt += "\nAvailable Paths:\n"
    for u, v, attr in G.edges(data=True):
        cost = attr.get('weight', 'Unknown')
        user_prompt += f"- From {u} to {v} → Cost: {cost}\n"

    user_prompt += f"""Recommend {n_points} waypoint after compute the cheapest rount from start node {start} to goal node {goal}."""

    text = textwrap.dedent(f"""<|im_start|>system
    You are a path-finding robot that computes the cheapest route in a directed graph.  
    Your task is to return exactly **{n_points}** intermediate waypoint nodes (in order)  
    that should appear between the start node `{start}` and goal node `{goal}`  
    for the most cost-efficient route.
    RULES:
    • Focus only on the **internally chosen candidate paths** and **cost comparisons**.  
    • Do not restate the full input. Think **inside the graph**, not by rewriting it. 
    • *\Exclude the start node ({start}) and the goal node ({goal}) from the output.
    • Do not output or describe the full path.
    • The output must contain **exactly {n_points} integer node IDs** in Python list format, and **only that**.
    • Avoid loops (e.g., [30, 10, 30]) or revisiting the same node twice.
    • Prefer the lowest cumulative cost route (cost-prioritized reasoning, like A*).
    • If a path becomes clearly worse than a better one, discard it early (prune it).
    • Finish your reasoning and then answer with the final waypoint list.
    FORMAT: [id1, id2, ...]  # unquoted integer node IDs, in route order only<|im_end|>
    <|im_start|>user
    GRAPH:
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
    Recommend 1 waypoint after compute the cheapest rount from start node 40 to goal node 10.<|im_end|>
    <|im_start|>assistant
    <think>
    Okay, let's tackle this problem. The user wants me to find the cheapest route from node 40 to node 10 and then recommend one waypoint in between. The available nodes are 10, 20, 30, 40. 
    The paths given are: ... Okay.
    So the start is 40 and the goal is 10. Let me think about the possible paths.
    First, from 40, there are two direct paths: to 20 (cost 1) and to 30 (cost 2). Let me explore both options.
    Option 1: 40 → 20 → 30 → 10. Let's calculate the cost. 40 to 20 is 1, 20 to 30 is 3, then 30 to 10 is 1. Total cost is 1+3+1=5.
    Option 2: 40 → 30 → 10. That's 2 (from 40 to 30) plus 1 (from 30 to 10) = total 3. That's cheaper than option 1. So the direct path through 30 is better.
    Wait, are there any other paths? Let me check. From 40, can we go to 30 directly, then to 10. That's the second option. What about other routes? Like 40 → 20 → 30 → 10, which we already considered. Are there any other nodes? The available nodes are only 10, 20, 30, 40. So no other nodes to go through. So the cheapest path is 40 → 30 → 10 with total cost 3.
    So the waypoints between 40 and 10 would be 30. But the user wants exactly one waypoint. So the answer should be [30]. Let me confirm again. The start is 40, goal is 10. The path is 40 → 30 → 10. So the waypoint is 30. That's the only node in between. So the output should be [30]. 
    I need to make sure there's no other path with a lower cost. Let me check again. What if there's a path through 20 and then some other way? Like 40 → 20 → 30 → 10, which totals 1+3+1=5. That's more expensive than 3. So the best path is indeed 40 → 30 → 10. Therefore, the waypoint is 30. The answer is [30].
    </think>
    [30]<|im_end|>
    <|im_start|>user
    GRAPH:
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
    Recommend 2 waypoint after compute the cheapest rount from start node 60 to goal node 20.<|im_end|>
    <|im_start|>assistant
    <think>
    Okay, let's tackle this problem. The user wants the cheapest route from node 60 to node 20, and then to recommend two waypoints in between. The available nodes are 10, 20, 30, 40, 50, 60, 70. The paths are a bit more complex now with more nodes.
    First, I need to figure out the shortest path from 60 to 20. Let me list all possible paths and calculate their costs. But since there are multiple paths, maybe using Dijkstra's algorithm would be better. Let me think step by step.
    Starting at node 60. From 60, the direct paths are to 50 (cost 4), 30 (cost 2), 20 (cost 6), and 70 (cost 1). So the cheapest immediate step from 60 is to 70 with cost 1. Then from 70, the paths are to 10 (cost 3), 30 (cost 4), and 40 (cost 2). Let me explore those.
    Alternatively, from 60 to 30 (cost 2), then from 30 to 10 (cost 1), but that's going towards 10, not 20. Wait, but maybe from 30, there's a way to reach 20? Let me check the paths. From 30, the paths are to 10 (cost 1), 20 (cost 3?), wait, the original paths given include From 20 to 30 (cost 3), but not from 30 to 20. Wait, the available paths are directed. So from 30 to 20 isn't listed. So from 30, you can only go to 10 (cost 1) or maybe others? Let me check again.
    Looking back at the available paths: ... Okay.
    So, from 30, there's only a path to 10. So if I go from 60 to 30 (cost 2), then 30 to 10 (cost 1), then from 10, maybe to 30 again? But that's a loop. Alternatively, from 10, maybe to 30 again, but that's not helpful. Hmm.
    Let me try the path starting from 60 to 70 (cost 1), then from 70 to 40 (cost 2). Now at 40, from there to 20 (cost 1). So total cost would be 1 (60→70) + 2 (70→40) + 1 (40→20) = 4. That's a total of 4.
    Another path: 60→70→40→20. That's 1+2+1=4. Let's see if there's a cheaper path. What about 60→30 (cost 2), then 30→10 (cost 1), then 10→30 (cost 4?), but that's going back. Not helpful. Or from 10, maybe to 30 again? Not helpful. Alternatively, 60→50 (cost 4), then 50→20 (cost 5). Total cost 4+5=9, which is worse than 4.
    Another option: 60→50→40 (cost 4+3=7) then 40→20 (cost 1). Total 7+1=8. Still worse than 4.
    What about 60→70→40→30 (cost 1+2+2=5), then 30→10 (cost 1), then from 10 to... but we need to reach 20. Not helpful. So that's not useful.
    Alternatively, 60→70→30 (cost 1+4=5), then 30→10 (cost 1), but again, not helpful for reaching 20.
    So the path 60→70→40→20 is the cheapest so far with total cost 4. Let me check if there's another path with lower cost.
    What about 60→70→40→30→10→... no, that's going away from 20. So not helpful.
    Wait, is there a path from 60 to 50 to 40 to 20? 60→50 (4), 50→40 (3), 40→20 (1). Total 4+3+1=8. Still higher than 4.
    Another possibility: 60→70→40→30→20? Wait, is there a path from 30 to 20? The available paths don't mention that. The path from 20 to 30 is there, but not the reverse. So from 30, you can't go to 20. So that path is not possible.
    So the best path seems to be 60→70→40→20 with cost 4. But wait, maybe there's another route. Let me check 60→30 (cost 2), then 30→10 (cost 1), then 10→30 (cost 4) again? No, that loops. Not helpful.
    Alternatively, from 60 to 70 (cost 1), then 70 to 30 (cost 4), then 30 to 10 (cost 1), then from 10 to 30 again? No. Not helpful.
    Another option: 60→70→40→30→... but again, stuck.
    So the cheapest path is indeed 60→70→40→20 with cost 4. So the waypoints between 60 and 20 are 70 and 40. Wait, but the user wants exactly two waypoints. So the path is 60 → 70 → 40 → 20. Therefore, the waypoints are 70 and 40. Let me check if there's another path with two waypoints that might have a lower cost. For example, maybe 60→50→40→20? That's 4 (60→50) + 3 (50→40) +1 (40→20) = 8, which is more expensive. So no.
    Another possible path: 60→70→30→... but from 30, can't reach 20. So that's not useful. So the only valid path is 60→70→40→20. Therefore, the two waypoints are 70 and 40. So the answer should be [70, 40]. Let me make sure there's no other path with two waypoints that's cheaper. For example, if there's a path that goes through 50 and another node, but that seems more expensive. So yes, [70, 40] is the correct answer.
    </think>
    [70, 40]<|im_end|>
    <|im_start|>user
    {user_prompt}<|im_end|>
    <|im_start|>assistant
    <think>
    """)

    return text