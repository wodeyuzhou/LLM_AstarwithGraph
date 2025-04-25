import os
import re
import csv
import json
import numpy as np
import heapq
from matplotlib import colors
import matplotlib.pyplot as plt
import torch
import networkx as nx
from networkx.readwrite import json_graph
from graph_to_prompt import graph_to_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

def get_nums_list(s):
    nums_str = re.findall(r'\d+', s)  
    nums = list(map(int, nums_str))
    return  nums

class Request_llm:
    def __init__(self, model_name):
        if model_name == 'llama':
            self.max_new_tokens = 200
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="auto",load_in_8bit=True)
            #self.model.save_model("models/meta-llama/Llama-3.2-3B-Instruct")
        

        self.model.config.pad_token_id = self.model.config.eos_token_id

    def get_waypoints(self, j, start, goal, n_points):
        prompt = graph_to_prompt(j, start, goal, n_points)
        inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    return_attention_mask=True
                )
        
        input_ids = inputs["input_ids"]
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens
            )
        generated_token = outputs[:, input_ids.shape[-1] :]
        result = self.tokenizer.batch_decode(generated_token, skip_special_tokens=True)[0]
        waypoints = get_nums_list(result)
        return waypoints

def llm_astar(G, start, goal, heuristic, llm, obstacles=None):
    nodes = list(G.nodes)
    llm = list(set(llm))
    T = []
    for ts in llm:
        if ts in nodes:
            T.append(ts)
    if obstacles is None:
        obstacles = set()
    else:
        obstacles = set(obstacles)
    print(llm)
    # T.reverse()
    if not T or T[0] != start:
        T.insert(0, start)
    if T[-1] != goal:
        T.append(goal)
    T = [t for t in T if t not in obstacles]
    t_idx = 1 if len(T) > 1 else 0
    t = T[t_idx]
    dist_to_t = nx.single_source_dijkstra_path_length(G, t, weight='weight')
    count = 0
    storage = 0
    checking_edges = set()
    came_from = {}
    visited = set()

    g_score = {n: float('inf') for n in G.nodes}
    g_score[start] = 0

    f_score = {n: float('inf') for n in G.nodes}
    f_score[start] = g_score[start] + heuristic[start] + dist_to_t.get(start, float('inf'))

    open_set = []
    heapq.heappush(open_set, (f_score[start], g_score[start], start))

    while open_set:
        storage += len(open_set) + len(visited) + len(came_from)
        f_cur, g_cur, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], g_cur, count, storage, list(checking_edges), T
        visited.add(current)
        if current == t and t_idx < len(T) - 1:
            t_idx += 1
            t = T[t_idx]
            dist_to_t = nx.single_source_dijkstra_path_length(G, t, weight='weight')

            new_open = []
            for _, g_old, node in open_set:
                new_f = g_score[node] + heuristic[node] + dist_to_t.get(node, float('inf'))
                new_open.append((new_f, g_old, node))
            open_set = new_open
            heapq.heapify(open_set)


        for nbr in G.successors(current):
            if nbr in visited or nbr in obstacles:
                continue
            count += 1
            w = G[current][nbr].get('weight', 1)
            tentative_g = g_score[current] + w
            checking_edges.add((current, nbr))

            if tentative_g < g_score[nbr]:
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f = tentative_g + heuristic[nbr] + dist_to_t.get(nbr, float('inf'))
                f_score[nbr] = f
                heapq.heappush(open_set, (f, tentative_g, nbr))


    return None, None, count, storage, list(checking_edges), T


with open("graphs/sejong_bus.json", "r") as json_file:
    j = json.load(json_file)

G = json_graph.node_link_graph(j)
nodes = G.nodes
edges = G.edges
stations = list(G.nodes())

with open('Experiments_result/A_star.json', 'r') as jsonfile:
    experiments_samples_A_star = json.load(jsonfile)

# table(random) LLM-A*

llm_model = Request_llm('llama')

experiments_samples_LLM_A_star = []
        
num_request = 7

experiments_samples_LLM_A_star = []

for i, sample in (enumerate(tqdm(experiments_samples_A_star))):
    start, goal = sample['point']

    heuristic_table = {station: (((G.nodes[goal]['x'] - G.nodes[station]['x']) ** 2 + (G.nodes[goal]['y'] - G.nodes[station]['y']) ** 2) ** 0.5) for station in nodes}

    waypoints = llm_model.get_waypoints(j, start, goal, num_request)
    
    path, cost, count, storage, checking_edges, waypoints_using_llm = llm_astar(G, start, goal, heuristic_table, waypoints)
    if path is not None:
        experiments_samples_LLM_A_star.append(
            {'point' : (start, goal), 
             'path':path, 
             'cost':cost, 
             'count':count, 
             'storage':storage,
             'checking_edges' : len(checking_edges),
             'waypoints_of_llm' : waypoints,
             'waypoints_using_llm' : waypoints_using_llm,
             'success' : (len(waypoints) <= num_request + 1)
             })
        cost_ratio = 1 - cost/sample['cost']
        count_ratio = 1 - count/sample['count']
        storage_ratio = 1 - storage/sample['storage']
        checking_edges_ratio = 1 - len(checking_edges)/sample['checking_edges']
        print("Sample ", i)
        print("Count ratio       :", count_ratio)
        print("Storage ratio    ",  storage_ratio)
        print("Cost ratio        :", cost_ratio)
        print("# of Checking edges ratio:", checking_edges_ratio)


with open('Experiments_result/LLM_A_star_fewshot.json', 'w') as jsonfile:
	json.dump(experiments_samples_LLM_A_star, jsonfile, indent=4)