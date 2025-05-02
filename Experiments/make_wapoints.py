import json
from tqdm.auto import tqdm
from llm_astar import Request_llm
from networkx.readwrite import json_graph

with open('./Result/A_star.json', 'r') as jsonfile:
    A_star = json.load(jsonfile)

with open("../graphs/sejong_bus.json", "r") as json_file:
    j = json.load(json_file)

G = json_graph.node_link_graph(j)

llm_model = Request_llm('qwen', 'fewshot')
Qwen_fewshot = []

num_request = 7
for i, sample in (enumerate(tqdm(A_star))):
    start, goal = sample['point']
    waypoints = llm_model.get_waypoints(G, start, goal, num_request)
    
    Qwen_fewshot.append(
            {'point' : (start, goal), 
             'path':None, 
             'cost':None, 
             'count':None, 
             'storage':None,
             'checking_edges' : None,
             'waypoints_of_llm' : waypoints,
             'waypoints_using_llm' : None,
             'success' : (len(waypoints) <= num_request + 2)
             })