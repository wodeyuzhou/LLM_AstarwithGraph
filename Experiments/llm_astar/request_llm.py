import re
from .prompts import Qwen_prompt, Llama_prompt, Deepseek_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def get_nums_list(s):
    nums_str = re.findall(r'\d+', s)  
    nums = list(map(int, nums_str))
    return  nums

class Request_llm:
    def __init__(self, model_name, prompt_type):

        if model_name == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct",
                torch_dtype="auto",
                device_map="auto"
            )

            if prompt_type == 'fewshot':
                self.prompter = Llama_prompt.few_shot
            else : 
                self.prompter = Llama_prompt.cot
        
        if model_name == 'deepseek':
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
            self.model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer.model_max_length = self.model.config.max_position_embeddings

            if prompt_type == 'fewshot':
                self.prompter = Deepseek_prompt.few_shot
            else : 
                self.prompter = Deepseek_prompt.cot

        if model_name == 'qwen':

            rope = {
                "rope_type": "yarn",
                "factor": 4.0,                         
                "original_max_position_embeddings": 32768
            }

            cfg = AutoConfig.from_pretrained(
                    "Qwen/Qwen3-8B",
                    rope_scaling=rope
                    )

            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-8B",
                torch_dtype="auto",
                config=cfg,
                device_map="auto"
            )

            if prompt_type == 'fewshot':
                self.prompter = Qwen_prompt.few_shot
            else : 
                self.prompter = Qwen_prompt.cot

        self.max_new_tokens = 200
        self.think_index = self.tokenizer.encode("</think>")[-1]
        
    def get_waypoints(self, G, start, goal, n_points):

        text = self.prompter(G, start, goal, n_points)
        inputs = self.tokenizer([text], return_tensors="pt", return_attention_mask=True).to(self.model.device)
        input_ids = inputs["input_ids"]
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.model.device)
        generated_ids = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens
            )
        output_ids = generated_ids[0][len(input_ids[0]):].tolist() 
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(self.think_index)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return  get_nums_list(content)