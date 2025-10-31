from typing import Dict
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

NUM_SECONDS_TO_SLEEP = 0.3

# ---------------------------
# 🧠 Local Text Refinement Model (FLAN-T5)
# ---------------------------
print("[Refiner] Loading local FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to("cuda" if torch.cuda.is_available() else "cpu")
print("[Refiner] Model loaded successfully.\n")

# ---------------------------
# 🧩 Prompt Template
# ---------------------------
PROMPT_TEMPLATE = """You are a helpful vision-language assistant.
Given supplementary information, a question, and a passage,
refine the passage to ensure it matches the supplementary facts.
Keep output fluent and natural.

Supplementary Info:
{sup_info}

Query:
{query}

Passage:
{text}

Refined Passage:
"""

# ---------------------------
# 🧩 Local Replacement for get_output()
# ---------------------------
def get_output(query: str, text: str, sup_info: str, max_tokens: int = 512):
    prompt = PROMPT_TEMPLATE.format(query=query, sup_info=sup_info, text=text)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# 🧩 Refiner Class
# ---------------------------
class Refiner:
    """
    Input:
        'split_sents': list[str] - Sentences from the passage.
        'claim': dict - Generated claims (counting/specific/overall).
    Output:
        'output': str - Final refined passage.
    """

    def __init__(self, args):
        self.args = args
        print("[Refiner] Using local FLAN-T5 instead of OpenAI API.\n")

    def generate_output(self, sample: Dict):
        all_claim = sample['claim']
        global_entity_dict = sample['entity_info']

        # Build supplementary info
        sup_info = ""
        sup_info += all_claim.get('counting', '')

        # Add specific info
        if 'specific' in all_claim and len(all_claim['specific']) > 0:
            sup_info += "Specific:\n"
            specific_claim_list = []
            for entity, instance_claim in all_claim['specific'].items():
                cur_entity_claim_list = []
                for idx, instance_claim_list in enumerate(instance_claim):
                    cur_inst_bbox = global_entity_dict[entity]['bbox'][idx]
                    cur_entity_claim_list.append(f"{entity} {idx + 1}: {cur_inst_bbox} " + ' '.join(instance_claim_list))
                specific_claim_list.append('\n'.join(cur_entity_claim_list))
            sup_info += '\n\n'.join(specific_claim_list)
            sup_info += '\n\n'

        # Add overall info
        if 'overall' in all_claim and len(all_claim['overall']) > 0:
            sup_info += "Overall:\n"
            sup_info += '\n'.join(all_claim['overall'])
            sup_info += '\n\n'

        # Generate refined text
        sample['output'] = get_output(sample['query'], sample['input_desc'], sup_info)
        return sample
