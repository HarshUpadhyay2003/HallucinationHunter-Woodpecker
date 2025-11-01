import os
from typing import Dict
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def format_inputs(question: str, answer: str):
    """Combine question and answer into a single input prompt."""
    return f"Answer: {answer}\nQuestion: {question}"

def get_claim(processor, model, question, answer):
    """Generate claim text using BLIP-2 model."""
    input_text = format_inputs(question, answer)

    inputs = processor(
        text=input_text,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=64, num_beams=4, early_stopping=True)
    claim = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return claim

class ClaimGenerator:
    '''
        Input:
            dict: 
                'generated_questions': list of [question, entity]
                'generated_answers': list of dicts like:
                    {
                        "overall": [(qs, ans), ...],
                        "entity": [
                            [(qs, ans), ...],   # instance 1
                            ...
                        ]
                    }

        Output:
            dict:
                'claim': {
                    "specific": { entity: [[claim1, claim2, ...], ...] },
                    "overall": [...],
                    "counting": "Counting info text"
                }
    '''

    def __init__(self, args):
        self.args = args
        qa2c_model_path = args.qa2c_model_path

        print(f"[ClaimGenerator] Loading BLIP-2 model from: {qa2c_model_path}")
        self.processor = Blip2Processor.from_pretrained(qa2c_model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(qa2c_model_path).to('cuda:0')
        print("[ClaimGenerator] BLIP-2 model loaded successfully.")

    def generate_claim(self, sample: Dict):
        """Generate claims combining Q&A and counting information."""
        all_claim = {}

        # ---- Q&A-based claims ----
        generated_answers = sample['generated_answers']
        for answer_dict in generated_answers:
            for entity, answer_list in answer_dict.items():
                if entity == 'overall':
                    all_claim.setdefault('overall', [])
                    for qs, ans in answer_list:
                        clm = get_claim(self.processor, self.model, qs, ans)
                        all_claim['overall'] += clm
                else:
                    all_claim.setdefault('specific', {}).setdefault(entity, [])
                    for idx, entity_answer_list in enumerate(answer_list):
                        if idx + 1 > len(all_claim['specific'][entity]):
                            all_claim['specific'][entity].append([])
                        for qs, ans in entity_answer_list:
                            clm = get_claim(self.processor, self.model, qs, ans)
                            all_claim['specific'][entity][idx] += clm

        # ---- Counting-based claims ----
        counting_claim = "Counting:\n"
        for entity, ent_info in sample['entity_info'].items():
            ent_counts = ent_info['total_count']
            if ent_counts == 0:
                counting_claim += f"There is no {entity}.\n\n"
                continue
            else:
                counting_claim += f"There are {ent_counts} {entity}.\n"
                for idx, bbox in enumerate(ent_info['bbox']):
                    counting_claim += f"{entity} {idx+1}: {bbox}\n"
                counting_claim += "\n"

        all_claim['counting'] = counting_claim
        sample['claim'] = all_claim
        return sample
