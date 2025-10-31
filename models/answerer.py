import os
import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig
)
from typing import Dict


def get_answer(processor, model, img, qs):
    inputs = processor(img, qs, return_tensors="pt").to("cuda:0", torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text.strip()


def get_all_answers(processor, model, entity_list, qs, ent_info, input_img_path, cur_answers):
    # This should return a dict. Since a question may correspond to multiple instances of a same kind of object.
    if len(entity_list) > 1 or 'where' in qs.lower() or any([ent not in ent_info for ent in entity_list]):
        img = Image.open(input_img_path).convert('RGB')
        answer = get_answer(processor, model, img, qs)
        cur_answers.setdefault('overall', [])
        cur_answers['overall'].append((qs, answer))
    else:
        entity = entity_list[0]
        for idx, img_path in enumerate(ent_info[entity]['crop_path']):
            img = Image.open(img_path).convert('RGB')
            answer = get_answer(processor, model, img, qs)
            cur_answers.setdefault(entity, [])
            if idx + 1 > len(cur_answers[entity]):
                cur_answers[entity].append([])
            cur_answers[entity][idx].append((qs, answer))
    return cur_answers


class Answerer:
    '''
        BLIP-2 answer generator using 4-bit quantization for memory efficiency.
    '''

    def __init__(self, args):
        val_model_path = getattr(args, "val_model_path", None)
        self.args = args

        # Load processor
        self.processor = Blip2Processor.from_pretrained(val_model_path)

        # ✅ 4-bit quantization config (ultra memory-efficient)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NormalFloat4 – better accuracy
            bnb_4bit_use_double_quant=True,      # extra compression layer
            bnb_4bit_compute_dtype=torch.float16 # compute in fp16
        )

        # ✅ Auto device mapping across GPUs/CPU as needed
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            val_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate_answers(self, sample: Dict):
        generated_qs = sample['generated_questions']
        global_entity_dict = sample['entity_info']

        all_answers = []
        for gen_qs in generated_qs:
            if len(gen_qs) == 0:
                all_answers.append({})
                continue
            cur_answers = {}
            for cur_qs in gen_qs:
                qs, entity = cur_qs
                entity_list = [e.strip() for e in entity.split('.') if e.strip()]
                cur_answers = get_all_answers(
                    self.processor, self.model, entity_list, qs,
                    global_entity_dict, sample['img_path'], cur_answers
                )
            all_answers.append(cur_answers)

        sample['generated_answers'] = all_answers
        return sample
