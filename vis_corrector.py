# vis_corrector.py

from models.preprocessor import PreProcessor
from models.entity_extractor import EntityExtractor
from models.detector import Detector
from models.questioner import Questioner
from models.answerer import Answerer
from models.claim_generator import ClaimGenerator
from models.refiner import Refiner
from tqdm import tqdm
from typing import List, Dict

class Corrector:
    def __init__(self, args) -> None:
        # Initialize all submodules
        self.preprocessor = PreProcessor(args)
        self.entity_extractor = EntityExtractor(args)
        self.detector = Detector(args)
        self.questioner = Questioner(args)
        self.answerer = Answerer(args)
        if not hasattr(args, "val_model_path") or args.val_model_path is None:
            args.val_model_path = "models/blip2/blip2_flant5_xl"
        if not hasattr(args, "qa2c_model_path") or args.qa2c_model_path is None:
            args.qa2c_model_path = "models/qa2c_model"
        self.claim_generator = ClaimGenerator(args)
        self.refiner = Refiner(args)
        print("✅ All models loaded successfully.")

    def correct(self, sample: Dict):
        """
        Process a single sample dict containing:
          - 'input_desc': description text
          - 'input_img': image path
        """
        sample = self.preprocessor.generate_sentences(sample)
        sample = self.entity_extractor.extract_entity(sample)
        sample = self.detector.detect_objects(sample)
        sample = self.questioner.generate_questions(sample)
        sample = self.answerer.generate_answers(sample)
        sample = self.claim_generator.generate_claim(sample)
        sample = self.refiner.generate_output(sample)
        return sample

    def batch_correct(self, samples: List[Dict]):
        """
        Batched correction with tqdm progress.
        """
        results = []
        for s in tqdm(samples, desc="🔧 Correcting batch", total=len(samples)):
            try:
                results.append(self.correct(s))
            except Exception as e:
                print(f"[WARN] Skipped sample due to error: {e}")
                results.append(s)
        return results
