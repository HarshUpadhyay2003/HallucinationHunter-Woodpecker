"""
Hallucination Confidence Scoring (HCS) Module

This module implements a Hallucination Confidence Scoring system that evaluates
the confidence level of generated text descriptions against visual evidence.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import re
from collections import defaultdict

# Global scorer instance to avoid reinitialization
_global_scorer = None
_global_scorer_device = None


class HallucinationConfidenceScorer:
    """
    Hallucination Confidence Scorer (HCS) for evaluating text-image alignment.
    
    This class provides methods to score the confidence of generated descriptions
    by analyzing various factors such as entity detection, spatial relationships,
    and consistency with visual evidence.
    """
    
    def __init__(self, device: str = "cpu", silent: bool = False):
        """
        Initialize the HCS scorer.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda')
            silent: If True, suppress initialization message
        """
        self.device = device
        if not silent:
            print(f"🔹 Initializing global HCS models on device: {device}")
    
    def extract_entities_from_text(self, text: str) -> List[str]:
        """
        Extract entities from text description.
        
        Args:
            text: Input text description
            
        Returns:
            List of extracted entities
        """
        # Simple entity extraction - can be enhanced with more sophisticated NLP
        entities = []
        
        # Extract common object patterns
        object_patterns = [
            r'\b(?:a|an|the)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b',  # "a car", "the red car"
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b',  # Proper nouns
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, text.lower())
            entities.extend(matches)
        
        # Remove duplicates and filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        entities = list(set([e.strip() for e in entities if e.strip() not in stop_words]))
        
        return entities
    
    def calculate_entity_coverage_score(self, text_entities: List[str], detected_entities: Dict) -> float:
        """
        Calculate how well text entities are covered by detected visual entities.
        
        Args:
            text_entities: Entities mentioned in text
            detected_entities: Entities detected in image
            
        Returns:
            Coverage score between 0 and 1
        """
        if not text_entities:
            return 0.0
        
        detected_entity_names = set(detected_entities.keys())
        text_entity_set = set(text_entities)
        
        # Calculate intersection
        covered_entities = text_entity_set.intersection(detected_entity_names)
        
        # Calculate coverage ratio
        coverage_score = len(covered_entities) / len(text_entity_set)
        
        return coverage_score
    
    def calculate_spatial_consistency_score(self, text: str, entity_info: Dict) -> float:
        """
        Calculate spatial consistency score based on detected bounding boxes.
        
        Args:
            text: Input text description
            entity_info: Dictionary containing entity bounding box information
            
        Returns:
            Spatial consistency score between 0 and 1
        """
        if not entity_info:
            return 0.0
        
        # Extract spatial relationships from text
        spatial_keywords = ['left', 'right', 'above', 'below', 'behind', 'in front', 'near', 'far', 'center', 'corner']
        spatial_mentions = sum(1 for keyword in spatial_keywords if keyword in text.lower())
        
        # Simple heuristic: more spatial mentions should correlate with more detected entities
        num_detected_entities = len(entity_info)
        
        if num_detected_entities == 0:
            return 0.0
        
        # Normalize spatial consistency score
        spatial_score = min(spatial_mentions / num_detected_entities, 1.0)
        
        return spatial_score
    
    def calculate_detection_confidence_score(self, entity_info: Dict) -> float:
        """
        Calculate confidence score based on detection quality.
        
        Args:
            entity_info: Dictionary containing entity detection information
            
        Returns:
            Detection confidence score between 0 and 1
        """
        if not entity_info:
            return 0.0
        
        total_confidence = 0.0
        total_entities = 0
        
        for entity, info in entity_info.items():
            if 'total_count' in info and info['total_count'] > 0:
                # Higher count suggests more confident detection
                confidence = min(info['total_count'] / 5.0, 1.0)  # Normalize to max 5 detections
                total_confidence += confidence
                total_entities += 1
        
        if total_entities == 0:
            return 0.0
        
        return total_confidence / total_entities
    
    def calculate_hallucination_score(self, sample: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive hallucination confidence score.
        
        Args:
            sample: Sample dictionary containing text, image path, and entity information
            
        Returns:
            Dictionary containing various confidence scores
        """
        text = sample.get('input_desc', '')
        entity_info = sample.get('entity_info', {})
        
        # Extract entities from text
        text_entities = self.extract_entities_from_text(text)
        
        # Calculate individual scores
        entity_coverage = self.calculate_entity_coverage_score(text_entities, entity_info)
        spatial_consistency = self.calculate_spatial_consistency_score(text, entity_info)
        detection_confidence = self.calculate_detection_confidence_score(entity_info)
        
        # Calculate overall hallucination confidence score
        # Higher scores indicate lower likelihood of hallucination
        overall_score = (entity_coverage * 0.4 + spatial_consistency * 0.3 + detection_confidence * 0.3)
        
        scores = {
            'overall_hcs_score': overall_score,
            'entity_coverage_score': entity_coverage,
            'spatial_consistency_score': spatial_consistency,
            'detection_confidence_score': detection_confidence,
            'text_entities': text_entities,
            'detected_entities': list(entity_info.keys())
        }
        
        return scores
    
    def score_sample(self, sample: Dict) -> Dict:
        """
        Score a single sample for hallucination confidence.
        
        Args:
            sample: Sample dictionary containing all necessary information
            
        Returns:
            Updated sample dictionary with HCS scores
        """
        hcs_scores = self.calculate_hallucination_score(sample)
        
        # Add HCS scores to the sample
        sample['hcs_scores'] = hcs_scores
        
        return sample


# Convenience function for easy integration
def calculate_hcs_score(sample: Dict, device: str = "cpu") -> Dict:
    """
    Calculate HCS score for a sample.
    
    Uses a global scorer instance to avoid reinitialization overhead.
    
    Args:
        sample: Sample dictionary
        device: Device to run on
    
    Returns:
        Sample with HCS scores added
    """
    global _global_scorer, _global_scorer_device
    
    # Initialize global scorer only once
    if _global_scorer is None or _global_scorer_device != device:
        if _global_scorer is None:
            print("✅ HCS models loaded globally.")
        _global_scorer = HallucinationConfidenceScorer(device=device, silent=(_global_scorer is not None))
        _global_scorer_device = device
    
    return _global_scorer.score_sample(sample)
