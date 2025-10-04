"""
Modules package for Woodpecker hallucination detection and correction.
"""

from .hallucination_detector import HallucinationConfidenceScorer, calculate_hcs_score

__all__ = ['HallucinationConfidenceScorer', 'calculate_hcs_score']
