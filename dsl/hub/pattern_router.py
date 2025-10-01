"""
Pattern-Based Routing System - Simple & Reliable
===============================================

Basic pattern matching for common routing decisions.
LLM fallback handles complex cases - this is just for speed optimization.
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Pattern
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of routing patterns"""
    STALE_DEALS = "stale_deals"
    DATA_QUALITY_AUDIT = "data_quality_audit"
    DUPLICATE_DETECTION = "duplicate_detection"
    OWNERLESS_DEALS = "ownerless_deals"
    ACTIVITY_TRACKING = "activity_tracking"
    RISK_SCORING = "risk_scoring"

@dataclass
class PatternMatch:
    """Result of pattern matching"""
    workflow_category: str
    parameters: Dict[str, Any]
    confidence: float
    pattern_type: PatternType
    extracted_values: Dict[str, Any]
    processing_time_ms: float
    pattern_id: str = ""
    matched_text: str = ""

class PatternBasedRouter:
    """
    Simple Pattern Router - Focus on Speed & Reliability
    
    Handles obvious cases with basic regex patterns.
    Complex cases fall back to LLM - that's perfectly fine!
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_stats = {
            'total_attempts': 0,
            'successful_matches': 0,
            'avg_processing_time_ms': 0.0
        }
        
        # Simple, reliable patterns
        self.patterns = [
            {
                'pattern_id': 'deals_at_risk_simple',
                'regex': re.compile(r'(?i).*(?:deal|opportunit).*(?:at risk|risk|stuck|stall)', re.IGNORECASE),
                'workflow_category': 'pipeline_hygiene_stale_deals',
                'pattern_type': PatternType.STALE_DEALS,
                'confidence': 0.85,
                'default_params': {'stale_days_threshold': 60}
            },
            {
                'pattern_id': 'stale_deals_with_days',
                'regex': re.compile(r'(?i).*(?:deal|opportunit).*(?:more than|over|>)\s*(\d+)\s*days?', re.IGNORECASE),
                'workflow_category': 'pipeline_hygiene_stale_deals', 
                'pattern_type': PatternType.STALE_DEALS,
                'confidence': 0.95,
                'default_params': {'stale_days_threshold': 60},
                'param_extractor': lambda m: {'stale_days_threshold': int(m.group(1))}
            },
            {
                'pattern_id': 'activity_tracking_simple',
                'regex': re.compile(r'(?i).*(?:deal|opportunit).*(?:no activit|missing activit|no call)', re.IGNORECASE),
                'workflow_category': 'activity_tracking_audit',
                'pattern_type': PatternType.ACTIVITY_TRACKING,
                'confidence': 0.80,
                'default_params': {'activity_days_threshold': 14}
            },
            {
                'pattern_id': 'data_quality_simple',
                'regex': re.compile(r'(?i).*(?:audit|check).*(?:data quality|missing.*field)', re.IGNORECASE),
                'workflow_category': 'data_quality_audit',
                'pattern_type': PatternType.DATA_QUALITY_AUDIT,
                'confidence': 0.90,
                'default_params': {}
            }
        ]
        
        self.logger.info(f"🎯 Pattern Router initialized with {len(self.patterns)} simple patterns")
    
    async def match_pattern(self, user_input: str) -> Optional[PatternMatch]:
        """Try simple pattern matching - fall back to LLM for complex cases"""
        start_time = time.time()
        self.performance_stats['total_attempts'] += 1
        
        normalized_input = user_input.lower().strip()
        
        # Try each pattern
        for pattern in self.patterns:
            match = pattern['regex'].search(normalized_input)
            if match:
                self.performance_stats['successful_matches'] += 1
                
                # Extract parameters if extractor exists
                params = pattern['default_params'].copy()
                if 'param_extractor' in pattern:
                    try:
                        extracted = pattern['param_extractor'](match)
                        params.update(extracted)
                    except Exception as e:
                        self.logger.warning(f"Parameter extraction failed: {e}")
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                result = PatternMatch(
                    workflow_category=pattern['workflow_category'],
                    parameters=params,
                    confidence=pattern['confidence'],
                    pattern_type=pattern['pattern_type'],
                    extracted_values={},
                    processing_time_ms=processing_time_ms,
                    pattern_id=pattern['pattern_id'],
                    matched_text=user_input
                )
                
                self.logger.debug(f"🎯 Pattern match: {pattern['pattern_id']} (confidence: {pattern['confidence']})")
                return result
        
        # No match - let LLM handle it
        processing_time_ms = (time.time() - start_time) * 1000
        self._update_avg_time(processing_time_ms)
        
        return None
    
    def _update_avg_time(self, processing_time_ms: float):
        """Update average processing time"""
        total_time = self.performance_stats.get('avg_processing_time_ms', 0.0) * (self.performance_stats['total_attempts'] - 1)
        self.performance_stats['avg_processing_time_ms'] = (total_time + processing_time_ms) / self.performance_stats['total_attempts']
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_attempts = self.performance_stats['total_attempts']
        successful_matches = self.performance_stats['successful_matches']
        
        return {
            'total_attempts': total_attempts,
            'successful_matches': successful_matches,
            'no_matches': total_attempts - successful_matches,
            'match_rate': (successful_matches / total_attempts * 100) if total_attempts > 0 else 0,
            'avg_processing_time_ms': self.performance_stats['avg_processing_time_ms']
        }