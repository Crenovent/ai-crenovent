"""
Confidence Scoring Engine
Calculates confidence scores for field assignments and identifies low-confidence assignments.

Features:
- Field-level confidence scoring
- Overall assignment confidence
- Low-confidence identification
- Assignment validation
- Confidence calibration
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfidenceScoringEngine:
    """
    Calculates confidence scores for RBA field assignments
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'very_low': 0.2
        }
        
        # Field importance weights
        self.field_weights = {
            'region': 0.25,
            'segment': 0.20,
            'territory': 0.15,
            'area': 0.15,
            'district': 0.10,
            'level': 0.10,
            'modules': 0.05
        }

    def calculate_assignment_confidence(self, user_data: Dict[str, Any], assignments: Dict[str, Any], 
                                      company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate confidence scores for all field assignments
        """
        try:
            field_confidences = {}
            confidence_details = {}
            
            # Calculate confidence for each assigned field
            for field_name, assigned_value in assignments.items():
                if field_name in self.field_weights:
                    confidence_result = self._calculate_field_confidence(
                        field_name, assigned_value, user_data, company_profile
                    )
                    field_confidences[field_name] = confidence_result['score']
                    confidence_details[field_name] = confidence_result['details']
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(field_confidences)
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(overall_confidence)
            
            # Identify validation issues
            validation_issues = self._validate_assignments(assignments, user_data, company_profile)
            
            # Determine if review is required
            requires_review = (
                overall_confidence < self.confidence_thresholds['medium'] or
                len(validation_issues) > 0
            )
            
            return {
                'overall_confidence': round(overall_confidence, 3),
                'confidence_level': confidence_level,
                'field_confidences': field_confidences,
                'confidence_details': confidence_details,
                'validation_issues': validation_issues,
                'requires_review': requires_review,
                'low_confidence_fields': [
                    field for field, score in field_confidences.items() 
                    if score < self.confidence_thresholds['medium']
                ]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Confidence calculation failed: {e}")
            return self._get_default_confidence_result()

    def _calculate_field_confidence(self, field_name: str, assigned_value: str, 
                                  user_data: Dict[str, Any], company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate confidence score for a specific field assignment
        """
        confidence_factors = []
        details = []
        
        # Factor 1: Data quality and completeness
        data_quality_score = self._assess_data_quality(user_data)
        confidence_factors.append(data_quality_score * 0.3)
        details.append(f"Data quality: {data_quality_score:.2f}")
        
        # Factor 2: Assignment method confidence
        method_confidence = self._assess_assignment_method(field_name, assigned_value, user_data)
        confidence_factors.append(method_confidence * 0.4)
        details.append(f"Assignment method: {method_confidence:.2f}")
        
        # Factor 3: Consistency with company profile
        profile_consistency = self._assess_profile_consistency(field_name, assigned_value, company_profile)
        confidence_factors.append(profile_consistency * 0.2)
        details.append(f"Profile consistency: {profile_consistency:.2f}")
        
        # Factor 4: Cross-field validation
        cross_validation = self._assess_cross_field_validation(field_name, assigned_value, user_data)
        confidence_factors.append(cross_validation * 0.1)
        details.append(f"Cross-field validation: {cross_validation:.2f}")
        
        # Calculate final score
        final_score = sum(confidence_factors)
        
        return {
            'score': round(final_score, 3),
            'details': details,
            'factors': {
                'data_quality': data_quality_score,
                'method_confidence': method_confidence,
                'profile_consistency': profile_consistency,
                'cross_validation': cross_validation
            }
        }

    def _assess_data_quality(self, user_data: Dict[str, Any]) -> float:
        """
        Assess the quality and completeness of input data
        """
        required_fields = ['email', 'name', 'job_title']
        optional_fields = ['department', 'manager', 'location', 'phone']
        
        # Check required fields
        required_score = sum(1 for field in required_fields if user_data.get(field))
        required_ratio = required_score / len(required_fields)
        
        # Check optional fields
        optional_score = sum(1 for field in optional_fields if user_data.get(field))
        optional_ratio = optional_score / len(optional_fields)
        
        # Assess data richness
        total_fields = len([v for v in user_data.values() if v and str(v).strip()])
        richness_score = min(1.0, total_fields / 10)  # Normalize to max 10 fields
        
        # Combined data quality score
        return (required_ratio * 0.6) + (optional_ratio * 0.2) + (richness_score * 0.2)

    def _assess_assignment_method(self, field_name: str, assigned_value: str, user_data: Dict[str, Any]) -> float:
        """
        Assess confidence based on the assignment method used
        """
        job_title = str(user_data.get('job_title', '')).lower()
        department = str(user_data.get('department', '')).lower()
        
        if field_name == 'region':
            # High confidence for location-based or title-based assignment
            if user_data.get('location') or 'global' in job_title or 'international' in job_title:
                return 0.9
            elif any(geo in job_title for geo in ['americas', 'emea', 'apac']):
                return 0.8
            else:
                return 0.5  # Hash-based assignment
        
        elif field_name == 'segment':
            # High confidence for title-based segment assignment
            if any(seg in job_title for seg in ['enterprise', 'strategic', 'commercial', 'smb']):
                return 0.9
            elif any(level in job_title for level in ['vp', 'director', 'manager']):
                return 0.7
            else:
                return 0.6
        
        elif field_name == 'level':
            # High confidence for title-based level detection
            level_indicators = ['ceo', 'vp', 'director', 'manager', 'senior', 'junior']
            if any(indicator in job_title for indicator in level_indicators):
                return 0.9
            else:
                return 0.4
        
        elif field_name in ['territory', 'area', 'district']:
            # Medium confidence for geographic assignments
            if user_data.get('location'):
                return 0.8
            else:
                return 0.5
        
        else:
            return 0.6  # Default confidence

    def _assess_profile_consistency(self, field_name: str, assigned_value: str, company_profile: Dict[str, Any]) -> float:
        """
        Assess consistency with detected company profile
        """
        model_type = company_profile.get('detected_hierarchy_model', 'custom')
        
        if field_name == 'region' and model_type in ['geographic', 'hybrid']:
            return 0.9
        elif field_name == 'segment' and model_type in ['segment', 'hybrid']:
            return 0.9
        elif field_name in ['territory', 'area'] and model_type == 'geographic':
            return 0.8
        else:
            return 0.6

    def _assess_cross_field_validation(self, field_name: str, assigned_value: str, user_data: Dict[str, Any]) -> float:
        """
        Assess consistency across related fields
        """
        # Check for logical consistency between fields
        job_title = str(user_data.get('job_title', '')).lower()
        
        if field_name == 'segment':
            # Check if segment aligns with job level
            if 'enterprise' in assigned_value.lower():
                if any(level in job_title for level in ['vp', 'director', 'senior']):
                    return 0.9
                else:
                    return 0.6
            elif 'smb' in assigned_value.lower():
                if any(level in job_title for level in ['coordinator', 'specialist', 'associate']):
                    return 0.9
                else:
                    return 0.6
        
        return 0.7  # Default cross-validation score

    def _calculate_overall_confidence(self, field_confidences: Dict[str, float]) -> float:
        """
        Calculate weighted overall confidence score
        """
        if not field_confidences:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for field_name, confidence in field_confidences.items():
            weight = self.field_weights.get(field_name, 0.1)
            weighted_sum += confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _get_confidence_level(self, confidence_score: float) -> str:
        """
        Convert confidence score to categorical level
        """
        if confidence_score >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence_score >= self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence_score >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'

    def _validate_assignments(self, assignments: Dict[str, Any], user_data: Dict[str, Any], 
                            company_profile: Dict[str, Any]) -> List[str]:
        """
        Validate assignments for logical consistency
        """
        issues = []
        
        # Check for missing critical assignments
        if not assignments.get('region'):
            issues.append("Missing region assignment")
        
        if not assignments.get('segment'):
            issues.append("Missing segment assignment")
        
        # Check for inconsistent level assignment
        job_title = str(user_data.get('job_title', '')).lower()
        assigned_level = assignments.get('level', '').lower()
        
        if 'ceo' in job_title and assigned_level != 'c_level':
            issues.append("CEO title but not assigned C-level")
        
        if 'vp' in job_title and assigned_level not in ['vp_level', 'c_level']:
            issues.append("VP title but not assigned VP level")
        
        # Check geographic consistency
        if assignments.get('region') == 'north_america' and assignments.get('territory') in ['london', 'paris']:
            issues.append("Geographic inconsistency: NA region with European territory")
        
        return issues

    def _get_default_confidence_result(self) -> Dict[str, Any]:
        """
        Return default confidence result when calculation fails
        """
        return {
            'overall_confidence': 0.5,
            'confidence_level': 'medium',
            'field_confidences': {},
            'confidence_details': {},
            'validation_issues': ['Confidence calculation failed'],
            'requires_review': True,
            'low_confidence_fields': []
        }

    def batch_calculate_confidence(self, assignments_batch: List[Dict[str, Any]], 
                                 company_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate confidence scores for a batch of assignments
        """
        results = []
        
        for assignment_data in assignments_batch:
            user_data = assignment_data.get('user_data', {})
            assignments = assignment_data.get('assignments', {})
            
            confidence_result = self.calculate_assignment_confidence(
                user_data, assignments, company_profile
            )
            
            results.append({
                'user_email': user_data.get('email'),
                'confidence_result': confidence_result
            })
        
        return results

    def get_confidence_statistics(self, confidence_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics from confidence results
        """
        if not confidence_results:
            return {}
        
        confidence_scores = [r['confidence_result']['overall_confidence'] for r in confidence_results]
        requires_review_count = sum(1 for r in confidence_results if r['confidence_result']['requires_review'])
        
        return {
            'total_assignments': len(confidence_results),
            'average_confidence': round(sum(confidence_scores) / len(confidence_scores), 3),
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores),
            'requires_review_count': requires_review_count,
            'requires_review_percentage': round((requires_review_count / len(confidence_results)) * 100, 1),
            'confidence_distribution': {
                'high': sum(1 for score in confidence_scores if score >= self.confidence_thresholds['high']),
                'medium': sum(1 for score in confidence_scores if self.confidence_thresholds['medium'] <= score < self.confidence_thresholds['high']),
                'low': sum(1 for score in confidence_scores if self.confidence_thresholds['low'] <= score < self.confidence_thresholds['medium']),
                'very_low': sum(1 for score in confidence_scores if score < self.confidence_thresholds['low'])
            }
        }
