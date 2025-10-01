"""
Centralized Module Assignment Logic
Single source of truth for assigning modules to users based on 4-category system:
1. Core Modules (Everyone)
2. Industry-Based Modules (All users in that industry)
3. Leadership Modules (Everyone except IC/Leaf nodes)
4. RevOps-Only Modules (Only revenue_manager role)
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CentralizedModuleAssignment:
    """
    Centralized module assignment based on 4-category system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 1. CORE MODULES - Everyone gets these
        self.core_modules = [
            'Calendar',
            'Lets Meet', 
            'Cruxx',
            'RevAIPro Action Center',
            'Rhythm of Business',
            'JIT',
            'Tell Me'
        ]
        
        # 2. INDUSTRY-BASED MODULES
        self.industry_modules = {
            'saas': [
                'Forecasting',
                'Pipeline', 
                'Planning',
                'DealDesk',
                'My Customers',
                'Partner Management',
                'Customer Success',
                'Sales Engineering',
                'Marketing operations',
                'Contract Management',
                'Market Intelligence',
                'Compensation'
            ],
            'banking': [
                # Banking-specific modules would go here
                'Forecasting',
                'Pipeline',
                'Planning'
            ],
            'insurance': [
                # Insurance-specific modules would go here
                'Forecasting',
                'Pipeline',
                'Planning'
            ]
            # Add more industries as needed
        }
        
        # 3. LEADERSHIP MODULES - Everyone except IC/Leaf nodes
        self.leadership_modules = [
            'Enablement',
            'Performance Management'
        ]
        
        # 4. REVOPS-ONLY MODULES - Only revenue_manager role
        self.revops_modules = [
            'RevOps Engineering'  # This includes all submodules in the sidebar
        ]
    
    def assign_modules(self, 
                      user_role: str,
                      job_title: str = "",
                      role_function: str = "",
                      industry: str = "saas",
                      has_direct_reports: bool = False,
                      is_leaf_node: bool = True) -> List[str]:
        """
        Assign modules to a user based on the centralized 4-category system
        
        Args:
            user_role: The user's role (admin, revenue_manager, etc.)
            job_title: Job title for leadership detection
            role_function: Role function for additional context
            industry: Industry type (saas, banking, insurance, etc.)
            has_direct_reports: Whether user has people reporting to them
            is_leaf_node: Whether user is a leaf node in hierarchy (IC)
            
        Returns:
            List of module names assigned to the user
        """
        assigned_modules = []
        
        # 1. CORE MODULES - Everyone gets these
        assigned_modules.extend(self.core_modules)
        self.logger.debug(f"✅ Added core modules: {self.core_modules}")
        
        # 2. INDUSTRY-BASED MODULES - All users in that industry
        industry_key = industry.lower().replace(' ', '').replace('_', '')
        if industry_key in self.industry_modules:
            assigned_modules.extend(self.industry_modules[industry_key])
            self.logger.debug(f"✅ Added {industry} industry modules: {self.industry_modules[industry_key]}")
        else:
            # Fallback to SaaS modules if industry not found
            assigned_modules.extend(self.industry_modules['saas'])
            self.logger.warning(f"⚠️ Industry '{industry}' not found, using SaaS modules as fallback")
        
        # 3. LEADERSHIP MODULES - Everyone except IC/Leaf nodes
        is_leadership = self._detect_leadership(job_title, role_function, has_direct_reports, is_leaf_node)
        if is_leadership:
            assigned_modules.extend(self.leadership_modules)
            self.logger.debug(f"✅ Added leadership modules: {self.leadership_modules}")
        else:
            self.logger.debug(f"⏭️ Skipped leadership modules (user is IC/leaf node)")
        
        # 4. REVOPS-ONLY MODULES - Only revenue_manager role
        if user_role == 'revenue_manager':
            assigned_modules.extend(self.revops_modules)
            self.logger.debug(f"✅ Added RevOps modules: {self.revops_modules}")
        
        # Remove duplicates and return
        final_modules = list(dict.fromkeys(assigned_modules))  # Preserves order, removes duplicates
        
        self.logger.info(f"🎯 Final modules for user (role: {user_role}, industry: {industry}, leadership: {is_leadership}): {final_modules}")
        
        return final_modules
    
    def _detect_leadership(self, 
                          job_title: str, 
                          role_function: str, 
                          has_direct_reports: bool, 
                          is_leaf_node: bool) -> bool:
        """
        Detect if user is in leadership position
        
        Leadership criteria:
        1. Job title contains "Manager", "Director", "VP", "Head", "Chief", "SVP", etc.
        2. Has direct reports
        3. Not a leaf node in hierarchy
        
        Args:
            job_title: User's job title
            role_function: User's role function
            has_direct_reports: Whether user has direct reports
            is_leaf_node: Whether user is leaf node in hierarchy
            
        Returns:
            True if user is in leadership position
        """
        # Check job title for leadership keywords
        title_lower = (job_title or "").lower()
        function_lower = (role_function or "").lower()
        
        leadership_keywords = [
            'manager', 'director', 'vp', 'head', 'chief', 'svp', 
            'senior manager', 'team lead', 'supervisor', 'lead'
        ]
        
        has_leadership_title = any(keyword in title_lower + " " + function_lower for keyword in leadership_keywords)
        
        # Leadership if: has leadership title OR has direct reports OR not a leaf node
        is_leadership = has_leadership_title or has_direct_reports or not is_leaf_node
        
        self.logger.debug(f"🔍 Leadership detection for '{job_title}' + '{role_function}':")
        self.logger.debug(f"  - Leadership title: {has_leadership_title}")
        self.logger.debug(f"  - Has direct reports: {has_direct_reports}")
        self.logger.debug(f"  - Is leaf node: {is_leaf_node}")
        self.logger.debug(f"  - Final leadership status: {is_leadership}")
        
        return is_leadership
    
    def get_modules_by_category(self) -> Dict[str, List[str]]:
        """
        Get modules organized by category for documentation/debugging
        
        Returns:
            Dictionary with categories and their modules
        """
        return {
            'core': self.core_modules,
            'industry_saas': self.industry_modules.get('saas', []),
            'industry_banking': self.industry_modules.get('banking', []),
            'industry_insurance': self.industry_modules.get('insurance', []),
            'leadership': self.leadership_modules,
            'revops_only': self.revops_modules
        }
    
    def validate_module_assignment(self, assigned_modules: List[str]) -> Dict[str, Any]:
        """
        Validate that module assignment follows the rules
        
        Args:
            assigned_modules: List of assigned modules
            
        Returns:
            Validation results with any issues found
        """
        issues = []
        
        # Check that core modules are present
        missing_core = [m for m in self.core_modules if m not in assigned_modules]
        if missing_core:
            issues.append(f"Missing core modules: {missing_core}")
        
        # Check for unknown modules
        all_known_modules = (
            self.core_modules + 
            self.leadership_modules + 
            self.revops_modules +
            [m for modules in self.industry_modules.values() for m in modules]
        )
        
        unknown_modules = [m for m in assigned_modules if m not in all_known_modules]
        if unknown_modules:
            issues.append(f"Unknown modules: {unknown_modules}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_modules': len(assigned_modules),
            'core_modules_count': len([m for m in assigned_modules if m in self.core_modules]),
            'leadership_modules_count': len([m for m in assigned_modules if m in self.leadership_modules]),
            'revops_modules_count': len([m for m in assigned_modules if m in self.revops_modules])
        }


# Global instance for easy import
centralized_module_assignment = CentralizedModuleAssignment()
