"""
Improved Hierarchy Builder with Graph-Based Algorithm

This module provides a robust hierarchy construction algorithm that handles:
- Circular reference detection and resolution
- Email-based relationship validation  
- Orphan node handling
- Accurate level inference based on hierarchy depth
- Performance optimization for large datasets
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque, defaultdict
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class EmployeeNode:
    """Represents an employee node in the hierarchy tree"""
    email: str
    name: str
    title: str
    manager_email: Optional[str] = None
    level: Optional[str] = None
    hierarchy_depth: int = 0
    children: List['EmployeeNode'] = None
    raw_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class HierarchyValidationResult:
    """Results of hierarchy validation"""
    valid_relationships: List[Dict[str, str]]
    invalid_emails: List[Dict[str, str]]
    missing_managers: List[Dict[str, str]]
    circular_references: List[List[str]]
    orphaned_employees: List[str]
    hierarchy_health_score: float

class ImprovedHierarchyBuilder:
    """
    Advanced hierarchy builder using graph theory algorithms
    
    Features:
    - Cycle detection using DFS
    - Email validation and normalization
    - BFS-based level assignment
    - Orphan node handling
    - Performance optimized for large datasets
    """
    
    def __init__(self):
        self.email_to_node: Dict[str, EmployeeNode] = {}
        self.manager_to_reports: Dict[str, List[str]] = defaultdict(list)
        self.validation_result: Optional[HierarchyValidationResult] = None
        # DYNAMIC LEVEL MAPPINGS - Based purely on hierarchy depth from reporting relationships
        # No hardcoded assumptions about organizational structure
        self.level_mappings = self._build_dynamic_level_mappings()
        
        # Track hierarchy metrics for dynamic adjustment
        self.max_observed_depth = 0
        self.hierarchy_stats = {
            'total_employees': 0,
            'root_nodes': 0,
            'orphaned_nodes': 0,
            'max_reporting_depth': 0
        }

    def _build_dynamic_level_mappings(self) -> Dict[int, str]:
        """
        Build dynamic level mappings based on hierarchy depth.
        This is completely flexible and adapts to any organizational structure.
        """
        # Base level mapping - can be extended dynamically
        base_mappings = {
            0: 'L0',  # Top level (CEO, Founder)
            1: 'L1',  # Executive level
            2: 'L2',  # Senior leadership
            3: 'L3',  # Middle management
            4: 'L4',  # Team leadership
            5: 'L5',  # Senior individual contributors
            6: 'L6',  # Individual contributors
            7: 'L7',  # Junior level
        }
        
        # Can be extended for deeper hierarchies
        for i in range(8, 15):
            base_mappings[i] = f'L{i}'
        
        return base_mappings

    def _update_hierarchy_stats(self, nodes: List[EmployeeNode]) -> None:
        """Update hierarchy statistics for dynamic level adjustment"""
        self.hierarchy_stats['total_employees'] = len(nodes)
        
        root_count = 0
        orphaned_count = 0
        max_depth = 0
        
        for node in nodes:
            if not node.manager_email:
                root_count += 1
            if node.hierarchy_depth > max_depth:
                max_depth = node.hierarchy_depth
                
        self.hierarchy_stats['root_nodes'] = root_count
        self.hierarchy_stats['max_reporting_depth'] = max_depth
        self.max_observed_depth = max_depth

    def build_hierarchy_from_dataframe(self, df: pd.DataFrame) -> Tuple[List[EmployeeNode], HierarchyValidationResult]:
        """
        Main entry point: Build hierarchy from pandas DataFrame
        
        Args:
            df: DataFrame with employee data including email and manager_email columns
            
        Returns:
            Tuple of (root_nodes, validation_result)
        """
        logger.info(f"🏗️ Building hierarchy from {len(df)} employee records")
        
        # Step 1: Extract and normalize employee data
        employees = self._extract_employees_from_df(df)
        
        # Step 2: Validate relationships
        self.validation_result = self._validate_reporting_structure(employees)
        
        # Step 3: Build hierarchy tree
        root_nodes = self._build_hierarchy_tree(employees)
        
        # Update hierarchy statistics
        self._update_hierarchy_stats(employees)
        
        # Log hierarchy placements for debugging
        for node in employees:
            self._log_hierarchy_placement(node)
        
        # **CRITICAL FIX**: Convert email-based relationships to user_id-based for database storage
        self._convert_emails_to_user_ids(employees)
        
        logger.info(
            f"✅ Hierarchy built successfully: "
            f"{len(employees)} employees, "
            f"{len(root_nodes)} root nodes, "
            f"max depth: {self.max_observed_depth}, "
            f"health score: {self.validation_result.hierarchy_health_score:.2f}, "
            f"levels: {set(node.level for node in employees)}"
        )
        
        return root_nodes, self.validation_result

    def _extract_employees_from_df(self, df: pd.DataFrame) -> List[EmployeeNode]:
        """Extract employee data from DataFrame and create nodes"""
        employees = []
        
        for _, row in df.iterrows():
            # Normalize email addresses
            email = str(row.get('Email', '')).lower().strip()
            # Try multiple possible column names for manager email
            manager_email = (str(row.get('Reports To Email', '')) or 
                           str(row.get('Reporting Email', '')) or 
                           str(row.get('Manager Email', ''))).lower().strip()
            
            # Skip if no email
            if not email or email == 'nan':
                logger.warning(f"Skipping employee with no email: {row.get('Name', 'Unknown')}")
                continue
            
            # Clean manager email
            if manager_email == 'nan' or not manager_email:
                manager_email = None
            
            employee = EmployeeNode(
                email=email,
                name=str(row.get('Name', 'Unknown')),
                title=str(row.get('Role Title', '')),
                manager_email=manager_email,
                level=str(row.get('Level', '')) if row.get('Level') else None,
                raw_data=dict(row)
            )
            
            employees.append(employee)
            self.email_to_node[email] = employee
            
            # Build manager-to-reports mapping
            if manager_email:
                self.manager_to_reports[manager_email].append(email)
        
        logger.info(f"📊 Extracted {len(employees)} valid employee records")
        return employees
    
    def _convert_emails_to_user_ids(self, employees: List[EmployeeNode]) -> None:
        """
        Convert email-based reporting relationships to user_id-based relationships.
        This is critical for database storage where reports_to must be a user_id.
        """
        logger.info("🔄 Converting email-based relationships to user_id-based relationships...")
        
        # Create email-to-user_id mapping (simulate database auto-generated IDs)
        email_to_user_id = {}
        for i, employee in enumerate(employees, start=1):
            # Generate sequential user IDs (in real system, these would be from database)
            user_id = 2000 + i  # Start from 2000 to avoid conflicts with existing users
            email_to_user_id[employee.email] = user_id
            employee.raw_data['user_id'] = user_id
            logger.debug(f"📧 {employee.email} -> user_id: {user_id}")
        
        # Convert manager emails to manager user IDs
        conversion_count = 0
        for employee in employees:
            if employee.manager_email:
                manager_user_id = email_to_user_id.get(employee.manager_email)
                if manager_user_id:
                    employee.raw_data['manager_user_id'] = manager_user_id
                    conversion_count += 1
                    logger.debug(f"✅ {employee.name}: manager_email {employee.manager_email} -> manager_user_id {manager_user_id}")
                else:
                    # Manager not found in dataset (external manager)
                    employee.raw_data['manager_user_id'] = None
                    logger.warning(f"⚠️ {employee.name}: manager_email {employee.manager_email} not found in dataset")
            else:
                # No manager = org leader
                employee.raw_data['manager_user_id'] = None
                logger.debug(f"👑 {employee.name}: no manager = org leader")
        
        logger.info(f"✅ Converted {conversion_count} email relationships to user_id relationships")
        logger.info(f"📊 Email-to-UserID mapping: {len(email_to_user_id)} employees")
        
        # Log sample conversions for debugging
        sample_conversions = [(emp.name, emp.manager_email, emp.raw_data.get('manager_user_id')) 
                             for emp in employees[:5]]
        logger.info(f"🔍 Sample conversions: {sample_conversions}")

    def _validate_reporting_structure(self, employees: List[EmployeeNode]) -> HierarchyValidationResult:
        """Validate reporting relationships and detect issues"""
        logger.info("🔍 Validating reporting structure...")
        
        valid_relationships = []
        invalid_emails = []
        missing_managers = []
        orphaned_employees = []
        
        email_set = {emp.email for emp in employees}
        
        for employee in employees:
            if employee.manager_email:
                if employee.manager_email in email_set:
                    valid_relationships.append({
                        'employee': employee.email,
                        'manager': employee.manager_email
                    })
                else:
                    missing_managers.append({
                        'employee': employee.email,
                        'manager': employee.manager_email,
                        'action': 'treat_as_root'
                    })
            else:
                orphaned_employees.append(employee.email)
        
        # Detect circular references
        circular_references = self._detect_circular_references()
        
        # Calculate health score
        total_employees = len(employees)
        valid_count = len(valid_relationships)
        issues_count = len(missing_managers) + len(circular_references)
        health_score = (valid_count / total_employees) * 100 if total_employees > 0 else 0
        
        if issues_count > 0:
            health_score *= (1 - (issues_count / total_employees))
        
        result = HierarchyValidationResult(
            valid_relationships=valid_relationships,
            invalid_emails=invalid_emails,
            missing_managers=missing_managers,
            circular_references=circular_references,
            orphaned_employees=orphaned_employees,
            hierarchy_health_score=max(0, health_score)
        )
        
        logger.info(f"📈 Validation complete: {len(valid_relationships)} valid relationships, "
                   f"{len(circular_references)} circular references, "
                   f"{len(missing_managers)} missing managers, "
                   f"health score: {health_score:.2f}%")
        
        return result

    def _detect_circular_references(self) -> List[List[str]]:
        """Detect circular references using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs_cycle_detection(email: str, path: List[str]) -> None:
            if email in rec_stack:
                # Found a cycle
                cycle_start = path.index(email)
                cycle = path[cycle_start:] + [email]
                cycles.append(cycle)
                logger.warning(f"🔄 Detected circular reference: {' -> '.join(cycle)}")
                return
            
            if email in visited:
                return
            
            visited.add(email)
            rec_stack.add(email)
            
            # Follow reporting relationship
            if email in self.email_to_node:
                node = self.email_to_node[email]
                if node.manager_email and node.manager_email in self.email_to_node:
                    dfs_cycle_detection(node.manager_email, path + [email])
            
            rec_stack.remove(email)
        
        # Check each employee for cycles
        for email in self.email_to_node:
            if email not in visited:
                dfs_cycle_detection(email, [])
        
        return cycles

    def _build_hierarchy_tree(self, employees: List[EmployeeNode]) -> List[EmployeeNode]:
        """Build the actual hierarchy tree structure"""
        logger.info("🌳 Building hierarchy tree structure...")
        
        # Step 1: Resolve circular references by breaking weakest links
        self._resolve_circular_references()
        
        # Step 2: Find root nodes
        root_nodes = self._find_root_nodes()
        
        # Step 3: Build parent-child relationships
        self._build_parent_child_relationships()
        
        # Step 4: Assign levels using BFS
        self._assign_levels_bfs(root_nodes)
        
        # Step 5: Sort hierarchy by seniority
        root_nodes = self._sort_hierarchy_by_seniority(root_nodes)
        
        logger.info(f"🎯 Tree structure complete: {len(root_nodes)} root nodes")
        return root_nodes

    def _resolve_circular_references(self) -> None:
        """Resolve circular references by breaking the weakest link"""
        if not self.validation_result or not self.validation_result.circular_references:
            return
        
        for cycle in self.validation_result.circular_references:
            # Break the cycle by removing the manager relationship for the lowest-level employee
            weakest_link = self._find_weakest_link_in_cycle(cycle)
            if weakest_link in self.email_to_node:
                node = self.email_to_node[weakest_link]
                logger.warning(f"🔧 Breaking circular reference: removing manager for {node.name}")
                node.manager_email = None

    def _find_weakest_link_in_cycle(self, cycle: List[str]) -> str:
        """Find the employee with the lowest implied level in a cycle"""
        scores = {}
        
        for email in cycle:
            if email in self.email_to_node:
                node = self.email_to_node[email]
                score = self._calculate_seniority_score(node)
                scores[email] = score
        
        # Return the email with the lowest score (weakest link)
        return min(scores.keys(), key=lambda x: scores[x]) if scores else cycle[0]

    def _find_root_nodes(self) -> List[EmployeeNode]:
        """Find employees who don't have managers (org leaders) - they become hierarchy roots"""
        root_nodes = []
        
        for email, node in self.email_to_node.items():
            # CRITICAL: Empty/null manager_email = org leader = root of hierarchy
            if not node.manager_email or node.manager_email.strip() == "":
                root_nodes.append(node)
                logger.info(f"🏆 ORG LEADER (Root): {node.name} ({node.email}) - No reporting manager")
            elif node.manager_email not in self.email_to_node:
                root_nodes.append(node)
                logger.warning(f"📊 Orphaned root: {node.name} ({node.email}) - Manager {node.manager_email} not in dataset")
        
        if len(root_nodes) == 0:
            logger.error("❌ No root nodes found - hierarchy is invalid!")
        elif len(root_nodes) == 1:
            logger.info(f"✅ Perfect hierarchy: Single org leader found - {root_nodes[0].name}")
        else:
            logger.info(f"🏆 Multiple org leaders found: {len(root_nodes)} - All will report to RevOp Manager")
            for root in root_nodes:
                logger.info(f"📊 Org Leader: {root.name} ({root.email}) - Will report to RevOp Manager")
                # Update manager_email to indicate they should report to RevOp Manager
                root.manager_email = "revop_manager_email"  # Backend will replace with actual RevOp Manager email
        
        return root_nodes

    def _build_parent_child_relationships(self) -> None:
        """Build parent-child relationships for the tree"""
        for email, node in self.email_to_node.items():
            if node.manager_email and node.manager_email in self.email_to_node:
                manager = self.email_to_node[node.manager_email]
                manager.children.append(node)

    def _assign_levels_bfs(self, root_nodes: List[EmployeeNode]) -> None:
        """
        Assign levels using breadth-first search based PURELY on reporting hierarchy depth.
        No title analysis - levels are determined solely by reporting relationships.
        """
        queue = deque([(node, 0) for node in root_nodes])
        processed = set()
        
        while queue:
            current_node, depth = queue.popleft()
            
            if current_node.email in processed:
                continue
            
            processed.add(current_node.email)
            
            # PURE HIERARCHY-BASED LEVEL ASSIGNMENT
            # Level is determined ONLY by depth in reporting structure
            hierarchy_level = self._convert_depth_to_level(depth)
            
            current_node.level = hierarchy_level
            current_node.hierarchy_depth = depth
            
            logger.debug(f"📊 Level assigned: {current_node.name} -> {hierarchy_level} (depth: {depth}) [Reports to: {current_node.manager_email or 'None'}]")
            
            # Add children to queue
            for child in current_node.children:
                queue.append((child, depth + 1))

    def _convert_depth_to_level(self, depth: int) -> str:
        """
        Convert hierarchy depth to level string based PURELY on reporting relationships.
        No title analysis - completely dynamic and flexible.
        """
        # Ensure we have mappings for any depth
        if depth not in self.level_mappings:
            # Dynamically extend mappings for deeper hierarchies
            self.level_mappings[depth] = f'L{depth}'
        
        return self.level_mappings[depth]

    def _convert_level_to_depth(self, level: str) -> int:
        """Convert level string to depth"""
        reverse_mapping = {v: k for k, v in self.level_mappings.items()}
        return reverse_mapping.get(level, 7)

    def _log_hierarchy_placement(self, node: EmployeeNode) -> None:
        """Log hierarchy placement for debugging and validation - PURELY email-based"""
        logger.debug(
            f"🏗️ Hierarchy placement: {node.name} "
            f"(Level: {node.level}, Depth: {node.hierarchy_depth}) "
            f"Reports to: {node.manager_email or 'ROOT'} "
            f"Direct reports: {len(node.children)}"
        )

    def _calculate_seniority_score(self, node: EmployeeNode) -> int:
        """Calculate seniority score for sorting"""
        # Base score from level
        level_scores = {'M7': 100, 'M6': 90, 'M5': 80, 'M4': 70, 'M3': 60, 'M2': 50, 'M1': 40, 'IC': 30}
        base_score = level_scores.get(node.level, 30)
        
        # Adjust based on hierarchy depth (lower depth = higher score)
        depth_adjustment = max(0, 50 - (node.hierarchy_depth * 5))
        
        # Adjust based on title keywords
        title_adjustment = 0
        title_lower = node.title.lower()
        if any(keyword in title_lower for keyword in ['ceo', 'chief', 'president']):
            title_adjustment = 20
        elif any(keyword in title_lower for keyword in ['vp', 'vice president', 'director']):
            title_adjustment = 10
        elif any(keyword in title_lower for keyword in ['manager', 'lead']):
            title_adjustment = 5
        
        return base_score + depth_adjustment + title_adjustment

    def _sort_hierarchy_by_seniority(self, nodes: List[EmployeeNode]) -> List[EmployeeNode]:
        """Sort hierarchy nodes by seniority score"""
        sorted_nodes = sorted(nodes, key=self._calculate_seniority_score, reverse=True)
        
        # Recursively sort children
        for node in sorted_nodes:
            if node.children:
                node.children = self._sort_hierarchy_by_seniority(node.children)
        
        return sorted_nodes

    def convert_to_dataframe(self, root_nodes: List[EmployeeNode]) -> pd.DataFrame:
        """Convert hierarchy tree back to DataFrame format"""
        records = []
        
        def traverse_tree(node: EmployeeNode, parent_email: Optional[str] = None):
            record = {
                'Name': node.name,
                'Email': node.email,
                'Role Title': node.title,
                'Reporting Email': node.manager_email or '',
                'Level': node.level,
                'Hierarchy Depth': node.hierarchy_depth,
                'Children Count': len(node.children)
            }
            
            # Include original data if available
            if node.raw_data:
                for key, value in node.raw_data.items():
                    if key not in record:
                        record[key] = value
            
            records.append(record)
            
            # Traverse children
            for child in node.children:
                traverse_tree(child, node.email)
        
        # Traverse all root nodes
        for root in root_nodes:
            traverse_tree(root)
        
        return pd.DataFrame(records)

    def generate_hierarchy_report(self) -> Dict[str, Any]:
        """Generate comprehensive hierarchy analysis report"""
        if not self.validation_result:
            return {"error": "No validation result available"}
        
        total_employees = len(self.email_to_node)
        root_count = len([node for node in self.email_to_node.values() 
                         if not node.manager_email or node.manager_email not in self.email_to_node])
        
        # Calculate depth distribution
        depth_distribution = defaultdict(int)
        level_distribution = defaultdict(int)
        
        for node in self.email_to_node.values():
            depth_distribution[node.hierarchy_depth] += 1
            level_distribution[node.level or 'Unknown'] += 1
        
        # Calculate span of control statistics
        span_of_control = [len(node.children) for node in self.email_to_node.values() if node.children]
        avg_span = sum(span_of_control) / len(span_of_control) if span_of_control else 0
        max_span = max(span_of_control) if span_of_control else 0
        
        return {
            "hierarchy_health": {
                "health_score": self.validation_result.hierarchy_health_score,
                "total_employees": total_employees,
                "valid_relationships": len(self.validation_result.valid_relationships),
                "issues_count": len(self.validation_result.missing_managers) + 
                               len(self.validation_result.circular_references)
            },
            "structure_analysis": {
                "root_nodes_count": root_count,
                "max_depth": max(depth_distribution.keys()) if depth_distribution else 0,
                "avg_span_of_control": round(avg_span, 2),
                "max_span_of_control": max_span,
                "depth_distribution": dict(depth_distribution),
                "level_distribution": dict(level_distribution)
            },
            "issues_detected": {
                "circular_references": self.validation_result.circular_references,
                "missing_managers": self.validation_result.missing_managers,
                "orphaned_employees": self.validation_result.orphaned_employees
            },
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on hierarchy analysis"""
        recommendations = []
        
        if not self.validation_result:
            return recommendations
        
        # Health score recommendations
        if self.validation_result.hierarchy_health_score < 70:
            recommendations.append("Hierarchy health is below 70%. Review and clean reporting relationships.")
        
        # Circular reference recommendations
        if self.validation_result.circular_references:
            recommendations.append(f"Found {len(self.validation_result.circular_references)} circular references. "
                                 "These have been automatically resolved but should be corrected in source data.")
        
        # Missing managers recommendations
        if self.validation_result.missing_managers:
            recommendations.append(f"{len(self.validation_result.missing_managers)} employees have managers "
                                 "not found in dataset. Verify manager email addresses.")
        
        # Span of control recommendations
        span_of_control = [len(node.children) for node in self.email_to_node.values() if node.children]
        if span_of_control and max(span_of_control) > 15:
            recommendations.append("Some managers have very large teams (>15 direct reports). "
                                 "Consider organizational restructuring.")
        
        return recommendations
