"""
Time-bomb Guard Service - Task 6.2.40
======================================

Time-bomb guard (infinite loop patterns)
- Detects patterns that could lead to infinite loops or long-lived executions
- Enforces safeguards such as loop counters or timeouts
- Static heuristics and analyzer for pattern detection
- Requires explicit loop guards or wall-clock timeouts
- Backend implementation (no actual runtime instrumentation - that's runtime infrastructure)

Dependencies: Task 6.2.4 (IR), Task 6.2.39 (Complexity Limits)
Outputs: Loop safety analysis → prevents infinite execution and ensures bounded runtime
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class LoopType(Enum):
    """Types of loops detected"""
    FOR_LOOP = "for_loop"           # Bounded iteration
    WHILE_LOOP = "while_loop"       # Condition-based loop
    RECURSIVE_CALL = "recursive_call"  # Recursive function calls
    CYCLE_GRAPH = "cycle_graph"     # Graph cycles
    RETRY_LOOP = "retry_loop"       # Retry mechanisms
    POLLING_LOOP = "polling_loop"   # Polling operations

class LoopRisk(Enum):
    """Risk levels for loops"""
    SAFE = "safe"                   # Provably bounded
    LOW_RISK = "low_risk"          # Likely safe with guards
    MEDIUM_RISK = "medium_risk"    # Needs careful review
    HIGH_RISK = "high_risk"        # Dangerous patterns
    CRITICAL = "critical"          # Almost certainly problematic

class GuardType(Enum):
    """Types of loop guards"""
    ITERATION_COUNTER = "iteration_counter"     # Max iterations limit
    WALL_CLOCK_TIMEOUT = "wall_clock_timeout"   # Time-based timeout
    RESOURCE_LIMIT = "resource_limit"           # Memory/CPU limits
    CIRCUIT_BREAKER = "circuit_breaker"         # Circuit breaker pattern
    WATCHDOG_TIMER = "watchdog_timer"           # External watchdog
    CONDITION_VALIDATOR = "condition_validator"  # Validate termination conditions

@dataclass
class LoopGuard:
    """Loop guard specification"""
    guard_id: str
    guard_type: GuardType
    
    # Guard parameters
    max_iterations: Optional[int] = None
    timeout_seconds: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    
    # Condition validation
    termination_condition: Optional[str] = None
    progress_check: Optional[str] = None
    
    # Behavior on violation
    action_on_violation: str = "terminate"  # terminate, warn, throttle
    
    # Metadata
    description: str = ""
    rationale: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class LoopPattern:
    """Detected loop pattern"""
    pattern_id: str
    loop_type: LoopType
    risk_level: LoopRisk
    
    # Location information
    node_id: str
    node_type: str
    source_location: Optional[str] = None
    
    # Pattern details
    loop_condition: Optional[str] = None
    iteration_variable: Optional[str] = None
    termination_criteria: List[str] = field(default_factory=list)
    
    # Risk analysis
    risk_factors: List[str] = field(default_factory=list)
    unbounded_indicators: List[str] = field(default_factory=list)
    
    # Guard requirements
    required_guards: List[GuardType] = field(default_factory=list)
    recommended_guards: List[GuardType] = field(default_factory=list)
    
    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class TimeBombAnalysis:
    """Complete time-bomb analysis results"""
    analysis_id: str
    plan_id: str
    
    # Detected patterns
    loop_patterns: List[LoopPattern] = field(default_factory=list)
    
    # Risk assessment
    overall_risk_level: LoopRisk = LoopRisk.SAFE
    total_risk_score: float = 0.0
    
    # Guard analysis
    missing_guards: List[str] = field(default_factory=list)
    insufficient_guards: List[str] = field(default_factory=list)
    recommended_guards: List[LoopGuard] = field(default_factory=list)
    
    # Violations and issues
    safety_violations: List[str] = field(default_factory=list)
    potential_infinite_loops: List[str] = field(default_factory=list)
    
    # Recommendations
    remediation_steps: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Performance estimates
    estimated_max_execution_time: Optional[float] = None
    estimated_max_iterations: Optional[int] = None
    
    # Metadata
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_duration_ms: float = 0.0

@dataclass
class LoopInstrumentationPlan:
    """Plan for instrumenting loops with time-bomb guards"""
    plan_id: str
    instrumentation_id: str
    
    # Instrumentation points
    guard_insertions: List[Dict[str, Any]] = field(default_factory=list)
    timeout_injections: List[Dict[str, Any]] = field(default_factory=list)
    counter_insertions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Runtime configuration
    global_timeout_seconds: float = 300.0  # 5 minutes default
    global_max_iterations: int = 10000
    enable_progress_monitoring: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Task 6.2.40: Time-bomb Guard Service
class TimeBombGuardService:
    """Service for detecting and preventing infinite loop patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection rules
        self.loop_detection_patterns = self._initialize_loop_patterns()
        self.risk_assessment_rules = self._initialize_risk_rules()
        
        # Guard templates
        self.guard_templates = self._initialize_guard_templates()
        
        # Analysis cache
        self.analysis_cache: Dict[str, TimeBombAnalysis] = {}
        
        # Statistics
        self.timebomb_stats = {
            'total_analyses': 0,
            'loops_detected': 0,
            'high_risk_patterns': 0,
            'guards_recommended': 0,
            'safety_violations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'loop_types': {loop_type.value: 0 for loop_type in LoopType},
            'risk_levels': {risk.value: 0 for risk in LoopRisk}
        }
    
    def analyze_time_bomb_patterns(self, plan_ir: Dict[str, Any]) -> TimeBombAnalysis:
        """
        Analyze a plan for time-bomb patterns (infinite loops, unbounded execution)
        
        Args:
            plan_ir: Plan intermediate representation
            
        Returns:
            TimeBombAnalysis with detailed results
        """
        start_time = datetime.now(timezone.utc)
        plan_id = plan_ir.get('plan_id', 'unknown')
        
        # Check cache
        cache_key = f"{plan_id}_{hash(str(plan_ir))}"
        if cache_key in self.analysis_cache:
            self.timebomb_stats['cache_hits'] += 1
            return self.analysis_cache[cache_key]
        
        self.timebomb_stats['cache_misses'] += 1
        
        # Create analysis instance
        analysis_id = f"timebomb_{plan_id}_{hash(str(plan_ir))}_{int(start_time.timestamp())}"
        analysis = TimeBombAnalysis(
            analysis_id=analysis_id,
            plan_id=plan_id
        )
        
        # Detect loop patterns
        analysis.loop_patterns = self._detect_loop_patterns(plan_ir)
        
        # Assess overall risk
        analysis.overall_risk_level, analysis.total_risk_score = self._assess_overall_risk(analysis.loop_patterns)
        
        # Analyze guard requirements
        self._analyze_guard_requirements(analysis, plan_ir)
        
        # Check for safety violations
        self._check_safety_violations(analysis)
        
        # Generate recommendations
        analysis.remediation_steps = self._generate_remediation_steps(analysis)
        analysis.optimization_suggestions = self._generate_optimization_suggestions(analysis)
        
        # Estimate execution bounds
        analysis.estimated_max_execution_time = self._estimate_max_execution_time(analysis.loop_patterns)
        analysis.estimated_max_iterations = self._estimate_max_iterations(analysis.loop_patterns)
        
        # Record analysis duration
        end_time = datetime.now(timezone.utc)
        analysis.analysis_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Cache result
        self.analysis_cache[cache_key] = analysis
        
        # Update statistics
        self._update_timebomb_stats(analysis)
        
        self.logger.info(f"✅ Analyzed time-bomb patterns: {plan_id} -> {analysis.overall_risk_level.value} risk ({len(analysis.loop_patterns)} patterns)")
        
        return analysis
    
    def validate_loop_safety(self, plan_ir: Dict[str, Any], 
                           strict_mode: bool = False) -> Dict[str, Any]:
        """
        Validate that all loops have appropriate safety guards
        
        Args:
            plan_ir: Plan intermediate representation
            strict_mode: If True, require guards for all loops
            
        Returns:
            Dictionary with validation results
        """
        analysis = self.analyze_time_bomb_patterns(plan_ir)
        
        # Determine validation result
        is_safe = True
        issues = []
        
        # Check for high-risk patterns
        high_risk_patterns = [p for p in analysis.loop_patterns if p.risk_level in [LoopRisk.HIGH_RISK, LoopRisk.CRITICAL]]
        if high_risk_patterns:
            is_safe = False
            issues.append(f"Found {len(high_risk_patterns)} high-risk loop patterns")
        
        # Check for missing guards
        if analysis.missing_guards:
            if strict_mode:
                is_safe = False
            issues.append(f"Missing guards: {', '.join(analysis.missing_guards)}")
        
        # Check for safety violations
        if analysis.safety_violations:
            is_safe = False
            issues.extend(analysis.safety_violations)
        
        # Check for potential infinite loops
        if analysis.potential_infinite_loops:
            is_safe = False
            issues.extend(analysis.potential_infinite_loops)
        
        return {
            'safe': is_safe,
            'plan_id': analysis.plan_id,
            'overall_risk': analysis.overall_risk_level.value,
            'risk_score': analysis.total_risk_score,
            'loops_detected': len(analysis.loop_patterns),
            'high_risk_patterns': len(high_risk_patterns),
            'issues': issues,
            'recommendations': analysis.remediation_steps,
            'estimated_max_time': analysis.estimated_max_execution_time,
            'estimated_max_iterations': analysis.estimated_max_iterations
        }
    
    def generate_loop_instrumentation_plan(self, plan_ir: Dict[str, Any],
                                         global_timeout: float = 300.0,
                                         global_max_iterations: int = 10000) -> LoopInstrumentationPlan:
        """
        Generate instrumentation plan to add time-bomb guards to loops
        
        Args:
            plan_ir: Plan intermediate representation
            global_timeout: Global timeout in seconds
            global_max_iterations: Global maximum iterations
            
        Returns:
            LoopInstrumentationPlan with instrumentation details
        """
        analysis = self.analyze_time_bomb_patterns(plan_ir)
        plan_id = plan_ir.get('plan_id', 'unknown')
        
        instrumentation_id = f"instr_{plan_id}_{hash(str(plan_ir))}_{int(datetime.now().timestamp())}"
        
        plan = LoopInstrumentationPlan(
            plan_id=plan_id,
            instrumentation_id=instrumentation_id,
            global_timeout_seconds=global_timeout,
            global_max_iterations=global_max_iterations
        )
        
        # Generate instrumentation for each loop pattern
        for loop_pattern in analysis.loop_patterns:
            # Add guard insertions
            if GuardType.ITERATION_COUNTER in loop_pattern.required_guards:
                plan.counter_insertions.append({
                    'node_id': loop_pattern.node_id,
                    'guard_type': 'iteration_counter',
                    'max_iterations': self._calculate_safe_iteration_limit(loop_pattern),
                    'action_on_violation': 'terminate'
                })
            
            if GuardType.WALL_CLOCK_TIMEOUT in loop_pattern.required_guards:
                plan.timeout_injections.append({
                    'node_id': loop_pattern.node_id,
                    'guard_type': 'wall_clock_timeout',
                    'timeout_seconds': self._calculate_safe_timeout(loop_pattern),
                    'action_on_violation': 'terminate'
                })
            
            # Add general guard insertions
            for guard_type in loop_pattern.required_guards:
                guard_config = self._generate_guard_config(loop_pattern, guard_type)
                if guard_config:
                    plan.guard_insertions.append(guard_config)
        
        self.logger.info(f"✅ Generated instrumentation plan: {len(plan.guard_insertions)} guards, {len(plan.timeout_injections)} timeouts")
        
        return plan
    
    def create_loop_guard(self, guard_type: GuardType, 
                         guard_config: Dict[str, Any]) -> LoopGuard:
        """
        Create a loop guard with specified configuration
        
        Args:
            guard_type: Type of guard to create
            guard_config: Configuration parameters
            
        Returns:
            LoopGuard instance
        """
        guard_id = f"guard_{guard_type.value}_{hash(str(guard_config))}_{int(datetime.now().timestamp())}"
        
        guard = LoopGuard(
            guard_id=guard_id,
            guard_type=guard_type,
            description=guard_config.get('description', f'{guard_type.value} guard'),
            rationale=guard_config.get('rationale', 'Prevent infinite execution')
        )
        
        # Set type-specific parameters
        if guard_type == GuardType.ITERATION_COUNTER:
            guard.max_iterations = guard_config.get('max_iterations', 10000)
        elif guard_type == GuardType.WALL_CLOCK_TIMEOUT:
            guard.timeout_seconds = guard_config.get('timeout_seconds', 300.0)
        elif guard_type == GuardType.RESOURCE_LIMIT:
            guard.max_memory_mb = guard_config.get('max_memory_mb', 512.0)
            guard.max_cpu_percent = guard_config.get('max_cpu_percent', 80.0)
        elif guard_type == GuardType.CONDITION_VALIDATOR:
            guard.termination_condition = guard_config.get('termination_condition')
            guard.progress_check = guard_config.get('progress_check')
        
        # Set violation action
        guard.action_on_violation = guard_config.get('action_on_violation', 'terminate')
        
        return guard
    
    def get_loop_safety_recommendations(self, plan_ir: Dict[str, Any]) -> List[str]:
        """
        Get safety recommendations for loops in a plan
        
        Args:
            plan_ir: Plan intermediate representation
            
        Returns:
            List of safety recommendations
        """
        analysis = self.analyze_time_bomb_patterns(plan_ir)
        
        recommendations = []
        recommendations.extend(analysis.remediation_steps)
        recommendations.extend(analysis.optimization_suggestions)
        
        # Add specific recommendations based on patterns
        for pattern in analysis.loop_patterns:
            if pattern.risk_level == LoopRisk.CRITICAL:
                recommendations.append(f"CRITICAL: Review loop in {pattern.node_id} - high risk of infinite execution")
            elif pattern.risk_level == LoopRisk.HIGH_RISK:
                recommendations.append(f"Add explicit guards to loop in {pattern.node_id}")
            
            if not pattern.termination_criteria:
                recommendations.append(f"Define clear termination criteria for loop in {pattern.node_id}")
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_timebomb_statistics(self) -> Dict[str, Any]:
        """Get time-bomb guard service statistics"""
        return {
            **self.timebomb_stats,
            'analysis_cache_size': len(self.analysis_cache),
            'guard_templates': len(self.guard_templates)
        }
    
    def _detect_loop_patterns(self, plan_ir: Dict[str, Any]) -> List[LoopPattern]:
        """Detect loop patterns in the plan IR"""
        patterns = []
        
        plan_graph = plan_ir.get('plan_graph', {})
        nodes = plan_graph.get('nodes', [])
        edges = plan_graph.get('edges', [])
        
        # Build adjacency list for cycle detection
        adjacency = defaultdict(list)
        for edge in edges:
            source = edge.get('source', edge.get('from'))
            target = edge.get('target', edge.get('to'))
            if source and target:
                adjacency[source].append(target)
        
        # Detect different types of loops
        patterns.extend(self._detect_explicit_loops(nodes))
        patterns.extend(self._detect_graph_cycles(nodes, adjacency))
        patterns.extend(self._detect_recursive_patterns(nodes, edges))
        patterns.extend(self._detect_retry_patterns(nodes))
        patterns.extend(self._detect_polling_patterns(nodes))
        
        return patterns
    
    def _detect_explicit_loops(self, nodes: List[Dict[str, Any]]) -> List[LoopPattern]:
        """Detect explicit loop constructs"""
        patterns = []
        
        for node in nodes:
            node_id = node.get('id', '')
            node_type = node.get('type', '')
            
            # Check for loop node types
            if any(keyword in node_type.lower() for keyword in ['loop', 'while', 'for', 'iterate']):
                loop_type = LoopType.FOR_LOOP if 'for' in node_type.lower() else LoopType.WHILE_LOOP
                
                pattern = LoopPattern(
                    pattern_id=f"loop_{node_id}",
                    loop_type=loop_type,
                    risk_level=LoopRisk.MEDIUM_RISK,
                    node_id=node_id,
                    node_type=node_type
                )
                
                # Analyze loop configuration
                loop_config = node.get('loop_config', {})
                pattern.loop_condition = loop_config.get('condition')
                pattern.iteration_variable = loop_config.get('iterator')
                
                # Check for termination criteria
                if loop_config.get('max_iterations'):
                    pattern.termination_criteria.append(f"max_iterations: {loop_config['max_iterations']}")
                    pattern.risk_level = LoopRisk.LOW_RISK
                
                if loop_config.get('timeout_seconds'):
                    pattern.termination_criteria.append(f"timeout: {loop_config['timeout_seconds']}s")
                    pattern.risk_level = LoopRisk.LOW_RISK
                
                # Assess risk factors
                if not pattern.termination_criteria:
                    pattern.risk_factors.append("No explicit termination criteria")
                    pattern.risk_level = LoopRisk.HIGH_RISK
                
                if pattern.loop_condition and 'true' in pattern.loop_condition.lower():
                    pattern.risk_factors.append("Potentially infinite condition")
                    pattern.unbounded_indicators.append("Condition may always be true")
                    pattern.risk_level = LoopRisk.CRITICAL
                
                # Set required guards
                if pattern.risk_level in [LoopRisk.HIGH_RISK, LoopRisk.CRITICAL]:
                    pattern.required_guards.extend([GuardType.ITERATION_COUNTER, GuardType.WALL_CLOCK_TIMEOUT])
                else:
                    pattern.recommended_guards.extend([GuardType.ITERATION_COUNTER])
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_graph_cycles(self, nodes: List[Dict[str, Any]], 
                           adjacency: Dict[str, List[str]]) -> List[LoopPattern]:
        """Detect cycles in the execution graph"""
        patterns = []
        
        # Find strongly connected components (cycles)
        node_ids = [node.get('id') for node in nodes if node.get('id')]
        cycles = self._find_strongly_connected_components(adjacency, node_ids)
        
        for cycle in cycles:
            if len(cycle) > 1:  # Actual cycle (not self-loop)
                # Create pattern for the cycle
                pattern = LoopPattern(
                    pattern_id=f"cycle_{'_'.join(cycle[:3])}",  # Use first 3 nodes for ID
                    loop_type=LoopType.CYCLE_GRAPH,
                    risk_level=LoopRisk.HIGH_RISK,
                    node_id=cycle[0],  # Representative node
                    node_type="graph_cycle"
                )
                
                pattern.risk_factors.append(f"Graph cycle involving {len(cycle)} nodes")
                pattern.unbounded_indicators.append("No explicit termination condition for cycle")
                
                # Cycles in execution graphs are inherently risky
                pattern.required_guards.extend([
                    GuardType.ITERATION_COUNTER,
                    GuardType.WALL_CLOCK_TIMEOUT,
                    GuardType.CIRCUIT_BREAKER
                ])
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_recursive_patterns(self, nodes: List[Dict[str, Any]], 
                                 edges: List[Dict[str, Any]]) -> List[LoopPattern]:
        """Detect recursive call patterns"""
        patterns = []
        
        # Look for self-referencing nodes or recursive calls
        for node in nodes:
            node_id = node.get('id', '')
            node_type = node.get('type', '')
            
            # Check for self-loops in edges
            self_referencing = any(
                edge.get('source') == node_id and edge.get('target') == node_id 
                for edge in edges
            )
            
            # Check for recursive function calls
            recursive_calls = any(
                'recursive' in str(node.get(key, '')).lower() or 'self' in str(node.get(key, '')).lower()
                for key in ['function', 'call', 'action', 'implementation']
            )
            
            if self_referencing or recursive_calls:
                pattern = LoopPattern(
                    pattern_id=f"recursive_{node_id}",
                    loop_type=LoopType.RECURSIVE_CALL,
                    risk_level=LoopRisk.HIGH_RISK,
                    node_id=node_id,
                    node_type=node_type
                )
                
                pattern.risk_factors.append("Recursive execution pattern")
                
                # Check for recursion limits
                recursion_config = node.get('recursion_config', {})
                if recursion_config.get('max_depth'):
                    pattern.termination_criteria.append(f"max_depth: {recursion_config['max_depth']}")
                    pattern.risk_level = LoopRisk.MEDIUM_RISK
                else:
                    pattern.unbounded_indicators.append("No recursion depth limit")
                
                pattern.required_guards.extend([GuardType.ITERATION_COUNTER, GuardType.RESOURCE_LIMIT])
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_retry_patterns(self, nodes: List[Dict[str, Any]]) -> List[LoopPattern]:
        """Detect retry mechanism patterns"""
        patterns = []
        
        for node in nodes:
            node_id = node.get('id', '')
            node_type = node.get('type', '')
            
            # Check for retry-related keywords
            retry_indicators = ['retry', 'repeat', 'attempt', 'backoff']
            if any(keyword in node_type.lower() for keyword in retry_indicators):
                pattern = LoopPattern(
                    pattern_id=f"retry_{node_id}",
                    loop_type=LoopType.RETRY_LOOP,
                    risk_level=LoopRisk.MEDIUM_RISK,
                    node_id=node_id,
                    node_type=node_type
                )
                
                # Check retry configuration
                retry_config = node.get('retry_config', {})
                if retry_config.get('max_attempts'):
                    pattern.termination_criteria.append(f"max_attempts: {retry_config['max_attempts']}")
                    pattern.risk_level = LoopRisk.LOW_RISK
                
                if retry_config.get('timeout'):
                    pattern.termination_criteria.append(f"timeout: {retry_config['timeout']}")
                
                if not pattern.termination_criteria:
                    pattern.risk_factors.append("No retry limits specified")
                    pattern.risk_level = LoopRisk.HIGH_RISK
                    pattern.unbounded_indicators.append("Unlimited retry attempts")
                
                pattern.recommended_guards.extend([GuardType.ITERATION_COUNTER, GuardType.CIRCUIT_BREAKER])
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_polling_patterns(self, nodes: List[Dict[str, Any]]) -> List[LoopPattern]:
        """Detect polling operation patterns"""
        patterns = []
        
        for node in nodes:
            node_id = node.get('id', '')
            node_type = node.get('type', '')
            
            # Check for polling-related keywords
            polling_indicators = ['poll', 'wait', 'check', 'monitor', 'watch']
            if any(keyword in node_type.lower() for keyword in polling_indicators):
                pattern = LoopPattern(
                    pattern_id=f"polling_{node_id}",
                    loop_type=LoopType.POLLING_LOOP,
                    risk_level=LoopRisk.MEDIUM_RISK,
                    node_id=node_id,
                    node_type=node_type
                )
                
                # Check polling configuration
                polling_config = node.get('polling_config', {})
                if polling_config.get('max_polls'):
                    pattern.termination_criteria.append(f"max_polls: {polling_config['max_polls']}")
                
                if polling_config.get('timeout'):
                    pattern.termination_criteria.append(f"timeout: {polling_config['timeout']}")
                
                if polling_config.get('interval'):
                    pattern.termination_criteria.append(f"interval: {polling_config['interval']}")
                
                if not pattern.termination_criteria:
                    pattern.risk_factors.append("No polling limits specified")
                    pattern.risk_level = LoopRisk.HIGH_RISK
                    pattern.unbounded_indicators.append("Unlimited polling")
                
                pattern.required_guards.extend([GuardType.WALL_CLOCK_TIMEOUT, GuardType.ITERATION_COUNTER])
                
                patterns.append(pattern)
        
        return patterns
    
    def _assess_overall_risk(self, patterns: List[LoopPattern]) -> Tuple[LoopRisk, float]:
        """Assess overall risk level and score"""
        if not patterns:
            return LoopRisk.SAFE, 0.0
        
        # Risk scoring
        risk_scores = {
            LoopRisk.SAFE: 0,
            LoopRisk.LOW_RISK: 1,
            LoopRisk.MEDIUM_RISK: 3,
            LoopRisk.HIGH_RISK: 7,
            LoopRisk.CRITICAL: 10
        }
        
        total_score = sum(risk_scores[pattern.risk_level] for pattern in patterns)
        max_risk = max(pattern.risk_level for pattern in patterns)
        
        # Overall risk is the maximum individual risk, but score is cumulative
        return max_risk, total_score
    
    def _analyze_guard_requirements(self, analysis: TimeBombAnalysis, plan_ir: Dict[str, Any]):
        """Analyze guard requirements for the plan"""
        # Check existing guards in the plan
        existing_guards = self._extract_existing_guards(plan_ir)
        
        # Determine missing guards
        required_guard_types = set()
        for pattern in analysis.loop_patterns:
            required_guard_types.update(pattern.required_guards)
        
        existing_guard_types = set(guard.guard_type for guard in existing_guards)
        missing_guard_types = required_guard_types - existing_guard_types
        
        analysis.missing_guards = [guard_type.value for guard_type in missing_guard_types]
        
        # Check for insufficient guards
        for pattern in analysis.loop_patterns:
            if pattern.risk_level == LoopRisk.CRITICAL and not any(
                guard.guard_type in pattern.required_guards for guard in existing_guards
            ):
                analysis.insufficient_guards.append(f"Critical pattern {pattern.pattern_id} lacks required guards")
        
        # Generate recommended guards
        for guard_type in missing_guard_types:
            guard_config = self.guard_templates.get(guard_type, {})
            guard = self.create_loop_guard(guard_type, guard_config)
            analysis.recommended_guards.append(guard)
    
    def _check_safety_violations(self, analysis: TimeBombAnalysis):
        """Check for safety violations"""
        for pattern in analysis.loop_patterns:
            if pattern.risk_level == LoopRisk.CRITICAL:
                analysis.safety_violations.append(f"Critical risk pattern: {pattern.pattern_id}")
            
            if pattern.unbounded_indicators:
                analysis.potential_infinite_loops.append(
                    f"Potential infinite loop in {pattern.node_id}: {', '.join(pattern.unbounded_indicators)}"
                )
    
    def _generate_remediation_steps(self, analysis: TimeBombAnalysis) -> List[str]:
        """Generate remediation steps"""
        steps = []
        
        if analysis.missing_guards:
            steps.append(f"Add missing guards: {', '.join(analysis.missing_guards)}")
        
        for pattern in analysis.loop_patterns:
            if pattern.risk_level in [LoopRisk.HIGH_RISK, LoopRisk.CRITICAL]:
                steps.append(f"Review and add guards to {pattern.loop_type.value} in {pattern.node_id}")
            
            if not pattern.termination_criteria:
                steps.append(f"Define explicit termination criteria for loop in {pattern.node_id}")
        
        if analysis.potential_infinite_loops:
            steps.append("Review potential infinite loops and add appropriate safeguards")
        
        return steps
    
    def _generate_optimization_suggestions(self, analysis: TimeBombAnalysis) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Count patterns by type
        pattern_counts = defaultdict(int)
        for pattern in analysis.loop_patterns:
            pattern_counts[pattern.loop_type] += 1
        
        if pattern_counts[LoopType.RETRY_LOOP] > 3:
            suggestions.append("Consider consolidating multiple retry mechanisms")
        
        if pattern_counts[LoopType.POLLING_LOOP] > 2:
            suggestions.append("Consider using event-driven patterns instead of polling")
        
        if pattern_counts[LoopType.CYCLE_GRAPH] > 0:
            suggestions.append("Review graph cycles for potential elimination or optimization")
        
        return suggestions
    
    def _estimate_max_execution_time(self, patterns: List[LoopPattern]) -> Optional[float]:
        """Estimate maximum execution time based on patterns"""
        if not patterns:
            return None
        
        max_time = 0.0
        
        for pattern in patterns:
            # Estimate time based on pattern type and configuration
            if pattern.loop_type == LoopType.POLLING_LOOP:
                # Assume polling operations take longer
                max_time += 300.0  # 5 minutes
            elif pattern.loop_type == LoopType.RETRY_LOOP:
                # Retry loops depend on retry configuration
                max_time += 60.0   # 1 minute
            else:
                # General loops
                max_time += 30.0   # 30 seconds
        
        return max_time
    
    def _estimate_max_iterations(self, patterns: List[LoopPattern]) -> Optional[int]:
        """Estimate maximum iterations based on patterns"""
        if not patterns:
            return None
        
        max_iterations = 0
        
        for pattern in patterns:
            # Extract iteration limits from termination criteria
            for criteria in pattern.termination_criteria:
                if 'max_iterations' in criteria or 'max_attempts' in criteria or 'max_polls' in criteria:
                    # Extract number from criteria string
                    import re
                    numbers = re.findall(r'\d+', criteria)
                    if numbers:
                        max_iterations += int(numbers[0])
            
            # Default estimates if no explicit limits
            if not any('max_' in criteria for criteria in pattern.termination_criteria):
                if pattern.risk_level == LoopRisk.CRITICAL:
                    max_iterations += 100000  # Very high for unbounded loops
                else:
                    max_iterations += 10000   # Default estimate
        
        return max_iterations
    
    def _extract_existing_guards(self, plan_ir: Dict[str, Any]) -> List[LoopGuard]:
        """Extract existing loop guards from plan IR"""
        guards = []
        
        # Check for guards in plan metadata
        plan_metadata = plan_ir.get('metadata', {})
        guard_configs = plan_metadata.get('loop_guards', [])
        
        for guard_config in guard_configs:
            guard_type = GuardType(guard_config.get('type', 'iteration_counter'))
            guard = self.create_loop_guard(guard_type, guard_config)
            guards.append(guard)
        
        return guards
    
    def _calculate_safe_iteration_limit(self, pattern: LoopPattern) -> int:
        """Calculate safe iteration limit for a pattern"""
        base_limit = 10000
        
        if pattern.risk_level == LoopRisk.CRITICAL:
            return 100  # Very conservative
        elif pattern.risk_level == LoopRisk.HIGH_RISK:
            return 1000
        elif pattern.risk_level == LoopRisk.MEDIUM_RISK:
            return 5000
        else:
            return base_limit
    
    def _calculate_safe_timeout(self, pattern: LoopPattern) -> float:
        """Calculate safe timeout for a pattern"""
        base_timeout = 300.0  # 5 minutes
        
        if pattern.loop_type == LoopType.POLLING_LOOP:
            return 600.0  # 10 minutes for polling
        elif pattern.loop_type == LoopType.RETRY_LOOP:
            return 120.0  # 2 minutes for retries
        else:
            return base_timeout
    
    def _generate_guard_config(self, pattern: LoopPattern, guard_type: GuardType) -> Optional[Dict[str, Any]]:
        """Generate guard configuration for a pattern"""
        config = {
            'node_id': pattern.node_id,
            'guard_type': guard_type.value,
            'pattern_id': pattern.pattern_id
        }
        
        if guard_type == GuardType.ITERATION_COUNTER:
            config['max_iterations'] = self._calculate_safe_iteration_limit(pattern)
        elif guard_type == GuardType.WALL_CLOCK_TIMEOUT:
            config['timeout_seconds'] = self._calculate_safe_timeout(pattern)
        elif guard_type == GuardType.CIRCUIT_BREAKER:
            config['failure_threshold'] = 5
            config['recovery_timeout'] = 60.0
        else:
            return None
        
        config['action_on_violation'] = 'terminate'
        
        return config
    
    def _find_strongly_connected_components(self, adjacency: Dict[str, List[str]], 
                                          node_ids: List[str]) -> List[List[str]]:
        """Find strongly connected components (cycles) using Tarjan's algorithm"""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        components = []
        
        def strongconnect(node):
            # Set the depth index for this node to the smallest unused index
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            # Consider successors of node
            for successor in adjacency.get(node, []):
                if successor not in index:
                    # Successor has not yet been visited; recurse on it
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif on_stack[successor]:
                    # Successor is in stack and hence in the current SCC
                    lowlinks[node] = min(lowlinks[node], index[successor])
            
            # If node is a root node, pop the stack and create an SCC
            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                components.append(component)
        
        for node in node_ids:
            if node and node not in index:
                strongconnect(node)
        
        # Return only components with more than 1 node (actual cycles)
        return [comp for comp in components if len(comp) > 1]
    
    def _initialize_loop_patterns(self) -> Dict[str, Any]:
        """Initialize loop detection patterns"""
        return {
            'explicit_loops': ['loop', 'while', 'for', 'iterate'],
            'retry_patterns': ['retry', 'repeat', 'attempt', 'backoff'],
            'polling_patterns': ['poll', 'wait', 'check', 'monitor', 'watch'],
            'recursive_patterns': ['recursive', 'self', 'recurse']
        }
    
    def _initialize_risk_rules(self) -> Dict[str, Any]:
        """Initialize risk assessment rules"""
        return {
            'critical_indicators': ['while true', 'infinite', 'forever'],
            'high_risk_indicators': ['no limit', 'unbounded', 'unlimited'],
            'medium_risk_indicators': ['large limit', 'high iteration'],
            'safe_indicators': ['bounded', 'limited', 'timeout']
        }
    
    def _initialize_guard_templates(self) -> Dict[GuardType, Dict[str, Any]]:
        """Initialize guard templates"""
        return {
            GuardType.ITERATION_COUNTER: {
                'max_iterations': 10000,
                'action_on_violation': 'terminate',
                'description': 'Iteration counter guard'
            },
            GuardType.WALL_CLOCK_TIMEOUT: {
                'timeout_seconds': 300.0,
                'action_on_violation': 'terminate',
                'description': 'Wall clock timeout guard'
            },
            GuardType.RESOURCE_LIMIT: {
                'max_memory_mb': 512.0,
                'max_cpu_percent': 80.0,
                'action_on_violation': 'terminate',
                'description': 'Resource limit guard'
            },
            GuardType.CIRCUIT_BREAKER: {
                'failure_threshold': 5,
                'recovery_timeout': 60.0,
                'action_on_violation': 'circuit_break',
                'description': 'Circuit breaker guard'
            }
        }
    
    def _update_timebomb_stats(self, analysis: TimeBombAnalysis):
        """Update time-bomb statistics"""
        self.timebomb_stats['total_analyses'] += 1
        self.timebomb_stats['loops_detected'] += len(analysis.loop_patterns)
        
        high_risk_patterns = len([p for p in analysis.loop_patterns if p.risk_level in [LoopRisk.HIGH_RISK, LoopRisk.CRITICAL]])
        self.timebomb_stats['high_risk_patterns'] += high_risk_patterns
        
        self.timebomb_stats['guards_recommended'] += len(analysis.recommended_guards)
        self.timebomb_stats['safety_violations'] += len(analysis.safety_violations)
        
        # Update by type and risk
        for pattern in analysis.loop_patterns:
            self.timebomb_stats['loop_types'][pattern.loop_type.value] += 1
            self.timebomb_stats['risk_levels'][pattern.risk_level.value] += 1

# API Interface
class TimeBombGuardAPI:
    """API interface for time-bomb guard operations"""
    
    def __init__(self, timebomb_service: Optional[TimeBombGuardService] = None):
        self.timebomb_service = timebomb_service or TimeBombGuardService()
    
    def analyze_time_bombs(self, plan_ir: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to analyze time-bomb patterns"""
        try:
            analysis = self.timebomb_service.analyze_time_bomb_patterns(plan_ir)
            
            return {
                'success': True,
                'analysis': asdict(analysis)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def validate_loop_safety(self, plan_ir: Dict[str, Any], strict_mode: bool = False) -> Dict[str, Any]:
        """API endpoint to validate loop safety"""
        try:
            result = self.timebomb_service.validate_loop_safety(plan_ir, strict_mode)
            
            return {
                'success': True,
                **result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_safety_recommendations(self, plan_ir: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint to get safety recommendations"""
        try:
            recommendations = self.timebomb_service.get_loop_safety_recommendations(plan_ir)
            
            return {
                'success': True,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def create_test_plan_with_loops(include_risky_patterns: bool = False) -> Dict[str, Any]:
    """Create a test plan with various loop patterns"""
    nodes = [
        {
            'id': 'start',
            'type': 'start',
            'name': 'Start Node'
        },
        {
            'id': 'for_loop',
            'type': 'for_loop',
            'name': 'Bounded For Loop',
            'loop_config': {
                'condition': 'i < 10',
                'max_iterations': 10,
                'iterator': 'i'
            }
        },
        {
            'id': 'retry_node',
            'type': 'retry_action',
            'name': 'Retry Node',
            'retry_config': {
                'max_attempts': 3,
                'timeout': 30
            }
        },
        {
            'id': 'polling_node',
            'type': 'poll_status',
            'name': 'Polling Node',
            'polling_config': {
                'interval': 5,
                'timeout': 60
            }
        }
    ]
    
    edges = [
        {'source': 'start', 'target': 'for_loop'},
        {'source': 'for_loop', 'target': 'retry_node'},
        {'source': 'retry_node', 'target': 'polling_node'}
    ]
    
    if include_risky_patterns:
        # Add risky patterns
        risky_nodes = [
            {
                'id': 'infinite_while',
                'type': 'while_loop',
                'name': 'Potentially Infinite While Loop',
                'loop_config': {
                    'condition': 'true'  # Always true - risky!
                }
            },
            {
                'id': 'unbounded_retry',
                'type': 'retry_action',
                'name': 'Unbounded Retry',
                # No retry_config - unlimited retries
            },
            {
                'id': 'recursive_node',
                'type': 'recursive_function',
                'name': 'Recursive Node'
                # No recursion_config - unlimited depth
            }
        ]
        
        nodes.extend(risky_nodes)
        
        # Add edges including a cycle
        edges.extend([
            {'source': 'polling_node', 'target': 'infinite_while'},
            {'source': 'infinite_while', 'target': 'unbounded_retry'},
            {'source': 'unbounded_retry', 'target': 'recursive_node'},
            {'source': 'recursive_node', 'target': 'recursive_node'},  # Self-loop
            {'source': 'unbounded_retry', 'target': 'polling_node'}     # Cycle
        ])
    
    return {
        'plan_id': f'test_loop_plan_{"risky" if include_risky_patterns else "safe"}',
        'plan_graph': {
            'nodes': nodes,
            'edges': edges
        },
        'metadata': {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'loop_guards': [
                {
                    'type': 'iteration_counter',
                    'max_iterations': 1000,
                    'node_id': 'for_loop'
                }
            ] if not include_risky_patterns else []
        }
    }

def run_timebomb_guard_tests():
    """Run comprehensive time-bomb guard tests"""
    print("=== Time-bomb Guard Service Tests ===")
    
    # Initialize service
    timebomb_service = TimeBombGuardService()
    timebomb_api = TimeBombGuardAPI(timebomb_service)
    
    # Test 1: Safe plan analysis
    print("\n1. Testing safe plan analysis...")
    
    safe_plan = create_test_plan_with_loops(include_risky_patterns=False)
    safe_analysis = timebomb_service.analyze_time_bomb_patterns(safe_plan)
    
    print(f"   Safe plan analysis:")
    print(f"     Overall risk: {safe_analysis.overall_risk_level.value}")
    print(f"     Risk score: {safe_analysis.total_risk_score:.1f}")
    print(f"     Loop patterns detected: {len(safe_analysis.loop_patterns)}")
    print(f"     Safety violations: {len(safe_analysis.safety_violations)}")
    
    for pattern in safe_analysis.loop_patterns:
        print(f"     - {pattern.loop_type.value} in {pattern.node_id}: {pattern.risk_level.value}")
    
    # Test 2: Risky plan analysis
    print("\n2. Testing risky plan analysis...")
    
    risky_plan = create_test_plan_with_loops(include_risky_patterns=True)
    risky_analysis = timebomb_service.analyze_time_bomb_patterns(risky_plan)
    
    print(f"   Risky plan analysis:")
    print(f"     Overall risk: {risky_analysis.overall_risk_level.value}")
    print(f"     Risk score: {risky_analysis.total_risk_score:.1f}")
    print(f"     Loop patterns detected: {len(risky_analysis.loop_patterns)}")
    print(f"     Safety violations: {len(risky_analysis.safety_violations)}")
    print(f"     Potential infinite loops: {len(risky_analysis.potential_infinite_loops)}")
    
    high_risk_patterns = [p for p in risky_analysis.loop_patterns if p.risk_level in [LoopRisk.HIGH_RISK, LoopRisk.CRITICAL]]
    print(f"     High-risk patterns: {len(high_risk_patterns)}")
    
    for pattern in high_risk_patterns:
        print(f"     - {pattern.loop_type.value} in {pattern.node_id}: {pattern.risk_level.value}")
        if pattern.unbounded_indicators:
            print(f"       Unbounded indicators: {pattern.unbounded_indicators}")
    
    # Test 3: Loop safety validation
    print("\n3. Testing loop safety validation...")
    
    # Validate safe plan
    safe_validation = timebomb_service.validate_loop_safety(safe_plan)
    print(f"   Safe plan validation: {'✅ PASS' if safe_validation['safe'] else '❌ FAIL'}")
    print(f"     Issues: {len(safe_validation['issues'])}")
    
    # Validate risky plan
    risky_validation = timebomb_service.validate_loop_safety(risky_plan, strict_mode=True)
    print(f"   Risky plan validation (strict): {'❌ FAIL' if not risky_validation['safe'] else '✅ UNEXPECTED PASS'}")
    print(f"     Issues: {len(risky_validation['issues'])}")
    for issue in risky_validation['issues'][:3]:  # Show first 3
        print(f"       - {issue}")
    
    # Test 4: Loop guard creation
    print("\n4. Testing loop guard creation...")
    
    guard_configs = [
        (GuardType.ITERATION_COUNTER, {'max_iterations': 5000, 'description': 'Test iteration guard'}),
        (GuardType.WALL_CLOCK_TIMEOUT, {'timeout_seconds': 120.0, 'description': 'Test timeout guard'}),
        (GuardType.RESOURCE_LIMIT, {'max_memory_mb': 256.0, 'max_cpu_percent': 70.0})
    ]
    
    for guard_type, config in guard_configs:
        guard = timebomb_service.create_loop_guard(guard_type, config)
        print(f"   Created {guard_type.value} guard: {guard.guard_id}")
        print(f"     Description: {guard.description}")
        if guard.max_iterations:
            print(f"     Max iterations: {guard.max_iterations}")
        if guard.timeout_seconds:
            print(f"     Timeout: {guard.timeout_seconds}s")
    
    # Test 5: Instrumentation plan generation
    print("\n5. Testing instrumentation plan generation...")
    
    instrumentation_plan = timebomb_service.generate_loop_instrumentation_plan(
        risky_plan, global_timeout=180.0, global_max_iterations=5000
    )
    
    print(f"   Instrumentation plan generated:")
    print(f"     Guard insertions: {len(instrumentation_plan.guard_insertions)}")
    print(f"     Timeout injections: {len(instrumentation_plan.timeout_injections)}")
    print(f"     Counter insertions: {len(instrumentation_plan.counter_insertions)}")
    print(f"     Global timeout: {instrumentation_plan.global_timeout_seconds}s")
    print(f"     Global max iterations: {instrumentation_plan.global_max_iterations}")
    
    for insertion in instrumentation_plan.guard_insertions[:3]:  # Show first 3
        print(f"     - {insertion['guard_type']} for {insertion['node_id']}")
    
    # Test 6: Safety recommendations
    print("\n6. Testing safety recommendations...")
    
    recommendations = timebomb_service.get_loop_safety_recommendations(risky_plan)
    print(f"   Safety recommendations: {len(recommendations)}")
    for rec in recommendations[:5]:  # Show first 5
        print(f"     - {rec}")
    
    # Test 7: Performance estimation
    print("\n7. Testing performance estimation...")
    
    print(f"   Safe plan estimates:")
    print(f"     Max execution time: {safe_analysis.estimated_max_execution_time}s")
    print(f"     Max iterations: {safe_analysis.estimated_max_iterations}")
    
    print(f"   Risky plan estimates:")
    print(f"     Max execution time: {risky_analysis.estimated_max_execution_time}s")
    print(f"     Max iterations: {risky_analysis.estimated_max_iterations}")
    
    # Test 8: API interface
    print("\n8. Testing API interface...")
    
    # Test API analysis
    api_analysis_result = timebomb_api.analyze_time_bombs(safe_plan)
    print(f"   API analysis: {'✅ PASS' if api_analysis_result['success'] else '❌ FAIL'}")
    
    # Test API validation
    api_validation_result = timebomb_api.validate_loop_safety(risky_plan, strict_mode=True)
    print(f"   API validation: {'✅ PASS' if api_validation_result['success'] else '❌ FAIL'}")
    if api_validation_result['success']:
        print(f"     Plan safety: {'✅ SAFE' if api_validation_result['safe'] else '❌ UNSAFE'}")
    
    # Test API recommendations
    api_recommendations_result = timebomb_api.get_safety_recommendations(risky_plan)
    print(f"   API recommendations: {'✅ PASS' if api_recommendations_result['success'] else '❌ FAIL'}")
    if api_recommendations_result['success']:
        print(f"     Recommendations: {len(api_recommendations_result['recommendations'])}")
    
    # Test 9: Edge cases
    print("\n9. Testing edge cases...")
    
    # Empty plan
    empty_plan = {'plan_id': 'empty', 'plan_graph': {'nodes': [], 'edges': []}}
    empty_analysis = timebomb_service.analyze_time_bomb_patterns(empty_plan)
    print(f"   Empty plan: {empty_analysis.overall_risk_level.value} risk, {len(empty_analysis.loop_patterns)} patterns")
    
    # Plan with only non-loop nodes
    simple_plan = {
        'plan_id': 'simple',
        'plan_graph': {
            'nodes': [
                {'id': 'step1', 'type': 'action', 'name': 'Simple Action'},
                {'id': 'step2', 'type': 'decision', 'name': 'Simple Decision'}
            ],
            'edges': [{'source': 'step1', 'target': 'step2'}]
        }
    }
    simple_analysis = timebomb_service.analyze_time_bomb_patterns(simple_plan)
    print(f"   Simple plan: {simple_analysis.overall_risk_level.value} risk, {len(simple_analysis.loop_patterns)} patterns")
    
    # Test 10: Statistics
    print("\n10. Testing statistics...")
    
    stats = timebomb_service.get_timebomb_statistics()
    print(f"   Total analyses: {stats['total_analyses']}")
    print(f"   Loops detected: {stats['loops_detected']}")
    print(f"   High-risk patterns: {stats['high_risk_patterns']}")
    print(f"   Guards recommended: {stats['guards_recommended']}")
    print(f"   Safety violations: {stats['safety_violations']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Loop types: {stats['loop_types']}")
    print(f"   Risk levels: {stats['risk_levels']}")
    
    print(f"\n=== Test Summary ===")
    print(f"Time-bomb guard service tested successfully")
    print(f"Total analyses: {stats['total_analyses']}")
    print(f"Loops detected: {stats['loops_detected']}")
    print(f"High-risk patterns: {stats['high_risk_patterns']}")
    print(f"Safety violations: {stats['safety_violations']}")
    print(f"Cache hit rate: {stats['cache_hits']}/{stats['cache_hits'] + stats['cache_misses']}")
    
    return timebomb_service, timebomb_api

if __name__ == "__main__":
    run_timebomb_guard_tests()



