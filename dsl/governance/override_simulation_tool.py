"""
Task 14.1-T52: Build override simulation tool (force overrides for testing)
CLI tool for simulating override scenarios for UAT and testing
"""

import asyncio
import json
import logging
import uuid
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class OverrideScenario(Enum):
    """Predefined override scenarios for testing"""
    EMERGENCY_DEAL_APPROVAL = "emergency_deal_approval"
    POLICY_BYPASS_SOX = "policy_bypass_sox"
    RBI_COMPLIANCE_EXEMPTION = "rbi_compliance_exemption"
    GDPR_ERASURE_OVERRIDE = "gdpr_erasure_override"
    HIPAA_PHI_ACCESS = "hipaa_phi_access"
    PCI_DSS_PAYMENT_OVERRIDE = "pci_dss_payment_override"
    IRDAI_CLAIMS_EMERGENCY = "irdai_claims_emergency"
    SLA_BREACH_OVERRIDE = "sla_breach_override"
    MANUAL_APPROVAL_BYPASS = "manual_approval_bypass"
    RISK_THRESHOLD_OVERRIDE = "risk_threshold_override"


class RiskLevel(Enum):
    """Risk levels for override scenarios"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OverrideTestScenario:
    """Override test scenario configuration"""
    scenario_id: str
    name: str
    description: str
    
    # Override details
    override_type: str
    component_affected: str
    risk_level: RiskLevel
    
    # Test data
    original_value: Dict[str, Any]
    overridden_value: Dict[str, Any]
    override_reason: str
    business_justification: str
    
    # Compliance context
    compliance_frameworks: List[str] = field(default_factory=list)
    industry_code: str = "SaaS"
    
    # Expected outcomes
    expected_approval_required: bool = True
    expected_escalation_chain: List[str] = field(default_factory=list)
    expected_evidence_fields: List[str] = field(default_factory=list)
    
    # Timing
    expires_in_hours: int = 24
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SimulationResult:
    """Result of override simulation"""
    scenario_id: str
    simulation_id: str
    
    # Execution details
    override_id: Optional[str] = None
    approval_id: Optional[str] = None
    
    # Status
    status: str = "pending"  # pending, success, failed, error
    
    # Validation results
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Performance metrics
    execution_time_ms: float = 0.0
    
    # Evidence generated
    evidence_pack_id: Optional[str] = None
    audit_trail_entries: int = 0
    
    # Compliance checks
    compliance_validations: Dict[str, bool] = field(default_factory=dict)
    
    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Error details
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None


class OverrideSimulationTool:
    """
    Override Simulation Tool - Task 14.1-T52
    
    CLI tool for testing override scenarios:
    - Simulates various override types and risk levels
    - Tests approval workflows and escalation chains
    - Validates evidence pack generation
    - Checks compliance framework enforcement
    - Measures performance and SLA adherence
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'simulation_timeout_seconds': 300,
            'max_concurrent_simulations': 10,
            'cleanup_test_data': True,
            'generate_reports': True,
            'validate_evidence_packs': True
        }
        
        # Predefined test scenarios
        self.test_scenarios: Dict[str, OverrideTestScenario] = {}
        self._initialize_test_scenarios()
        
        # Simulation tracking
        self.active_simulations: Dict[str, SimulationResult] = {}
        self.simulation_history: List[SimulationResult] = []
    
    def _initialize_test_scenarios(self):
        """Initialize predefined test scenarios"""
        
        # Emergency deal approval scenario
        emergency_deal = OverrideTestScenario(
            scenario_id="emergency_deal_approval",
            name="Emergency Deal Approval",
            description="Simulate emergency approval for high-value deal bypassing standard approval chain",
            override_type="emergency_approval",
            component_affected="deal_approval_workflow",
            risk_level=RiskLevel.HIGH,
            original_value={"approval_required": True, "approval_chain": ["sales_manager", "cfo", "ceo"]},
            overridden_value={"approval_required": False, "emergency_approved": True},
            override_reason="Customer threatening to cancel, immediate approval needed to save $2M deal",
            business_justification="Q4 revenue target at risk, customer has competitive offer expiring today",
            compliance_frameworks=["SOX"],
            industry_code="SaaS",
            expected_approval_required=True,
            expected_escalation_chain=["compliance_officer", "cfo"],
            expected_evidence_fields=["business_impact", "risk_assessment", "approval_chain"],
            expires_in_hours=4,
            tags=["emergency", "high_value", "revenue_critical"]
        )
        
        # SOX policy bypass scenario
        sox_bypass = OverrideTestScenario(
            scenario_id="policy_bypass_sox",
            name="SOX Policy Bypass",
            description="Test SOX compliance policy bypass for financial reporting adjustment",
            override_type="policy_bypass",
            component_affected="financial_reporting_policy",
            risk_level=RiskLevel.CRITICAL,
            original_value={"sox_controls_enabled": True, "segregation_of_duties": True},
            overridden_value={"sox_controls_enabled": False, "emergency_adjustment": True},
            override_reason="Critical accounting error discovered, immediate correction needed for earnings report",
            business_justification="SEC filing deadline in 2 hours, material misstatement must be corrected",
            compliance_frameworks=["SOX", "SEC"],
            industry_code="SaaS",
            expected_approval_required=True,
            expected_escalation_chain=["cfo", "audit_committee", "external_auditor"],
            expected_evidence_fields=["sox_justification", "auditor_approval", "ceo_sign_off"],
            expires_in_hours=2,
            tags=["sox", "critical", "financial_reporting", "sec_filing"]
        )
        
        # RBI compliance exemption scenario
        rbi_exemption = OverrideTestScenario(
            scenario_id="rbi_compliance_exemption",
            name="RBI Compliance Exemption",
            description="Test RBI regulatory exemption for urgent loan disbursal",
            override_type="regulatory_exemption",
            component_affected="rbi_loan_disbursal_policy",
            risk_level=RiskLevel.HIGH,
            original_value={"rbi_kyc_complete": False, "cooling_period_required": True},
            overridden_value={"emergency_disbursal": True, "post_disbursal_kyc": True},
            override_reason="Medical emergency loan, customer in ICU, family needs immediate funds",
            business_justification="Humanitarian grounds, customer relationship of 15 years, minimal risk",
            compliance_frameworks=["RBI", "KYC", "AML"],
            industry_code="Banking",
            expected_approval_required=True,
            expected_escalation_chain=["compliance_head", "chief_risk_officer", "md_ceo"],
            expected_evidence_fields=["medical_certificate", "customer_history", "risk_assessment"],
            expires_in_hours=1,
            tags=["rbi", "banking", "emergency", "humanitarian"]
        )
        
        # GDPR erasure override scenario
        gdpr_override = OverrideTestScenario(
            scenario_id="gdpr_erasure_override",
            name="GDPR Erasure Override",
            description="Test GDPR right-to-be-forgotten override for legal hold",
            override_type="data_retention_override",
            component_affected="gdpr_erasure_policy",
            risk_level=RiskLevel.MEDIUM,
            original_value={"erasure_requested": True, "retention_period_expired": True},
            overridden_value={"legal_hold_applied": True, "erasure_suspended": True},
            override_reason="Active litigation requires data preservation, court order received",
            business_justification="Legal obligation supersedes GDPR erasure right per Article 17(3)(e)",
            compliance_frameworks=["GDPR", "DPDP", "Legal_Hold"],
            industry_code="SaaS",
            expected_approval_required=True,
            expected_escalation_chain=["dpo", "legal_counsel"],
            expected_evidence_fields=["court_order", "legal_opinion", "dpo_approval"],
            expires_in_hours=72,
            tags=["gdpr", "privacy", "legal_hold", "litigation"]
        )
        
        # HIPAA PHI access scenario
        hipaa_access = OverrideTestScenario(
            scenario_id="hipaa_phi_access",
            name="HIPAA PHI Emergency Access",
            description="Test emergency PHI access override for patient care",
            override_type="emergency_access",
            component_affected="hipaa_phi_access_policy",
            risk_level=RiskLevel.HIGH,
            original_value={"phi_access_denied": True, "patient_consent_required": True},
            overridden_value={"emergency_access_granted": True, "life_threatening_situation": True},
            override_reason="Patient unconscious, emergency surgery required, PHI needed for treatment",
            business_justification="HIPAA emergency exception 45 CFR 164.510(a)(3), life-threatening situation",
            compliance_frameworks=["HIPAA", "Emergency_Care"],
            industry_code="Healthcare",
            expected_approval_required=True,
            expected_escalation_chain=["chief_medical_officer", "privacy_officer"],
            expected_evidence_fields=["medical_emergency_declaration", "physician_authorization"],
            expires_in_hours=6,
            tags=["hipaa", "healthcare", "emergency", "phi_access"]
        )
        
        # Store scenarios
        self.test_scenarios = {
            scenario.scenario_id: scenario for scenario in [
                emergency_deal, sox_bypass, rbi_exemption, gdpr_override, hipaa_access
            ]
        }
    
    async def run_simulation(
        self,
        scenario_id: str,
        tenant_id: int = 1300,
        user_id: int = 1323,
        custom_params: Dict[str, Any] = None
    ) -> SimulationResult:
        """Run override simulation for specified scenario"""
        
        if scenario_id not in self.test_scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = self.test_scenarios[scenario_id]
        simulation_id = str(uuid.uuid4())
        
        result = SimulationResult(
            scenario_id=scenario_id,
            simulation_id=simulation_id
        )
        
        self.active_simulations[simulation_id] = result
        
        try:
            self.logger.info(f"🧪 Starting override simulation: {scenario.name} ({simulation_id})")
            
            start_time = datetime.now(timezone.utc)
            
            # Step 1: Create override request
            override_result = await self._create_test_override(scenario, tenant_id, user_id, custom_params)
            result.override_id = override_result.get('override_id')
            result.approval_id = override_result.get('approval_id')
            
            # Step 2: Validate approval workflow
            approval_validation = await self._validate_approval_workflow(result, scenario)
            result.validation_errors.extend(approval_validation.get('errors', []))
            
            # Step 3: Check evidence pack generation
            evidence_validation = await self._validate_evidence_generation(result, scenario)
            result.evidence_pack_id = evidence_validation.get('evidence_pack_id')
            result.audit_trail_entries = evidence_validation.get('audit_entries', 0)
            
            # Step 4: Validate compliance checks
            compliance_validation = await self._validate_compliance_checks(result, scenario)
            result.compliance_validations = compliance_validation
            
            # Step 5: Performance metrics
            end_time = datetime.now(timezone.utc)
            result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
            result.completed_at = end_time
            
            # Determine overall status
            result.validation_passed = len(result.validation_errors) == 0
            result.status = "success" if result.validation_passed else "failed"
            
            self.logger.info(f"✅ Simulation completed: {scenario.name} - {result.status}")
            
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            result.completed_at = datetime.now(timezone.utc)
            self.logger.error(f"❌ Simulation failed: {scenario.name} - {e}")
        
        finally:
            # Move to history
            self.simulation_history.append(result)
            if simulation_id in self.active_simulations:
                del self.active_simulations[simulation_id]
            
            # Cleanup test data if configured
            if self.config['cleanup_test_data']:
                await self._cleanup_test_data(result)
        
        return result
    
    async def run_scenario_suite(
        self,
        scenario_ids: List[str] = None,
        tenant_id: int = 1300,
        user_id: int = 1323
    ) -> Dict[str, Any]:
        """Run multiple override scenarios as a test suite"""
        
        if scenario_ids is None:
            scenario_ids = list(self.test_scenarios.keys())
        
        suite_id = str(uuid.uuid4())
        suite_start = datetime.now(timezone.utc)
        
        self.logger.info(f"🧪 Starting override simulation suite: {len(scenario_ids)} scenarios")
        
        # Run scenarios concurrently (with limit)
        semaphore = asyncio.Semaphore(self.config['max_concurrent_simulations'])
        
        async def run_with_semaphore(scenario_id):
            async with semaphore:
                return await self.run_simulation(scenario_id, tenant_id, user_id)
        
        # Execute simulations
        tasks = [run_with_semaphore(scenario_id) for scenario_id in scenario_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_simulations = []
        failed_simulations = []
        error_simulations = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_simulations.append({
                    'scenario_id': scenario_ids[i],
                    'error': str(result)
                })
            elif result.status == "success":
                successful_simulations.append(result)
            else:
                failed_simulations.append(result)
        
        suite_end = datetime.now(timezone.utc)
        total_time_ms = (suite_end - suite_start).total_seconds() * 1000
        
        suite_result = {
            'suite_id': suite_id,
            'total_scenarios': len(scenario_ids),
            'successful': len(successful_simulations),
            'failed': len(failed_simulations),
            'errors': len(error_simulations),
            'success_rate': (len(successful_simulations) / len(scenario_ids)) * 100,
            'total_execution_time_ms': total_time_ms,
            'average_execution_time_ms': total_time_ms / len(scenario_ids),
            'results': {
                'successful': [r.__dict__ for r in successful_simulations],
                'failed': [r.__dict__ for r in failed_simulations],
                'errors': error_simulations
            },
            'started_at': suite_start.isoformat(),
            'completed_at': suite_end.isoformat()
        }
        
        self.logger.info(f"✅ Suite completed: {suite_result['success_rate']:.1f}% success rate")
        
        # Generate report if configured
        if self.config['generate_reports']:
            await self._generate_suite_report(suite_result)
        
        return suite_result
    
    async def _create_test_override(
        self,
        scenario: OverrideTestScenario,
        tenant_id: int,
        user_id: int,
        custom_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create test override request"""
        
        # In production, this would call the actual override service
        # For simulation, we create a mock override request
        
        override_id = str(uuid.uuid4())
        approval_id = str(uuid.uuid4()) if scenario.expected_approval_required else None
        
        override_data = {
            'override_id': override_id,
            'tenant_id': tenant_id,
            'user_id': user_id,
            'override_type': scenario.override_type,
            'component_affected': scenario.component_affected,
            'original_value': scenario.original_value,
            'overridden_value': scenario.overridden_value,
            'override_reason': scenario.override_reason,
            'business_justification': scenario.business_justification,
            'risk_level': scenario.risk_level.value,
            'compliance_frameworks': scenario.compliance_frameworks,
            'approval_required': scenario.expected_approval_required,
            'approval_id': approval_id,
            'expires_at': datetime.now(timezone.utc) + timedelta(hours=scenario.expires_in_hours),
            'created_at': datetime.now(timezone.utc)
        }
        
        # Apply custom parameters
        if custom_params:
            override_data.update(custom_params)
        
        # Store in database if available
        if self.db_pool:
            await self._store_test_override(override_data)
        
        self.logger.info(f"📝 Created test override: {override_id}")
        
        return {
            'override_id': override_id,
            'approval_id': approval_id,
            'status': 'created'
        }
    
    async def _validate_approval_workflow(
        self, result: SimulationResult, scenario: OverrideTestScenario
    ) -> Dict[str, Any]:
        """Validate approval workflow behavior"""
        
        validation_errors = []
        
        # Check if approval was created when expected
        if scenario.expected_approval_required and not result.approval_id:
            validation_errors.append("Expected approval workflow to be created but none found")
        
        if not scenario.expected_approval_required and result.approval_id:
            validation_errors.append("Approval workflow created when none was expected")
        
        # Validate escalation chain (mock validation)
        if result.approval_id and scenario.expected_escalation_chain:
            # In production, this would query the actual approval chain
            expected_chain = set(scenario.expected_escalation_chain)
            actual_chain = set(["compliance_officer", "cfo"])  # Mock data
            
            if not expected_chain.issubset(actual_chain):
                missing_approvers = expected_chain - actual_chain
                validation_errors.append(f"Missing expected approvers: {missing_approvers}")
        
        return {
            'errors': validation_errors,
            'approval_workflow_valid': len(validation_errors) == 0
        }
    
    async def _validate_evidence_generation(
        self, result: SimulationResult, scenario: OverrideTestScenario
    ) -> Dict[str, Any]:
        """Validate evidence pack generation"""
        
        # Mock evidence pack generation
        evidence_pack_id = str(uuid.uuid4())
        audit_entries = len(scenario.expected_evidence_fields) + 2  # Base entries + evidence fields
        
        # In production, this would validate actual evidence pack content
        evidence_validation = {
            'evidence_pack_id': evidence_pack_id,
            'audit_entries': audit_entries,
            'evidence_fields_captured': scenario.expected_evidence_fields,
            'evidence_complete': True
        }
        
        return evidence_validation
    
    async def _validate_compliance_checks(
        self, result: SimulationResult, scenario: OverrideTestScenario
    ) -> Dict[str, bool]:
        """Validate compliance framework checks"""
        
        compliance_results = {}
        
        for framework in scenario.compliance_frameworks:
            # Mock compliance validation
            if framework == "SOX":
                compliance_results[framework] = scenario.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            elif framework == "GDPR":
                compliance_results[framework] = "legal_hold" in scenario.tags
            elif framework == "RBI":
                compliance_results[framework] = scenario.industry_code == "Banking"
            elif framework == "HIPAA":
                compliance_results[framework] = scenario.industry_code == "Healthcare"
            else:
                compliance_results[framework] = True  # Default pass
        
        return compliance_results
    
    async def _store_test_override(self, override_data: Dict[str, Any]):
        """Store test override in database"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO override_ledger (
                        override_id, tenant_id, user_id, override_type,
                        component_affected, original_value, overridden_value,
                        override_reason, business_impact, risk_level,
                        approval_required, expires_at, status, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                override_data['override_id'],
                override_data['tenant_id'],
                override_data['user_id'],
                override_data['override_type'],
                override_data['component_affected'],
                json.dumps(override_data['original_value']),
                json.dumps(override_data['overridden_value']),
                override_data['override_reason'],
                override_data['business_justification'],
                override_data['risk_level'],
                override_data['approval_required'],
                override_data['expires_at'],
                'active',
                json.dumps({
                    'simulation': True,
                    'scenario_id': override_data.get('scenario_id'),
                    'compliance_frameworks': override_data['compliance_frameworks']
                }))
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store test override: {e}")
    
    async def _cleanup_test_data(self, result: SimulationResult):
        """Cleanup test data after simulation"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Delete test override
                if result.override_id:
                    await conn.execute(
                        "DELETE FROM override_ledger WHERE override_id = $1",
                        result.override_id
                    )
                
                # Delete test approval
                if result.approval_id:
                    await conn.execute(
                        "DELETE FROM approval_ledger WHERE approval_id = $1",
                        result.approval_id
                    )
                
                # Delete test evidence pack
                if result.evidence_pack_id:
                    await conn.execute(
                        "DELETE FROM evidence_packs WHERE evidence_pack_id = $1",
                        result.evidence_pack_id
                    )
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup test data: {e}")
    
    async def _generate_suite_report(self, suite_result: Dict[str, Any]):
        """Generate simulation suite report"""
        
        report_content = f"""
# Override Simulation Suite Report

**Suite ID:** {suite_result['suite_id']}
**Execution Time:** {suite_result['total_execution_time_ms']:.0f}ms
**Success Rate:** {suite_result['success_rate']:.1f}%

## Summary
- Total Scenarios: {suite_result['total_scenarios']}
- Successful: {suite_result['successful']}
- Failed: {suite_result['failed']}
- Errors: {suite_result['errors']}

## Performance Metrics
- Average Execution Time: {suite_result['average_execution_time_ms']:.0f}ms
- Started: {suite_result['started_at']}
- Completed: {suite_result['completed_at']}

## Detailed Results

### Successful Simulations
"""
        
        for result in suite_result['results']['successful']:
            report_content += f"""
- **{result['scenario_id']}**: {result['execution_time_ms']:.0f}ms
  - Override ID: {result['override_id']}
  - Evidence Pack: {result['evidence_pack_id']}
  - Audit Entries: {result['audit_trail_entries']}
"""
        
        if suite_result['results']['failed']:
            report_content += "\n### Failed Simulations\n"
            for result in suite_result['results']['failed']:
                report_content += f"""
- **{result['scenario_id']}**: {len(result['validation_errors'])} errors
  - Errors: {', '.join(result['validation_errors'])}
"""
        
        if suite_result['results']['errors']:
            report_content += "\n### Error Simulations\n"
            for result in suite_result['results']['errors']:
                report_content += f"- **{result['scenario_id']}**: {result['error']}\n"
        
        # Save report
        report_filename = f"override_simulation_report_{suite_result['suite_id'][:8]}.md"
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📊 Generated simulation report: {report_filename}")
    
    def list_scenarios(self) -> Dict[str, Any]:
        """List available test scenarios"""
        
        scenarios_info = {}
        
        for scenario_id, scenario in self.test_scenarios.items():
            scenarios_info[scenario_id] = {
                'name': scenario.name,
                'description': scenario.description,
                'risk_level': scenario.risk_level.value,
                'industry_code': scenario.industry_code,
                'compliance_frameworks': scenario.compliance_frameworks,
                'tags': scenario.tags,
                'expected_approval_required': scenario.expected_approval_required,
                'expires_in_hours': scenario.expires_in_hours
            }
        
        return scenarios_info
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        
        total_simulations = len(self.simulation_history)
        successful_simulations = len([r for r in self.simulation_history if r.status == "success"])
        
        avg_execution_time = 0
        if self.simulation_history:
            avg_execution_time = sum(r.execution_time_ms for r in self.simulation_history) / total_simulations
        
        return {
            'total_simulations_run': total_simulations,
            'successful_simulations': successful_simulations,
            'success_rate': (successful_simulations / total_simulations * 100) if total_simulations > 0 else 0,
            'average_execution_time_ms': avg_execution_time,
            'active_simulations': len(self.active_simulations),
            'available_scenarios': len(self.test_scenarios),
            'last_simulation': self.simulation_history[-1].started_at.isoformat() if self.simulation_history else None
        }


# CLI Interface
async def main():
    """CLI interface for override simulation tool"""
    
    parser = argparse.ArgumentParser(description="Override Simulation Tool")
    parser.add_argument('--scenario', type=str, help='Scenario ID to run')
    parser.add_argument('--suite', action='store_true', help='Run full test suite')
    parser.add_argument('--list', action='store_true', help='List available scenarios')
    parser.add_argument('--tenant-id', type=int, default=1300, help='Tenant ID for testing')
    parser.add_argument('--user-id', type=int, default=1323, help='User ID for testing')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    # Initialize simulation tool
    tool = OverrideSimulationTool()
    await tool.initialize()
    
    if args.list:
        scenarios = tool.list_scenarios()
        print("\n🧪 Available Override Scenarios:")
        print("=" * 50)
        for scenario_id, info in scenarios.items():
            print(f"\n📋 {scenario_id}")
            print(f"   Name: {info['name']}")
            print(f"   Risk: {info['risk_level']}")
            print(f"   Industry: {info['industry_code']}")
            print(f"   Frameworks: {', '.join(info['compliance_frameworks'])}")
            print(f"   Description: {info['description']}")
        return
    
    if args.suite:
        print("🧪 Running override simulation suite...")
        result = await tool.run_scenario_suite(tenant_id=args.tenant_id, user_id=args.user_id)
        
        print(f"\n✅ Suite completed:")
        print(f"   Success Rate: {result['success_rate']:.1f}%")
        print(f"   Total Time: {result['total_execution_time_ms']:.0f}ms")
        print(f"   Successful: {result['successful']}/{result['total_scenarios']}")
        
        if result['failed'] > 0:
            print(f"   Failed: {result['failed']}")
        if result['errors'] > 0:
            print(f"   Errors: {result['errors']}")
        
        return
    
    if args.scenario:
        print(f"🧪 Running scenario: {args.scenario}")
        result = await tool.run_simulation(args.scenario, args.tenant_id, args.user_id)
        
        print(f"\n✅ Simulation completed:")
        print(f"   Status: {result.status}")
        print(f"   Execution Time: {result.execution_time_ms:.0f}ms")
        print(f"   Override ID: {result.override_id}")
        print(f"   Evidence Pack: {result.evidence_pack_id}")
        
        if result.validation_errors:
            print(f"   Validation Errors: {len(result.validation_errors)}")
            for error in result.validation_errors:
                print(f"     - {error}")
        
        return
    
    # Show statistics if no specific action
    stats = tool.get_simulation_statistics()
    print("\n📊 Override Simulation Statistics:")
    print("=" * 40)
    print(f"Total Simulations: {stats['total_simulations_run']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Average Time: {stats['average_execution_time_ms']:.0f}ms")
    print(f"Available Scenarios: {stats['available_scenarios']}")


if __name__ == "__main__":
    asyncio.run(main())
