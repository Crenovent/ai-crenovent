# Role-Based Access Control (RBAC) and Segregation of Duties (SoD) System
# Tasks 8.3-T01 to T35: RBAC definitions, SoD enforcement, ABAC integration

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)

class RoleLevel(Enum):
    """Role hierarchy levels"""
    EXECUTIVE = "executive"
    MANAGER = "manager"
    USER = "user"
    SYSTEM = "system"

class Permission(Enum):
    """System permissions"""
    # Workflow permissions
    WORKFLOW_CREATE = "workflow_create"
    WORKFLOW_READ = "workflow_read"
    WORKFLOW_UPDATE = "workflow_update"
    WORKFLOW_DELETE = "workflow_delete"
    WORKFLOW_EXECUTE = "workflow_execute"
    WORKFLOW_APPROVE = "workflow_approve"
    WORKFLOW_PUBLISH = "workflow_publish"
    
    # Policy permissions
    POLICY_CREATE = "policy_create"
    POLICY_READ = "policy_read"
    POLICY_UPDATE = "policy_update"
    POLICY_DELETE = "policy_delete"
    POLICY_ENFORCE = "policy_enforce"
    
    # Override permissions
    OVERRIDE_REQUEST = "override_request"
    OVERRIDE_APPROVE = "override_approve"
    OVERRIDE_REVOKE = "override_revoke"
    
    # Audit permissions
    AUDIT_READ = "audit_read"
    AUDIT_EXPORT = "audit_export"
    
    # Admin permissions
    USER_MANAGE = "user_manage"
    ROLE_MANAGE = "role_manage"
    TENANT_MANAGE = "tenant_manage"
    
    # Compliance permissions
    COMPLIANCE_READ = "compliance_read"
    COMPLIANCE_REPORT = "compliance_report"
    EVIDENCE_ACCESS = "evidence_access"

class AccessContext(Enum):
    """Access context attributes for ABAC"""
    TENANT = "tenant"
    REGION = "region"
    INDUSTRY = "industry"
    SENSITIVITY = "sensitivity"
    TIME_OF_DAY = "time_of_day"
    IP_RANGE = "ip_range"
    RISK_SCORE = "risk_score"

@dataclass
class Role:
    """Role definition with permissions and hierarchy"""
    role_id: str
    role_name: str
    role_level: RoleLevel
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: List[str] = field(default_factory=list)
    child_roles: List[str] = field(default_factory=list)
    
    # Metadata
    description: str = ""
    created_by: str = "system"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True
    
    # Industry and compliance
    industry_specific: bool = False
    compliance_frameworks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['permissions'] = [p.value for p in self.permissions]
        return data
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in self.permissions
    
    def add_permission(self, permission: Permission) -> None:
        """Add permission to role"""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission) -> None:
        """Remove permission from role"""
        self.permissions.discard(permission)

@dataclass
class UserRole:
    """User role assignment with context"""
    user_id: str
    role_id: str
    tenant_id: int
    
    # Context attributes for ABAC
    region: Optional[str] = None
    industry: Optional[str] = None
    department: Optional[str] = None
    
    # Temporal constraints
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    
    # Assignment metadata
    assigned_by: str = "system"
    assigned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def is_valid_now(self) -> bool:
        """Check if role assignment is currently valid"""
        if not self.is_active:
            return False
        
        now = datetime.now(timezone.utc)
        
        if self.valid_from:
            valid_from = datetime.fromisoformat(self.valid_from.replace('Z', '+00:00'))
            if now < valid_from:
                return False
        
        if self.valid_until:
            valid_until = datetime.fromisoformat(self.valid_until.replace('Z', '+00:00'))
            if now > valid_until:
                return False
        
        return True

@dataclass
class SoDRule:
    """Segregation of Duties rule definition"""
    rule_id: str
    rule_name: str
    description: str
    
    # Conflicting permissions/roles
    conflicting_permissions: List[Permission] = field(default_factory=list)
    conflicting_roles: List[str] = field(default_factory=list)
    
    # Scope
    applies_to_workflows: List[str] = field(default_factory=list)  # Workflow patterns
    applies_to_tenants: List[int] = field(default_factory=list)
    
    # Enforcement
    enforcement_level: str = "strict"  # strict, warning, monitor
    requires_approval: bool = True
    approval_roles: List[str] = field(default_factory=list)
    
    # Compliance
    compliance_frameworks: List[str] = field(default_factory=list)
    
    # Metadata
    created_by: str = "system"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['conflicting_permissions'] = [p.value for p in self.conflicting_permissions]
        return data

@dataclass
class AccessRequest:
    """Access request for audit and approval"""
    request_id: str
    user_id: str
    tenant_id: int
    requested_permission: Permission
    resource_id: str
    
    # Context
    request_context: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    
    # Status
    status: str = "pending"  # pending, approved, denied, expired
    requested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Approval
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    approval_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['requested_permission'] = self.requested_permission.value
        return data

class RBACManager:
    """
    Role-Based Access Control Manager
    Tasks 8.3-T01 to T15: Role catalog, hierarchies, RBAC schema
    """
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, List[UserRole]] = {}  # user_id -> roles
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self) -> None:
        """Initialize default role catalog"""
        
        # Executive roles
        cro_role = Role(
            role_id="cro",
            role_name="Chief Revenue Officer",
            role_level=RoleLevel.EXECUTIVE,
            permissions={
                Permission.WORKFLOW_READ, Permission.WORKFLOW_APPROVE, Permission.WORKFLOW_PUBLISH,
                Permission.POLICY_READ, Permission.OVERRIDE_APPROVE,
                Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
                Permission.COMPLIANCE_READ, Permission.COMPLIANCE_REPORT, Permission.EVIDENCE_ACCESS
            },
            description="Chief Revenue Officer with executive oversight",
            compliance_frameworks=["SOX", "GDPR"]
        )
        
        cfo_role = Role(
            role_id="cfo",
            role_name="Chief Financial Officer",
            role_level=RoleLevel.EXECUTIVE,
            permissions={
                Permission.WORKFLOW_READ, Permission.WORKFLOW_APPROVE,
                Permission.POLICY_READ, Permission.OVERRIDE_APPROVE,
                Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
                Permission.COMPLIANCE_READ, Permission.COMPLIANCE_REPORT, Permission.EVIDENCE_ACCESS
            },
            description="Chief Financial Officer with financial oversight",
            compliance_frameworks=["SOX", "GAAP"]
        )
        
        # Manager roles
        revops_manager_role = Role(
            role_id="revops_manager",
            role_name="Revenue Operations Manager",
            role_level=RoleLevel.MANAGER,
            permissions={
                Permission.WORKFLOW_CREATE, Permission.WORKFLOW_READ, Permission.WORKFLOW_UPDATE,
                Permission.WORKFLOW_EXECUTE, Permission.POLICY_READ,
                Permission.OVERRIDE_REQUEST, Permission.AUDIT_READ
            },
            description="Revenue Operations Manager with workflow management",
            parent_roles=["cro"]
        )
        
        compliance_manager_role = Role(
            role_id="compliance_manager",
            role_name="Compliance Manager",
            role_level=RoleLevel.MANAGER,
            permissions={
                Permission.WORKFLOW_READ, Permission.WORKFLOW_APPROVE,
                Permission.POLICY_CREATE, Permission.POLICY_READ, Permission.POLICY_UPDATE,
                Permission.OVERRIDE_APPROVE, Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
                Permission.COMPLIANCE_READ, Permission.COMPLIANCE_REPORT, Permission.EVIDENCE_ACCESS
            },
            description="Compliance Manager with governance oversight",
            compliance_frameworks=["SOX", "GDPR", "HIPAA", "RBI", "IRDAI"]
        )
        
        ops_manager_role = Role(
            role_id="ops_manager",
            role_name="Operations Manager",
            role_level=RoleLevel.MANAGER,
            permissions={
                Permission.WORKFLOW_READ, Permission.WORKFLOW_EXECUTE,
                Permission.POLICY_READ, Permission.POLICY_ENFORCE,
                Permission.OVERRIDE_REQUEST, Permission.AUDIT_READ
            },
            description="Operations Manager with execution oversight"
        )
        
        # User roles
        account_executive_role = Role(
            role_id="account_executive",
            role_name="Account Executive",
            role_level=RoleLevel.USER,
            permissions={
                Permission.WORKFLOW_READ, Permission.WORKFLOW_EXECUTE,
                Permission.OVERRIDE_REQUEST
            },
            description="Account Executive with limited workflow access",
            parent_roles=["revops_manager"]
        )
        
        developer_role = Role(
            role_id="developer",
            role_name="Developer",
            role_level=RoleLevel.USER,
            permissions={
                Permission.WORKFLOW_CREATE, Permission.WORKFLOW_READ, Permission.WORKFLOW_UPDATE,
                Permission.POLICY_READ, Permission.AUDIT_READ
            },
            description="Developer with workflow development access"
        )
        
        # System roles
        system_role = Role(
            role_id="system",
            role_name="System",
            role_level=RoleLevel.SYSTEM,
            permissions=set(Permission),  # All permissions
            description="System role with full access"
        )
        
        # Store roles
        roles = [cro_role, cfo_role, revops_manager_role, compliance_manager_role, 
                ops_manager_role, account_executive_role, developer_role, system_role]
        
        for role in roles:
            self.roles[role.role_id] = role
        
        # Set up role hierarchies
        self._setup_role_hierarchies()
        
        logger.info(f"✅ Initialized {len(self.roles)} default roles")
    
    def _setup_role_hierarchies(self) -> None:
        """Set up role inheritance hierarchies"""
        
        # CRO can delegate to RevOps Manager
        if "cro" in self.roles and "revops_manager" in self.roles:
            self.roles["cro"].child_roles.append("revops_manager")
            self.roles["revops_manager"].parent_roles.append("cro")
        
        # RevOps Manager can delegate to Account Executive
        if "revops_manager" in self.roles and "account_executive" in self.roles:
            self.roles["revops_manager"].child_roles.append("account_executive")
            self.roles["account_executive"].parent_roles.append("revops_manager")
    
    def create_role(self, role: Role) -> bool:
        """Create a new role"""
        try:
            if role.role_id in self.roles:
                logger.warning(f"⚠️ Role {role.role_id} already exists")
                return False
            
            self.roles[role.role_id] = role
            logger.info(f"✅ Created role: {role.role_id} ({role.role_name})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create role {role.role_id}: {e}")
            return False
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID"""
        return self.roles.get(role_id)
    
    def assign_role_to_user(self, user_role: UserRole) -> bool:
        """Assign role to user"""
        try:
            if user_role.role_id not in self.roles:
                logger.error(f"❌ Role {user_role.role_id} does not exist")
                return False
            
            if user_role.user_id not in self.user_roles:
                self.user_roles[user_role.user_id] = []
            
            # Check if user already has this role for this tenant
            existing_role = next(
                (ur for ur in self.user_roles[user_role.user_id] 
                 if ur.role_id == user_role.role_id and ur.tenant_id == user_role.tenant_id),
                None
            )
            
            if existing_role:
                logger.warning(f"⚠️ User {user_role.user_id} already has role {user_role.role_id} for tenant {user_role.tenant_id}")
                return False
            
            self.user_roles[user_role.user_id].append(user_role)
            logger.info(f"✅ Assigned role {user_role.role_id} to user {user_role.user_id} for tenant {user_role.tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to assign role: {e}")
            return False
    
    def get_user_roles(self, user_id: str, tenant_id: Optional[int] = None) -> List[UserRole]:
        """Get roles for user, optionally filtered by tenant"""
        
        user_roles = self.user_roles.get(user_id, [])
        
        if tenant_id:
            user_roles = [ur for ur in user_roles if ur.tenant_id == tenant_id]
        
        # Filter to only active and valid roles
        valid_roles = [ur for ur in user_roles if ur.is_valid_now()]
        
        return valid_roles
    
    def get_user_permissions(self, user_id: str, tenant_id: int) -> Set[Permission]:
        """Get all permissions for user in tenant (including inherited)"""
        
        user_roles = self.get_user_roles(user_id, tenant_id)
        all_permissions = set()
        
        for user_role in user_roles:
            role = self.roles.get(user_role.role_id)
            if role:
                all_permissions.update(role.permissions)
                
                # Add inherited permissions from parent roles
                inherited_permissions = self._get_inherited_permissions(role.role_id)
                all_permissions.update(inherited_permissions)
        
        return all_permissions
    
    def _get_inherited_permissions(self, role_id: str) -> Set[Permission]:
        """Get permissions inherited from parent roles"""
        
        inherited_permissions = set()
        role = self.roles.get(role_id)
        
        if role:
            for parent_role_id in role.parent_roles:
                parent_role = self.roles.get(parent_role_id)
                if parent_role:
                    inherited_permissions.update(parent_role.permissions)
                    # Recursively get permissions from grandparent roles
                    inherited_permissions.update(self._get_inherited_permissions(parent_role_id))
        
        return inherited_permissions
    
    def has_permission(self, user_id: str, tenant_id: int, permission: Permission) -> bool:
        """Check if user has specific permission in tenant"""
        
        user_permissions = self.get_user_permissions(user_id, tenant_id)
        return permission in user_permissions
    
    def list_roles(self, role_level: Optional[RoleLevel] = None, active_only: bool = True) -> List[Role]:
        """List roles with optional filtering"""
        
        roles = list(self.roles.values())
        
        if role_level:
            roles = [r for r in roles if r.role_level == role_level]
        
        if active_only:
            roles = [r for r in roles if r.is_active]
        
        return roles
    
    def get_role_statistics(self) -> Dict[str, Any]:
        """Get RBAC statistics"""
        
        total_roles = len(self.roles)
        active_roles = len([r for r in self.roles.values() if r.is_active])
        total_users = len(self.user_roles)
        total_assignments = sum(len(roles) for roles in self.user_roles.values())
        
        roles_by_level = {}
        for role in self.roles.values():
            level = role.role_level.value
            roles_by_level[level] = roles_by_level.get(level, 0) + 1
        
        return {
            'total_roles': total_roles,
            'active_roles': active_roles,
            'total_users': total_users,
            'total_role_assignments': total_assignments,
            'roles_by_level': roles_by_level
        }

class SoDEnforcer:
    """
    Segregation of Duties Enforcer
    Tasks 8.3-T16 to T25: SoD matrices, maker-checker rules
    """
    
    def __init__(self, rbac_manager: RBACManager):
        self.rbac_manager = rbac_manager
        self.sod_rules: Dict[str, SoDRule] = {}
        self.access_requests: Dict[str, AccessRequest] = {}
        
        # Initialize default SoD rules
        self._initialize_default_sod_rules()
    
    def _initialize_default_sod_rules(self) -> None:
        """Initialize default SoD rules"""
        
        # Workflow creation and approval separation
        workflow_sod_rule = SoDRule(
            rule_id="workflow_create_approve_separation",
            rule_name="Workflow Creation and Approval Separation",
            description="The same person cannot create and approve a workflow",
            conflicting_permissions=[Permission.WORKFLOW_CREATE, Permission.WORKFLOW_APPROVE],
            enforcement_level="strict",
            requires_approval=True,
            approval_roles=["compliance_manager", "cro", "cfo"],
            compliance_frameworks=["SOX", "GDPR"]
        )
        
        # Policy creation and enforcement separation
        policy_sod_rule = SoDRule(
            rule_id="policy_create_enforce_separation",
            rule_name="Policy Creation and Enforcement Separation",
            description="The same person cannot create and enforce policies",
            conflicting_permissions=[Permission.POLICY_CREATE, Permission.POLICY_ENFORCE],
            enforcement_level="strict",
            requires_approval=True,
            approval_roles=["compliance_manager"],
            compliance_frameworks=["SOX", "RBI", "IRDAI"]
        )
        
        # Override request and approval separation
        override_sod_rule = SoDRule(
            rule_id="override_request_approve_separation",
            rule_name="Override Request and Approval Separation",
            description="The same person cannot request and approve overrides",
            conflicting_permissions=[Permission.OVERRIDE_REQUEST, Permission.OVERRIDE_APPROVE],
            enforcement_level="strict",
            requires_approval=True,
            approval_roles=["compliance_manager", "cro", "cfo"],
            compliance_frameworks=["SOX", "GDPR", "RBI", "IRDAI"]
        )
        
        # Financial workflow separation (for SOX compliance)
        financial_sod_rule = SoDRule(
            rule_id="financial_workflow_separation",
            rule_name="Financial Workflow Separation",
            description="Financial workflows require separate creation, approval, and execution",
            conflicting_roles=["cfo", "revops_manager"],  # CFO cannot also be RevOps Manager
            applies_to_workflows=["financial_*", "revenue_*", "billing_*"],
            enforcement_level="strict",
            requires_approval=True,
            approval_roles=["compliance_manager"],
            compliance_frameworks=["SOX", "GAAP"]
        )
        
        # Store rules
        rules = [workflow_sod_rule, policy_sod_rule, override_sod_rule, financial_sod_rule]
        
        for rule in rules:
            self.sod_rules[rule.rule_id] = rule
        
        logger.info(f"✅ Initialized {len(self.sod_rules)} default SoD rules")
    
    def create_sod_rule(self, rule: SoDRule) -> bool:
        """Create a new SoD rule"""
        try:
            if rule.rule_id in self.sod_rules:
                logger.warning(f"⚠️ SoD rule {rule.rule_id} already exists")
                return False
            
            self.sod_rules[rule.rule_id] = rule
            logger.info(f"✅ Created SoD rule: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create SoD rule {rule.rule_id}: {e}")
            return False
    
    async def check_sod_compliance(
        self,
        user_id: str,
        tenant_id: int,
        requested_permission: Permission,
        resource_id: str,
        workflow_id: Optional[str] = None
    ) -> Tuple[bool, List[str], List[SoDRule]]:
        """
        Check SoD compliance for user action
        Returns: (is_compliant, violations, applicable_rules)
        """
        
        violations = []
        applicable_rules = []
        
        # Get user's current permissions
        user_permissions = self.rbac_manager.get_user_permissions(user_id, tenant_id)
        user_roles = [ur.role_id for ur in self.rbac_manager.get_user_roles(user_id, tenant_id)]
        
        # Check each SoD rule
        for rule in self.sod_rules.values():
            if not rule.is_active:
                continue
            
            # Check if rule applies to this tenant
            if rule.applies_to_tenants and tenant_id not in rule.applies_to_tenants:
                continue
            
            # Check if rule applies to this workflow
            if rule.applies_to_workflows and workflow_id:
                workflow_matches = any(
                    workflow_id.startswith(pattern.replace('*', ''))
                    for pattern in rule.applies_to_workflows
                )
                if not workflow_matches:
                    continue
            
            applicable_rules.append(rule)
            
            # Check permission conflicts
            if rule.conflicting_permissions:
                conflicting_perms = set(rule.conflicting_permissions)
                if requested_permission in conflicting_perms:
                    # Check if user already has any of the other conflicting permissions
                    other_conflicting_perms = conflicting_perms - {requested_permission}
                    if any(perm in user_permissions for perm in other_conflicting_perms):
                        violations.append(
                            f"SoD violation: User has conflicting permissions for rule '{rule.rule_name}'"
                        )
            
            # Check role conflicts
            if rule.conflicting_roles:
                user_conflicting_roles = set(user_roles) & set(rule.conflicting_roles)
                if len(user_conflicting_roles) > 1:
                    violations.append(
                        f"SoD violation: User has conflicting roles {user_conflicting_roles} for rule '{rule.rule_name}'"
                    )
        
        is_compliant = len(violations) == 0
        
        if not is_compliant:
            logger.warning(f"🚨 SoD violations detected for user {user_id}: {violations}")
        
        return is_compliant, violations, applicable_rules
    
    async def request_access_with_approval(
        self,
        user_id: str,
        tenant_id: int,
        requested_permission: Permission,
        resource_id: str,
        justification: str,
        context: Dict[str, Any] = None
    ) -> AccessRequest:
        """Request access that requires approval due to SoD"""
        
        request = AccessRequest(
            request_id=str(uuid.uuid4()),
            user_id=user_id,
            tenant_id=tenant_id,
            requested_permission=requested_permission,
            resource_id=resource_id,
            request_context=context or {},
            justification=justification
        )
        
        self.access_requests[request.request_id] = request
        
        logger.info(f"📋 Access request created: {request.request_id} for user {user_id}")
        
        return request
    
    async def approve_access_request(
        self,
        request_id: str,
        approver_id: str,
        approval_notes: str = ""
    ) -> bool:
        """Approve access request"""
        
        request = self.access_requests.get(request_id)
        if not request:
            logger.error(f"❌ Access request {request_id} not found")
            return False
        
        if request.status != "pending":
            logger.error(f"❌ Access request {request_id} is not pending (status: {request.status})")
            return False
        
        # Check if approver has permission to approve
        approver_permissions = self.rbac_manager.get_user_permissions(approver_id, request.tenant_id)
        if Permission.OVERRIDE_APPROVE not in approver_permissions:
            logger.error(f"❌ Approver {approver_id} does not have approval permission")
            return False
        
        # Update request
        request.status = "approved"
        request.approved_by = approver_id
        request.approved_at = datetime.now(timezone.utc).isoformat()
        request.approval_notes = approval_notes
        
        logger.info(f"✅ Access request {request_id} approved by {approver_id}")
        
        return True
    
    def get_pending_requests(self, tenant_id: Optional[int] = None) -> List[AccessRequest]:
        """Get pending access requests"""
        
        requests = [r for r in self.access_requests.values() if r.status == "pending"]
        
        if tenant_id:
            requests = [r for r in requests if r.tenant_id == tenant_id]
        
        # Sort by most recent first
        requests.sort(key=lambda r: r.requested_at, reverse=True)
        
        return requests
    
    def get_sod_statistics(self) -> Dict[str, Any]:
        """Get SoD statistics"""
        
        total_rules = len(self.sod_rules)
        active_rules = len([r for r in self.sod_rules.values() if r.is_active])
        total_requests = len(self.access_requests)
        
        requests_by_status = {}
        for request in self.access_requests.values():
            status = request.status
            requests_by_status[status] = requests_by_status.get(status, 0) + 1
        
        return {
            'total_sod_rules': total_rules,
            'active_sod_rules': active_rules,
            'total_access_requests': total_requests,
            'requests_by_status': requests_by_status
        }

class ABACManager:
    """
    Attribute-Based Access Control Manager
    Tasks 8.3-T04, T26 to T35: ABAC attributes, contextual controls
    """
    
    def __init__(self):
        self.abac_policies: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_abac_policies()
    
    def _initialize_default_abac_policies(self) -> None:
        """Initialize default ABAC policies"""
        
        # Regional access policy
        regional_policy = {
            'policy_id': 'regional_access_control',
            'name': 'Regional Access Control',
            'description': 'Control access based on user and data region',
            'rules': [
                {
                    'condition': 'user.region == data.region',
                    'action': 'allow',
                    'description': 'Allow access when user and data are in same region'
                },
                {
                    'condition': 'user.region == "US" and data.region == "EU" and user.has_gdpr_training == true',
                    'action': 'allow',
                    'description': 'Allow US users to access EU data if GDPR trained'
                },
                {
                    'condition': 'data.sensitivity == "high" and user.clearance_level < 3',
                    'action': 'deny',
                    'description': 'Deny access to high sensitivity data for low clearance users'
                }
            ],
            'compliance_frameworks': ['GDPR', 'DPDP', 'CCPA']
        }
        
        # Time-based access policy
        time_based_policy = {
            'policy_id': 'time_based_access_control',
            'name': 'Time-based Access Control',
            'description': 'Control access based on time of day and business hours',
            'rules': [
                {
                    'condition': 'time.hour >= 9 and time.hour <= 17 and time.weekday < 5',
                    'action': 'allow',
                    'description': 'Allow access during business hours (9 AM - 5 PM, weekdays)'
                },
                {
                    'condition': 'user.role == "system" or user.role == "ops_manager"',
                    'action': 'allow',
                    'description': 'Always allow system and ops manager access'
                },
                {
                    'condition': 'action.type == "emergency" and user.has_emergency_access == true',
                    'action': 'allow',
                    'description': 'Allow emergency access for authorized users'
                }
            ]
        }
        
        # Risk-based access policy
        risk_based_policy = {
            'policy_id': 'risk_based_access_control',
            'name': 'Risk-based Access Control',
            'description': 'Control access based on risk score and context',
            'rules': [
                {
                    'condition': 'user.risk_score <= 30 and action.risk_level == "low"',
                    'action': 'allow',
                    'description': 'Allow low-risk users to perform low-risk actions'
                },
                {
                    'condition': 'user.risk_score > 70 or action.risk_level == "high"',
                    'action': 'require_approval',
                    'description': 'Require approval for high-risk users or actions'
                },
                {
                    'condition': 'user.failed_logins > 3 and time.since_last_failure < 3600',
                    'action': 'deny',
                    'description': 'Deny access after multiple failed logins'
                }
            ]
        }
        
        # Store policies
        policies = [regional_policy, time_based_policy, risk_based_policy]
        
        for policy in policies:
            self.abac_policies[policy['policy_id']] = policy
        
        logger.info(f"✅ Initialized {len(self.abac_policies)} default ABAC policies")
    
    async def evaluate_abac_policy(
        self,
        user_context: Dict[str, Any],
        resource_context: Dict[str, Any],
        action_context: Dict[str, Any],
        environment_context: Dict[str, Any]
    ) -> Tuple[str, str, List[str]]:
        """
        Evaluate ABAC policies
        Returns: (decision, reason, applicable_policies)
        """
        
        # This is a simplified ABAC evaluation
        # In production, would use a proper policy engine like OPA
        
        applicable_policies = []
        decisions = []
        reasons = []
        
        for policy_id, policy in self.abac_policies.items():
            try:
                policy_decision = self._evaluate_policy_rules(
                    policy, user_context, resource_context, action_context, environment_context
                )
                
                if policy_decision['applies']:
                    applicable_policies.append(policy_id)
                    decisions.append(policy_decision['decision'])
                    reasons.append(policy_decision['reason'])
                    
            except Exception as e:
                logger.error(f"❌ Error evaluating ABAC policy {policy_id}: {e}")
                decisions.append('deny')
                reasons.append(f"Policy evaluation error: {str(e)}")
        
        # Determine final decision (most restrictive wins)
        if 'deny' in decisions:
            final_decision = 'deny'
            final_reason = next(reason for decision, reason in zip(decisions, reasons) if decision == 'deny')
        elif 'require_approval' in decisions:
            final_decision = 'require_approval'
            final_reason = next(reason for decision, reason in zip(decisions, reasons) if decision == 'require_approval')
        elif 'allow' in decisions:
            final_decision = 'allow'
            final_reason = "ABAC policies allow access"
        else:
            final_decision = 'deny'
            final_reason = "No applicable ABAC policies found"
        
        return final_decision, final_reason, applicable_policies
    
    def _evaluate_policy_rules(
        self,
        policy: Dict[str, Any],
        user_context: Dict[str, Any],
        resource_context: Dict[str, Any],
        action_context: Dict[str, Any],
        environment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate individual policy rules"""
        
        # Simplified rule evaluation
        # In production, would use a proper expression evaluator
        
        context = {
            'user': user_context,
            'data': resource_context,
            'action': action_context,
            'time': environment_context.get('time', {}),
            'environment': environment_context
        }
        
        for rule in policy.get('rules', []):
            condition = rule.get('condition', '')
            
            # Very basic condition evaluation (would use proper parser in production)
            try:
                if self._evaluate_simple_condition(condition, context):
                    return {
                        'applies': True,
                        'decision': rule.get('action', 'deny'),
                        'reason': rule.get('description', 'ABAC rule matched')
                    }
            except Exception as e:
                logger.error(f"❌ Error evaluating condition '{condition}': {e}")
        
        return {
            'applies': False,
            'decision': 'allow',
            'reason': 'No matching rules'
        }
    
    def _evaluate_simple_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Simple condition evaluation (mock implementation)"""
        
        # This is a very basic implementation
        # In production, would use a proper policy language like Rego (OPA)
        
        # For demo purposes, just check some basic conditions
        if 'user.region == data.region' in condition:
            return context.get('user', {}).get('region') == context.get('data', {}).get('region')
        
        if 'data.sensitivity == "high"' in condition:
            return context.get('data', {}).get('sensitivity') == 'high'
        
        if 'user.role == "system"' in condition:
            return context.get('user', {}).get('role') == 'system'
        
        # Default to false for unknown conditions
        return False

# Global instances
rbac_manager = RBACManager()
sod_enforcer = SoDEnforcer(rbac_manager)
abac_manager = ABACManager()
