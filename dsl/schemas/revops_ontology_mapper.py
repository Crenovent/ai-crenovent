# RevOps Ontology to Database Schema Mapper
# Tasks 7.2-T01 to T20: Canonical ontology, schema mapping, multi-tenant data models

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Core RevOps entity types"""
    # Core business entities
    TENANT = "tenant"
    USER = "user"
    ROLE = "role"
    
    # CRM entities
    ACCOUNT = "account"
    CONTACT = "contact"
    LEAD = "lead"
    OPPORTUNITY = "opportunity"
    STAGE = "stage"
    PIPELINE = "pipeline"
    
    # Product and pricing
    PRODUCT = "product"
    PRICE_BOOK = "price_book"
    DISCOUNT = "discount"
    
    # Contracts and subscriptions
    CONTRACT = "contract"
    ORDER = "order"
    SUBSCRIPTION = "subscription"
    
    # Billing and payments
    INVOICE = "invoice"
    PAYMENT = "payment"
    REFUND = "refund"
    AGING = "aging"
    
    # Support and service
    CASE = "case"
    TICKET = "ticket"
    SLA = "sla"
    ESCALATION = "escalation"
    
    # GTM and planning
    TERRITORY = "territory"
    QUOTA = "quota"
    FORECAST = "forecast"
    
    # Activities and engagement
    ACTIVITY = "activity"
    TASK = "task"
    MEETING = "meeting"
    EMAIL_LOG = "email_log"
    
    # Governance and compliance
    EVIDENCE = "evidence"
    OVERRIDE = "override"
    POLICY_APPLIED = "policy_applied"
    COMPLIANCE_TAG = "compliance_tag"
    DATA_CLASS = "data_class"
    RESIDENCY_ZONE = "residency_zone"

class DataType(Enum):
    """Database data types"""
    UUID = "uuid"
    STRING = "varchar"
    TEXT = "text"
    INTEGER = "integer"
    BIGINT = "bigint"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    DATE = "date"
    JSON = "jsonb"
    ARRAY = "array"

class ConstraintType(Enum):
    """Database constraint types"""
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    CHECK = "check"
    INDEX = "index"

@dataclass
class EntityAttribute:
    """Attribute definition for an entity"""
    name: str
    data_type: DataType
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    default_value: Optional[str] = None
    description: str = ""
    pii_classification: Optional[str] = None  # PII, SENSITIVE, PUBLIC
    compliance_tags: List[str] = field(default_factory=list)

@dataclass
class EntityRelationship:
    """Relationship between entities"""
    name: str
    from_entity: str
    to_entity: str
    relationship_type: str  # one_to_one, one_to_many, many_to_many
    foreign_key_column: str
    referenced_column: str = "id"
    cascade_delete: bool = False
    description: str = ""

@dataclass
class EntityConstraint:
    """Constraint definition for an entity"""
    name: str
    constraint_type: ConstraintType
    columns: List[str]
    referenced_table: Optional[str] = None
    referenced_columns: Optional[List[str]] = None
    condition: Optional[str] = None
    description: str = ""

@dataclass
class EntityDefinition:
    """Complete entity definition"""
    name: str
    entity_type: EntityType
    table_name: str
    description: str
    attributes: List[EntityAttribute] = field(default_factory=list)
    relationships: List[EntityRelationship] = field(default_factory=list)
    constraints: List[EntityConstraint] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    multi_tenant: bool = True
    audit_enabled: bool = True
    soft_delete: bool = True
    version_controlled: bool = False

class RevOpsOntologyMapper:
    """
    RevOps Ontology to Database Schema Mapper
    Tasks 7.2-T01 to T20: Canonical ontology, schema mapping, multi-tenant models
    """
    
    def __init__(self):
        self.entities: Dict[str, EntityDefinition] = {}
        self.naming_conventions = {
            'table_prefix': '',
            'column_case': 'snake_case',
            'foreign_key_suffix': '_id',
            'boolean_prefix': 'is_',
            'timestamp_suffix': '_at'
        }
        
        # Initialize canonical ontology
        self._initialize_canonical_ontology()
    
    def _initialize_canonical_ontology(self) -> None:
        """Initialize canonical RevOps ontology"""
        
        # Core multi-tenant entities
        self._define_tenant_entities()
        self._define_user_entities()
        
        # CRM entities
        self._define_crm_entities()
        
        # Product and pricing entities
        self._define_product_entities()
        
        # Contract and subscription entities
        self._define_contract_entities()
        
        # Billing entities
        self._define_billing_entities()
        
        # Support entities
        self._define_support_entities()
        
        # GTM entities
        self._define_gtm_entities()
        
        # Activity entities
        self._define_activity_entities()
        
        # Governance entities
        self._define_governance_entities()
        
        # Reference data entities
        self._define_reference_entities()
    
    def _define_tenant_entities(self) -> None:
        """Define tenant and user management entities"""
        
        # Tenant entity
        tenant = EntityDefinition(
            name="Tenant",
            entity_type=EntityType.TENANT,
            table_name="tenants",
            description="Multi-tenant organization container",
            multi_tenant=False  # Root entity
        )
        
        tenant.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("name", DataType.STRING, max_length=255, nullable=False, description="Tenant name"),
            EntityAttribute("slug", DataType.STRING, max_length=100, nullable=False, description="URL-safe identifier"),
            EntityAttribute("industry_code", DataType.STRING, max_length=50, nullable=False, description="Industry classification"),
            EntityAttribute("data_residency", DataType.STRING, max_length=10, nullable=False, description="Data residency region"),
            EntityAttribute("tier", DataType.STRING, max_length=50, nullable=False, description="Service tier"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="Tenant status"),
            EntityAttribute("compliance_frameworks", DataType.JSON, description="Applicable compliance frameworks"),
            EntityAttribute("feature_flags", DataType.JSON, description="Enabled features"),
            EntityAttribute("settings", DataType.JSON, description="Tenant-specific settings"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP, description="Soft delete timestamp")
        ]
        
        tenant.constraints = [
            EntityConstraint("pk_tenants", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("uk_tenants_slug", ConstraintType.UNIQUE, ["slug"]),
            EntityConstraint("idx_tenants_industry", ConstraintType.INDEX, ["industry_code"]),
            EntityConstraint("idx_tenants_status", ConstraintType.INDEX, ["status"])
        ]
        
        self.entities["tenant"] = tenant
        
        # User entity
        user = EntityDefinition(
            name="User",
            entity_type=EntityType.USER,
            table_name="users",
            description="System users with tenant association"
        )
        
        user.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("email", DataType.STRING, max_length=255, nullable=False, description="User email", pii_classification="PII"),
            EntityAttribute("first_name", DataType.STRING, max_length=100, description="First name", pii_classification="PII"),
            EntityAttribute("last_name", DataType.STRING, max_length=100, description="Last name", pii_classification="PII"),
            EntityAttribute("role", DataType.STRING, max_length=50, nullable=False, description="User role"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="User status"),
            EntityAttribute("last_login_at", DataType.TIMESTAMP, description="Last login timestamp"),
            EntityAttribute("preferences", DataType.JSON, description="User preferences"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        user.relationships = [
            EntityRelationship("tenant", "user", "tenant", "many_to_one", "tenant_id")
        ]
        
        user.constraints = [
            EntityConstraint("pk_users", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_users_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("uk_users_tenant_email", ConstraintType.UNIQUE, ["tenant_id", "email"]),
            EntityConstraint("idx_users_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_users_email", ConstraintType.INDEX, ["email"])
        ]
        
        self.entities["user"] = user
    
    def _define_crm_entities(self) -> None:
        """Define CRM entities"""
        
        # Account entity
        account = EntityDefinition(
            name="Account",
            entity_type=EntityType.ACCOUNT,
            table_name="accounts",
            description="Customer accounts/companies"
        )
        
        account.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("external_id", DataType.STRING, max_length=100, description="External system ID"),
            EntityAttribute("name", DataType.STRING, max_length=255, nullable=False, description="Account name"),
            EntityAttribute("type", DataType.STRING, max_length=50, description="Account type"),
            EntityAttribute("industry", DataType.STRING, max_length=100, description="Industry"),
            EntityAttribute("website", DataType.STRING, max_length=255, description="Website URL"),
            EntityAttribute("phone", DataType.STRING, max_length=50, description="Phone number", pii_classification="PII"),
            EntityAttribute("billing_address", DataType.JSON, description="Billing address", pii_classification="PII"),
            EntityAttribute("shipping_address", DataType.JSON, description="Shipping address", pii_classification="PII"),
            EntityAttribute("annual_revenue", DataType.DECIMAL, precision=15, scale=2, description="Annual revenue"),
            EntityAttribute("employee_count", DataType.INTEGER, description="Number of employees"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="Account status"),
            EntityAttribute("owner_id", DataType.UUID, description="Account owner"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        account.relationships = [
            EntityRelationship("tenant", "account", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("owner", "account", "user", "many_to_one", "owner_id")
        ]
        
        account.constraints = [
            EntityConstraint("pk_accounts", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_accounts_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_accounts_owner", ConstraintType.FOREIGN_KEY, ["owner_id"], "users", ["id"]),
            EntityConstraint("uk_accounts_tenant_external", ConstraintType.UNIQUE, ["tenant_id", "external_id"]),
            EntityConstraint("idx_accounts_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_accounts_name", ConstraintType.INDEX, ["name"]),
            EntityConstraint("idx_accounts_owner", ConstraintType.INDEX, ["owner_id"])
        ]
        
        self.entities["account"] = account
        
        # Contact entity
        contact = EntityDefinition(
            name="Contact",
            entity_type=EntityType.CONTACT,
            table_name="contacts",
            description="Individual contacts within accounts"
        )
        
        contact.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("account_id", DataType.UUID, description="Associated account"),
            EntityAttribute("external_id", DataType.STRING, max_length=100, description="External system ID"),
            EntityAttribute("first_name", DataType.STRING, max_length=100, description="First name", pii_classification="PII"),
            EntityAttribute("last_name", DataType.STRING, max_length=100, description="Last name", pii_classification="PII"),
            EntityAttribute("email", DataType.STRING, max_length=255, description="Email address", pii_classification="PII"),
            EntityAttribute("phone", DataType.STRING, max_length=50, description="Phone number", pii_classification="PII"),
            EntityAttribute("mobile", DataType.STRING, max_length=50, description="Mobile number", pii_classification="PII"),
            EntityAttribute("title", DataType.STRING, max_length=100, description="Job title"),
            EntityAttribute("department", DataType.STRING, max_length=100, description="Department"),
            EntityAttribute("mailing_address", DataType.JSON, description="Mailing address", pii_classification="PII"),
            EntityAttribute("lead_source", DataType.STRING, max_length=100, description="Lead source"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="Contact status"),
            EntityAttribute("owner_id", DataType.UUID, description="Contact owner"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        contact.relationships = [
            EntityRelationship("tenant", "contact", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("account", "contact", "account", "many_to_one", "account_id"),
            EntityRelationship("owner", "contact", "user", "many_to_one", "owner_id")
        ]
        
        contact.constraints = [
            EntityConstraint("pk_contacts", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_contacts_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_contacts_account", ConstraintType.FOREIGN_KEY, ["account_id"], "accounts", ["id"]),
            EntityConstraint("fk_contacts_owner", ConstraintType.FOREIGN_KEY, ["owner_id"], "users", ["id"]),
            EntityConstraint("uk_contacts_tenant_external", ConstraintType.UNIQUE, ["tenant_id", "external_id"]),
            EntityConstraint("idx_contacts_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_contacts_account", ConstraintType.INDEX, ["account_id"]),
            EntityConstraint("idx_contacts_email", ConstraintType.INDEX, ["email"])
        ]
        
        self.entities["contact"] = contact
        
        # Opportunity entity
        opportunity = EntityDefinition(
            name="Opportunity",
            entity_type=EntityType.OPPORTUNITY,
            table_name="opportunities",
            description="Sales opportunities and deals"
        )
        
        opportunity.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("account_id", DataType.UUID, nullable=False, description="Associated account"),
            EntityAttribute("external_id", DataType.STRING, max_length=100, description="External system ID"),
            EntityAttribute("name", DataType.STRING, max_length=255, nullable=False, description="Opportunity name"),
            EntityAttribute("stage", DataType.STRING, max_length=100, nullable=False, description="Current stage"),
            EntityAttribute("pipeline", DataType.STRING, max_length=100, description="Sales pipeline"),
            EntityAttribute("amount", DataType.DECIMAL, precision=15, scale=2, description="Deal amount"),
            EntityAttribute("probability", DataType.DECIMAL, precision=5, scale=2, description="Win probability %"),
            EntityAttribute("close_date", DataType.DATE, description="Expected close date"),
            EntityAttribute("lead_source", DataType.STRING, max_length=100, description="Lead source"),
            EntityAttribute("type", DataType.STRING, max_length=50, description="Opportunity type"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="Opportunity status"),
            EntityAttribute("owner_id", DataType.UUID, description="Opportunity owner"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        opportunity.relationships = [
            EntityRelationship("tenant", "opportunity", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("account", "opportunity", "account", "many_to_one", "account_id"),
            EntityRelationship("owner", "opportunity", "user", "many_to_one", "owner_id")
        ]
        
        opportunity.constraints = [
            EntityConstraint("pk_opportunities", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_opportunities_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_opportunities_account", ConstraintType.FOREIGN_KEY, ["account_id"], "accounts", ["id"]),
            EntityConstraint("fk_opportunities_owner", ConstraintType.FOREIGN_KEY, ["owner_id"], "users", ["id"]),
            EntityConstraint("uk_opportunities_tenant_external", ConstraintType.UNIQUE, ["tenant_id", "external_id"]),
            EntityConstraint("idx_opportunities_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_opportunities_account", ConstraintType.INDEX, ["account_id"]),
            EntityConstraint("idx_opportunities_stage", ConstraintType.INDEX, ["stage"]),
            EntityConstraint("idx_opportunities_close_date", ConstraintType.INDEX, ["close_date"])
        ]
        
        self.entities["opportunity"] = opportunity
    
    def _define_product_entities(self) -> None:
        """Define product and pricing entities"""
        
        # Product entity
        product = EntityDefinition(
            name="Product",
            entity_type=EntityType.PRODUCT,
            table_name="products",
            description="Products and services offered"
        )
        
        product.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("external_id", DataType.STRING, max_length=100, description="External system ID"),
            EntityAttribute("name", DataType.STRING, max_length=255, nullable=False, description="Product name"),
            EntityAttribute("code", DataType.STRING, max_length=100, description="Product code/SKU"),
            EntityAttribute("description", DataType.TEXT, description="Product description"),
            EntityAttribute("category", DataType.STRING, max_length=100, description="Product category"),
            EntityAttribute("family", DataType.STRING, max_length=100, description="Product family"),
            EntityAttribute("unit_price", DataType.DECIMAL, precision=15, scale=2, description="Unit price"),
            EntityAttribute("cost", DataType.DECIMAL, precision=15, scale=2, description="Product cost"),
            EntityAttribute("currency", DataType.STRING, max_length=3, description="Currency code"),
            EntityAttribute("is_active", DataType.BOOLEAN, nullable=False, default_value="true", description="Active status"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        product.relationships = [
            EntityRelationship("tenant", "product", "tenant", "many_to_one", "tenant_id")
        ]
        
        product.constraints = [
            EntityConstraint("pk_products", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_products_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("uk_products_tenant_code", ConstraintType.UNIQUE, ["tenant_id", "code"]),
            EntityConstraint("idx_products_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_products_category", ConstraintType.INDEX, ["category"]),
            EntityConstraint("idx_products_active", ConstraintType.INDEX, ["is_active"])
        ]
        
        self.entities["product"] = product
    
    def _define_contract_entities(self) -> None:
        """Define contract and subscription entities"""
        
        # Contract entity
        contract = EntityDefinition(
            name="Contract",
            entity_type=EntityType.CONTRACT,
            table_name="contracts",
            description="Customer contracts and agreements"
        )
        
        contract.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("account_id", DataType.UUID, nullable=False, description="Associated account"),
            EntityAttribute("opportunity_id", DataType.UUID, description="Source opportunity"),
            EntityAttribute("external_id", DataType.STRING, max_length=100, description="External system ID"),
            EntityAttribute("contract_number", DataType.STRING, max_length=100, nullable=False, description="Contract number"),
            EntityAttribute("name", DataType.STRING, max_length=255, description="Contract name"),
            EntityAttribute("type", DataType.STRING, max_length=50, description="Contract type"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="Contract status"),
            EntityAttribute("start_date", DataType.DATE, nullable=False, description="Contract start date"),
            EntityAttribute("end_date", DataType.DATE, description="Contract end date"),
            EntityAttribute("term_months", DataType.INTEGER, description="Contract term in months"),
            EntityAttribute("total_value", DataType.DECIMAL, precision=15, scale=2, description="Total contract value"),
            EntityAttribute("currency", DataType.STRING, max_length=3, description="Currency code"),
            EntityAttribute("billing_frequency", DataType.STRING, max_length=20, description="Billing frequency"),
            EntityAttribute("auto_renew", DataType.BOOLEAN, default_value="false", description="Auto-renewal flag"),
            EntityAttribute("owner_id", DataType.UUID, description="Contract owner"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        contract.relationships = [
            EntityRelationship("tenant", "contract", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("account", "contract", "account", "many_to_one", "account_id"),
            EntityRelationship("opportunity", "contract", "opportunity", "many_to_one", "opportunity_id"),
            EntityRelationship("owner", "contract", "user", "many_to_one", "owner_id")
        ]
        
        contract.constraints = [
            EntityConstraint("pk_contracts", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_contracts_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_contracts_account", ConstraintType.FOREIGN_KEY, ["account_id"], "accounts", ["id"]),
            EntityConstraint("fk_contracts_opportunity", ConstraintType.FOREIGN_KEY, ["opportunity_id"], "opportunities", ["id"]),
            EntityConstraint("fk_contracts_owner", ConstraintType.FOREIGN_KEY, ["owner_id"], "users", ["id"]),
            EntityConstraint("uk_contracts_tenant_number", ConstraintType.UNIQUE, ["tenant_id", "contract_number"]),
            EntityConstraint("idx_contracts_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_contracts_account", ConstraintType.INDEX, ["account_id"]),
            EntityConstraint("idx_contracts_status", ConstraintType.INDEX, ["status"]),
            EntityConstraint("idx_contracts_dates", ConstraintType.INDEX, ["start_date", "end_date"])
        ]
        
        self.entities["contract"] = contract
    
    def _define_billing_entities(self) -> None:
        """Define billing and payment entities"""
        
        # Invoice entity
        invoice = EntityDefinition(
            name="Invoice",
            entity_type=EntityType.INVOICE,
            table_name="invoices",
            description="Customer invoices"
        )
        
        invoice.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("account_id", DataType.UUID, nullable=False, description="Associated account"),
            EntityAttribute("contract_id", DataType.UUID, description="Associated contract"),
            EntityAttribute("external_id", DataType.STRING, max_length=100, description="External system ID"),
            EntityAttribute("invoice_number", DataType.STRING, max_length=100, nullable=False, description="Invoice number"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="Invoice status"),
            EntityAttribute("invoice_date", DataType.DATE, nullable=False, description="Invoice date"),
            EntityAttribute("due_date", DataType.DATE, description="Payment due date"),
            EntityAttribute("subtotal", DataType.DECIMAL, precision=15, scale=2, description="Subtotal amount"),
            EntityAttribute("tax_amount", DataType.DECIMAL, precision=15, scale=2, description="Tax amount"),
            EntityAttribute("total_amount", DataType.DECIMAL, precision=15, scale=2, nullable=False, description="Total amount"),
            EntityAttribute("paid_amount", DataType.DECIMAL, precision=15, scale=2, default_value="0", description="Amount paid"),
            EntityAttribute("balance", DataType.DECIMAL, precision=15, scale=2, description="Outstanding balance"),
            EntityAttribute("currency", DataType.STRING, max_length=3, description="Currency code"),
            EntityAttribute("payment_terms", DataType.STRING, max_length=50, description="Payment terms"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        invoice.relationships = [
            EntityRelationship("tenant", "invoice", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("account", "invoice", "account", "many_to_one", "account_id"),
            EntityRelationship("contract", "invoice", "contract", "many_to_one", "contract_id")
        ]
        
        invoice.constraints = [
            EntityConstraint("pk_invoices", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_invoices_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_invoices_account", ConstraintType.FOREIGN_KEY, ["account_id"], "accounts", ["id"]),
            EntityConstraint("fk_invoices_contract", ConstraintType.FOREIGN_KEY, ["contract_id"], "contracts", ["id"]),
            EntityConstraint("uk_invoices_tenant_number", ConstraintType.UNIQUE, ["tenant_id", "invoice_number"]),
            EntityConstraint("idx_invoices_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_invoices_account", ConstraintType.INDEX, ["account_id"]),
            EntityConstraint("idx_invoices_status", ConstraintType.INDEX, ["status"]),
            EntityConstraint("idx_invoices_due_date", ConstraintType.INDEX, ["due_date"])
        ]
        
        self.entities["invoice"] = invoice
    
    def _define_support_entities(self) -> None:
        """Define support and service entities"""
        
        # Case entity
        case = EntityDefinition(
            name="Case",
            entity_type=EntityType.CASE,
            table_name="cases",
            description="Customer support cases"
        )
        
        case.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("account_id", DataType.UUID, description="Associated account"),
            EntityAttribute("contact_id", DataType.UUID, description="Associated contact"),
            EntityAttribute("external_id", DataType.STRING, max_length=100, description="External system ID"),
            EntityAttribute("case_number", DataType.STRING, max_length=100, nullable=False, description="Case number"),
            EntityAttribute("subject", DataType.STRING, max_length=255, nullable=False, description="Case subject"),
            EntityAttribute("description", DataType.TEXT, description="Case description"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="Case status"),
            EntityAttribute("priority", DataType.STRING, max_length=20, description="Case priority"),
            EntityAttribute("type", DataType.STRING, max_length=50, description="Case type"),
            EntityAttribute("origin", DataType.STRING, max_length=50, description="Case origin"),
            EntityAttribute("owner_id", DataType.UUID, description="Case owner"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("closed_at", DataType.TIMESTAMP, description="Case closed timestamp"),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        case.relationships = [
            EntityRelationship("tenant", "case", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("account", "case", "account", "many_to_one", "account_id"),
            EntityRelationship("contact", "case", "contact", "many_to_one", "contact_id"),
            EntityRelationship("owner", "case", "user", "many_to_one", "owner_id")
        ]
        
        case.constraints = [
            EntityConstraint("pk_cases", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_cases_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_cases_account", ConstraintType.FOREIGN_KEY, ["account_id"], "accounts", ["id"]),
            EntityConstraint("fk_cases_contact", ConstraintType.FOREIGN_KEY, ["contact_id"], "contacts", ["id"]),
            EntityConstraint("fk_cases_owner", ConstraintType.FOREIGN_KEY, ["owner_id"], "users", ["id"]),
            EntityConstraint("uk_cases_tenant_number", ConstraintType.UNIQUE, ["tenant_id", "case_number"]),
            EntityConstraint("idx_cases_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_cases_account", ConstraintType.INDEX, ["account_id"]),
            EntityConstraint("idx_cases_status", ConstraintType.INDEX, ["status"]),
            EntityConstraint("idx_cases_priority", ConstraintType.INDEX, ["priority"])
        ]
        
        self.entities["case"] = case
    
    def _define_gtm_entities(self) -> None:
        """Define GTM and planning entities"""
        
        # Territory entity
        territory = EntityDefinition(
            name="Territory",
            entity_type=EntityType.TERRITORY,
            table_name="territories",
            description="Sales territories"
        )
        
        territory.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("name", DataType.STRING, max_length=255, nullable=False, description="Territory name"),
            EntityAttribute("code", DataType.STRING, max_length=50, description="Territory code"),
            EntityAttribute("type", DataType.STRING, max_length=50, description="Territory type"),
            EntityAttribute("parent_territory_id", DataType.UUID, description="Parent territory"),
            EntityAttribute("manager_id", DataType.UUID, description="Territory manager"),
            EntityAttribute("is_active", DataType.BOOLEAN, nullable=False, default_value="true", description="Active status"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        territory.relationships = [
            EntityRelationship("tenant", "territory", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("parent", "territory", "territory", "many_to_one", "parent_territory_id"),
            EntityRelationship("manager", "territory", "user", "many_to_one", "manager_id")
        ]
        
        territory.constraints = [
            EntityConstraint("pk_territories", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_territories_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_territories_parent", ConstraintType.FOREIGN_KEY, ["parent_territory_id"], "territories", ["id"]),
            EntityConstraint("fk_territories_manager", ConstraintType.FOREIGN_KEY, ["manager_id"], "users", ["id"]),
            EntityConstraint("uk_territories_tenant_code", ConstraintType.UNIQUE, ["tenant_id", "code"]),
            EntityConstraint("idx_territories_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_territories_manager", ConstraintType.INDEX, ["manager_id"])
        ]
        
        self.entities["territory"] = territory
    
    def _define_activity_entities(self) -> None:
        """Define activity and engagement entities"""
        
        # Activity entity
        activity = EntityDefinition(
            name="Activity",
            entity_type=EntityType.ACTIVITY,
            table_name="activities",
            description="Customer engagement activities"
        )
        
        activity.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("account_id", DataType.UUID, description="Associated account"),
            EntityAttribute("contact_id", DataType.UUID, description="Associated contact"),
            EntityAttribute("opportunity_id", DataType.UUID, description="Associated opportunity"),
            EntityAttribute("external_id", DataType.STRING, max_length=100, description="External system ID"),
            EntityAttribute("type", DataType.STRING, max_length=50, nullable=False, description="Activity type"),
            EntityAttribute("subject", DataType.STRING, max_length=255, description="Activity subject"),
            EntityAttribute("description", DataType.TEXT, description="Activity description"),
            EntityAttribute("status", DataType.STRING, max_length=20, nullable=False, description="Activity status"),
            EntityAttribute("priority", DataType.STRING, max_length=20, description="Activity priority"),
            EntityAttribute("due_date", DataType.TIMESTAMP, description="Due date"),
            EntityAttribute("completed_at", DataType.TIMESTAMP, description="Completion timestamp"),
            EntityAttribute("owner_id", DataType.UUID, description="Activity owner"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("deleted_at", DataType.TIMESTAMP)
        ]
        
        activity.relationships = [
            EntityRelationship("tenant", "activity", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("account", "activity", "account", "many_to_one", "account_id"),
            EntityRelationship("contact", "activity", "contact", "many_to_one", "contact_id"),
            EntityRelationship("opportunity", "activity", "opportunity", "many_to_one", "opportunity_id"),
            EntityRelationship("owner", "activity", "user", "many_to_one", "owner_id")
        ]
        
        activity.constraints = [
            EntityConstraint("pk_activities", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_activities_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_activities_account", ConstraintType.FOREIGN_KEY, ["account_id"], "accounts", ["id"]),
            EntityConstraint("fk_activities_contact", ConstraintType.FOREIGN_KEY, ["contact_id"], "contacts", ["id"]),
            EntityConstraint("fk_activities_opportunity", ConstraintType.FOREIGN_KEY, ["opportunity_id"], "opportunities", ["id"]),
            EntityConstraint("fk_activities_owner", ConstraintType.FOREIGN_KEY, ["owner_id"], "users", ["id"]),
            EntityConstraint("idx_activities_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_activities_type", ConstraintType.INDEX, ["type"]),
            EntityConstraint("idx_activities_due_date", ConstraintType.INDEX, ["due_date"])
        ]
        
        self.entities["activity"] = activity
    
    def _define_governance_entities(self) -> None:
        """Define governance and compliance entities"""
        
        # Evidence entity
        evidence = EntityDefinition(
            name="Evidence",
            entity_type=EntityType.EVIDENCE,
            table_name="evidence_packs",
            description="Governance evidence packs"
        )
        
        evidence.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("workflow_id", DataType.STRING, max_length=255, description="Associated workflow"),
            EntityAttribute("execution_id", DataType.STRING, max_length=255, description="Execution ID"),
            EntityAttribute("evidence_type", DataType.STRING, max_length=50, nullable=False, description="Evidence type"),
            EntityAttribute("policy_pack_version", DataType.STRING, max_length=50, description="Policy pack version"),
            EntityAttribute("compliance_frameworks", DataType.JSON, description="Applicable frameworks"),
            EntityAttribute("evidence_data", DataType.JSON, nullable=False, description="Evidence payload"),
            EntityAttribute("hash_chain", DataType.STRING, max_length=255, description="Tamper-evident hash"),
            EntityAttribute("digital_signature", DataType.TEXT, description="Digital signature"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("immutable", DataType.BOOLEAN, nullable=False, default_value="true", description="Immutability flag")
        ]
        
        evidence.relationships = [
            EntityRelationship("tenant", "evidence", "tenant", "many_to_one", "tenant_id")
        ]
        
        evidence.constraints = [
            EntityConstraint("pk_evidence", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_evidence_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("idx_evidence_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_evidence_workflow", ConstraintType.INDEX, ["workflow_id"]),
            EntityConstraint("idx_evidence_execution", ConstraintType.INDEX, ["execution_id"]),
            EntityConstraint("idx_evidence_type", ConstraintType.INDEX, ["evidence_type"])
        ]
        
        self.entities["evidence"] = evidence
        
        # Override entity
        override = EntityDefinition(
            name="Override",
            entity_type=EntityType.OVERRIDE,
            table_name="override_ledger",
            description="Policy override ledger"
        )
        
        override.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("tenant_id", DataType.UUID, nullable=False, description="Tenant association"),
            EntityAttribute("workflow_id", DataType.STRING, max_length=255, description="Associated workflow"),
            EntityAttribute("execution_id", DataType.STRING, max_length=255, description="Execution ID"),
            EntityAttribute("policy_id", DataType.STRING, max_length=255, nullable=False, description="Overridden policy"),
            EntityAttribute("override_type", DataType.STRING, max_length=50, nullable=False, description="Override type"),
            EntityAttribute("reason", DataType.TEXT, nullable=False, description="Override reason"),
            EntityAttribute("requested_by", DataType.UUID, nullable=False, description="Requester"),
            EntityAttribute("approved_by", DataType.UUID, description="Approver"),
            EntityAttribute("approval_status", DataType.STRING, max_length=20, nullable=False, description="Approval status"),
            EntityAttribute("expires_at", DataType.TIMESTAMP, description="Override expiration"),
            EntityAttribute("revoked_at", DataType.TIMESTAMP, description="Revocation timestamp"),
            EntityAttribute("revoked_by", DataType.UUID, description="Revoker"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False)
        ]
        
        override.relationships = [
            EntityRelationship("tenant", "override", "tenant", "many_to_one", "tenant_id"),
            EntityRelationship("requester", "override", "user", "many_to_one", "requested_by"),
            EntityRelationship("approver", "override", "user", "many_to_one", "approved_by"),
            EntityRelationship("revoker", "override", "user", "many_to_one", "revoked_by")
        ]
        
        override.constraints = [
            EntityConstraint("pk_overrides", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("fk_overrides_tenant", ConstraintType.FOREIGN_KEY, ["tenant_id"], "tenants", ["id"]),
            EntityConstraint("fk_overrides_requester", ConstraintType.FOREIGN_KEY, ["requested_by"], "users", ["id"]),
            EntityConstraint("fk_overrides_approver", ConstraintType.FOREIGN_KEY, ["approved_by"], "users", ["id"]),
            EntityConstraint("fk_overrides_revoker", ConstraintType.FOREIGN_KEY, ["revoked_by"], "users", ["id"]),
            EntityConstraint("idx_overrides_tenant", ConstraintType.INDEX, ["tenant_id"]),
            EntityConstraint("idx_overrides_workflow", ConstraintType.INDEX, ["workflow_id"]),
            EntityConstraint("idx_overrides_policy", ConstraintType.INDEX, ["policy_id"]),
            EntityConstraint("idx_overrides_status", ConstraintType.INDEX, ["approval_status"])
        ]
        
        self.entities["override"] = override
    
    def _define_reference_entities(self) -> None:
        """Define reference data entities"""
        
        # Compliance Tag entity
        compliance_tag = EntityDefinition(
            name="ComplianceTag",
            entity_type=EntityType.COMPLIANCE_TAG,
            table_name="compliance_tags",
            description="Compliance classification tags",
            multi_tenant=False  # Global reference data
        )
        
        compliance_tag.attributes = [
            EntityAttribute("id", DataType.UUID, description="Primary key"),
            EntityAttribute("code", DataType.STRING, max_length=50, nullable=False, description="Tag code"),
            EntityAttribute("name", DataType.STRING, max_length=255, nullable=False, description="Tag name"),
            EntityAttribute("description", DataType.TEXT, description="Tag description"),
            EntityAttribute("framework", DataType.STRING, max_length=50, description="Compliance framework"),
            EntityAttribute("severity", DataType.STRING, max_length=20, description="Severity level"),
            EntityAttribute("is_active", DataType.BOOLEAN, nullable=False, default_value="true", description="Active status"),
            EntityAttribute("created_at", DataType.TIMESTAMP, nullable=False),
            EntityAttribute("updated_at", DataType.TIMESTAMP, nullable=False)
        ]
        
        compliance_tag.constraints = [
            EntityConstraint("pk_compliance_tags", ConstraintType.PRIMARY_KEY, ["id"]),
            EntityConstraint("uk_compliance_tags_code", ConstraintType.UNIQUE, ["code"]),
            EntityConstraint("idx_compliance_tags_framework", ConstraintType.INDEX, ["framework"])
        ]
        
        self.entities["compliance_tag"] = compliance_tag
    
    def generate_ddl_script(self, include_rls: bool = False) -> str:
        """Generate DDL script for all entities"""
        
        ddl_statements = []
        
        # Add header
        ddl_statements.append("-- RevOps Canonical Schema DDL")
        ddl_statements.append("-- Generated by RevOps Ontology Mapper")
        ddl_statements.append(f"-- Generated at: {datetime.now(timezone.utc).isoformat()}")
        ddl_statements.append("")
        
        # Add extensions
        ddl_statements.append("-- Required extensions")
        ddl_statements.append('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        ddl_statements.append('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
        ddl_statements.append("")
        
        # Create tables in dependency order
        creation_order = self._get_table_creation_order()
        
        for entity_name in creation_order:
            entity = self.entities[entity_name]
            ddl_statements.append(self._generate_table_ddl(entity))
            ddl_statements.append("")
        
        # Add foreign key constraints
        ddl_statements.append("-- Foreign key constraints")
        for entity_name in creation_order:
            entity = self.entities[entity_name]
            fk_statements = self._generate_foreign_key_ddl(entity)
            if fk_statements:
                ddl_statements.extend(fk_statements)
                ddl_statements.append("")
        
        # Add indexes
        ddl_statements.append("-- Indexes")
        for entity_name in creation_order:
            entity = self.entities[entity_name]
            index_statements = self._generate_index_ddl(entity)
            if index_statements:
                ddl_statements.extend(index_statements)
                ddl_statements.append("")
        
        # Add RLS policies if requested
        if include_rls:
            ddl_statements.append("-- Row Level Security Policies")
            for entity_name in creation_order:
                entity = self.entities[entity_name]
                if entity.multi_tenant:
                    rls_statements = self._generate_rls_ddl(entity)
                    if rls_statements:
                        ddl_statements.extend(rls_statements)
                        ddl_statements.append("")
        
        return "\n".join(ddl_statements)
    
    def _get_table_creation_order(self) -> List[str]:
        """Get table creation order based on dependencies"""
        # Simple dependency resolution - in practice would use topological sort
        ordered_entities = [
            "compliance_tag",  # No dependencies
            "tenant",          # No dependencies
            "user",            # Depends on tenant
            "account",         # Depends on tenant, user
            "contact",         # Depends on tenant, account, user
            "product",         # Depends on tenant
            "opportunity",     # Depends on tenant, account, user
            "contract",        # Depends on tenant, account, opportunity, user
            "invoice",         # Depends on tenant, account, contract
            "case",            # Depends on tenant, account, contact, user
            "territory",       # Depends on tenant, user
            "activity",        # Depends on tenant, account, contact, opportunity, user
            "evidence",        # Depends on tenant
            "override"         # Depends on tenant, user
        ]
        
        return [name for name in ordered_entities if name in self.entities]
    
    def _generate_table_ddl(self, entity: EntityDefinition) -> str:
        """Generate CREATE TABLE DDL for entity"""
        
        lines = [f"CREATE TABLE {entity.table_name} ("]
        
        # Add columns
        column_lines = []
        for attr in entity.attributes:
            column_def = f"    {attr.name} {self._get_sql_type(attr)}"
            
            if not attr.nullable:
                column_def += " NOT NULL"
            
            if attr.default_value:
                column_def += f" DEFAULT {attr.default_value}"
            
            column_lines.append(column_def)
        
        lines.append(",\n".join(column_lines))
        lines.append(");")
        
        # Add table comment
        if entity.description:
            lines.append(f"COMMENT ON TABLE {entity.table_name} IS '{entity.description}';")
        
        # Add column comments
        for attr in entity.attributes:
            if attr.description:
                lines.append(f"COMMENT ON COLUMN {entity.table_name}.{attr.name} IS '{attr.description}';")
        
        return "\n".join(lines)
    
    def _generate_foreign_key_ddl(self, entity: EntityDefinition) -> List[str]:
        """Generate foreign key constraint DDL"""
        
        statements = []
        
        for constraint in entity.constraints:
            if constraint.constraint_type == ConstraintType.FOREIGN_KEY:
                fk_ddl = (
                    f"ALTER TABLE {entity.table_name} "
                    f"ADD CONSTRAINT {constraint.name} "
                    f"FOREIGN KEY ({', '.join(constraint.columns)}) "
                    f"REFERENCES {constraint.referenced_table} ({', '.join(constraint.referenced_columns or ['id'])});"
                )
                statements.append(fk_ddl)
        
        return statements
    
    def _generate_index_ddl(self, entity: EntityDefinition) -> List[str]:
        """Generate index DDL"""
        
        statements = []
        
        for constraint in entity.constraints:
            if constraint.constraint_type == ConstraintType.INDEX:
                index_name = constraint.name
                columns = ', '.join(constraint.columns)
                index_ddl = f"CREATE INDEX {index_name} ON {entity.table_name} ({columns});"
                statements.append(index_ddl)
            elif constraint.constraint_type == ConstraintType.UNIQUE:
                unique_name = constraint.name
                columns = ', '.join(constraint.columns)
                unique_ddl = f"ALTER TABLE {entity.table_name} ADD CONSTRAINT {unique_name} UNIQUE ({columns});"
                statements.append(unique_ddl)
        
        return statements
    
    def _generate_rls_ddl(self, entity: EntityDefinition) -> List[str]:
        """Generate RLS policy DDL"""
        
        statements = []
        
        if entity.multi_tenant:
            # Enable RLS
            statements.append(f"ALTER TABLE {entity.table_name} ENABLE ROW LEVEL SECURITY;")
            
            # Create tenant isolation policy
            policy_name = f"tenant_isolation_{entity.table_name}"
            policy_ddl = (
                f"CREATE POLICY {policy_name} ON {entity.table_name} "
                f"FOR ALL TO PUBLIC "
                f"USING (tenant_id = current_setting('app.current_tenant_id')::uuid);"
            )
            statements.append(policy_ddl)
        
        return statements
    
    def _get_sql_type(self, attr: EntityAttribute) -> str:
        """Convert attribute data type to SQL type"""
        
        type_mappings = {
            DataType.UUID: "UUID",
            DataType.STRING: f"VARCHAR({attr.max_length or 255})",
            DataType.TEXT: "TEXT",
            DataType.INTEGER: "INTEGER",
            DataType.BIGINT: "BIGINT",
            DataType.DECIMAL: f"DECIMAL({attr.precision or 15},{attr.scale or 2})",
            DataType.BOOLEAN: "BOOLEAN",
            DataType.TIMESTAMP: "TIMESTAMP WITH TIME ZONE",
            DataType.DATE: "DATE",
            DataType.JSON: "JSONB",
            DataType.ARRAY: "TEXT[]"
        }
        
        return type_mappings.get(attr.data_type, "TEXT")
    
    def get_entity_summary(self) -> Dict[str, Any]:
        """Get summary of all entities"""
        
        summary = {
            'total_entities': len(self.entities),
            'entities_by_type': {},
            'multi_tenant_entities': 0,
            'total_attributes': 0,
            'total_relationships': 0,
            'pii_attributes': 0,
            'entities': {}
        }
        
        for entity_name, entity in self.entities.items():
            entity_type = entity.entity_type.value
            
            if entity_type not in summary['entities_by_type']:
                summary['entities_by_type'][entity_type] = 0
            summary['entities_by_type'][entity_type] += 1
            
            if entity.multi_tenant:
                summary['multi_tenant_entities'] += 1
            
            summary['total_attributes'] += len(entity.attributes)
            summary['total_relationships'] += len(entity.relationships)
            
            pii_count = sum(1 for attr in entity.attributes if attr.pii_classification)
            summary['pii_attributes'] += pii_count
            
            summary['entities'][entity_name] = {
                'table_name': entity.table_name,
                'entity_type': entity_type,
                'attributes_count': len(entity.attributes),
                'relationships_count': len(entity.relationships),
                'constraints_count': len(entity.constraints),
                'multi_tenant': entity.multi_tenant,
                'pii_attributes': pii_count
            }
        
        return summary

# Global ontology mapper
revops_ontology_mapper = RevOpsOntologyMapper()
