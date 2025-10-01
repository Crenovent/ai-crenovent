"""
Fabric Query Operator - Enhanced query operator with Microsoft Fabric integration
Provides access to real Salesforce CRM data for RBA workflows
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base import BaseOperator, OperatorContext, OperatorResult

logger = logging.getLogger(__name__)

class FabricQueryOperator(BaseOperator):
    """
    Enhanced query operator that can access both PostgreSQL and Microsoft Fabric
    Provides real CRM data for RBA agent workflows
    """
    
    def __init__(self, pool_manager=None):
        super().__init__()
        self.pool_manager = pool_manager
        self.fabric_service = None
        
    async def initialize(self):
        """Initialize the Fabric service if available"""
        try:
            if self.pool_manager:
                self.fabric_service = self.pool_manager.get_fabric_service()
                if not self.fabric_service:
                    # Try to initialize Fabric service
                    await self.pool_manager._initialize_fabric_service()
                    self.fabric_service = self.pool_manager.get_fabric_service()
                    
            logger.info("✅ Fabric Query Operator initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Fabric service not available: {e}")
            self.fabric_service = None

    async def execute(self, context: OperatorContext, params: Dict[str, Any]) -> OperatorResult:
        """
        Execute query with Fabric integration
        
        Params:
        - source: "postgresql" | "fabric" | "auto" (default: auto)
        - query_type: "opportunities" | "accounts" | "pipeline" | "forecast" | "custom"
        - filters: Dict with filter criteria
        - tenant_context: Tenant isolation parameters
        """
        
        try:
            source = params.get('source', 'auto')
            query_type = params.get('query_type', 'opportunities')
            filters = params.get('filters', {})
            
            logger.info(f"🔍 Fabric Query: {query_type} from {source}")
            
            # Determine data source
            if source == 'auto':
                # Use Fabric for CRM data, PostgreSQL for planning data
                source = 'fabric' if query_type in ['opportunities', 'accounts', 'contacts', 'activities'] else 'postgresql'
            
            # Execute query based on source
            if source == 'fabric' and self.fabric_service:
                result = await self._execute_fabric_query(context, query_type, filters)
            else:
                result = await self._execute_postgresql_query(context, query_type, filters)
            
            result_obj = OperatorResult(
                success=True,
                output_data={
                    "data": result.get('data', []),
                    "source": source,
                    "query_type": query_type,
                    "row_count": result.get('row_count', 0),
                    "execution_time_ms": result.get('execution_time_ms', 0)
                }
            )
            # Add governance evidence
            result_obj.add_evidence("data_source", source)
            result_obj.add_evidence("query_type", query_type)
            result_obj.add_evidence("tenant_id", context.tenant_id)
            result_obj.add_evidence("user_id", context.user_id)
            result_obj.add_evidence("filters_applied", filters)
            result_obj.add_evidence("timestamp", datetime.utcnow().isoformat())
            return result_obj
            
        except Exception as e:
            logger.error(f"❌ Fabric query error: {e}")
            result_obj = OperatorResult(
                success=False,
                output_data={
                    "error": str(e),
                    "fallback_data": await self._get_fallback_data(query_type)
                }
            )
            # Add governance evidence
            result_obj.add_evidence("error", str(e))
            result_obj.add_evidence("fallback_used", True)
            result_obj.add_evidence("timestamp", datetime.utcnow().isoformat())
            return result_obj

    async def _execute_fabric_query(self, context: OperatorContext, query_type: str, filters: Dict) -> Dict:
        """Execute query against Microsoft Fabric"""
        
        try:
            start_time = datetime.now()
            
            if query_type == 'opportunities':
                query = self._build_opportunities_query(filters, context.tenant_id)
            elif query_type == 'accounts':
                query = self._build_accounts_query(filters, context.tenant_id)
            elif query_type == 'pipeline':
                query = self._build_pipeline_query(filters, context.tenant_id)
            elif query_type == 'forecast':
                query = self._build_forecast_query(filters, context.tenant_id)
            else:
                query = filters.get('custom_query', 'SELECT 1 as test')
            
            # Execute query through Fabric service
            fabric_result = await self.fabric_service.execute_query(query)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if fabric_result.success:
                logger.info(f"✅ Fabric query successful: {fabric_result.row_count} rows in {execution_time:.2f}ms")
                return {
                    "data": fabric_result.data,
                    "row_count": fabric_result.row_count,
                    "execution_time_ms": execution_time
                }
            else:
                raise Exception(fabric_result.error_message)
                
        except Exception as e:
            logger.error(f"❌ Fabric query failed: {e}")
            raise

    async def _execute_postgresql_query(self, context: OperatorContext, query_type: str, filters: Dict) -> Dict:
        """Execute query against PostgreSQL"""
        
        try:
            start_time = datetime.now()
            
            if query_type == 'planning':
                query = self._build_planning_query(filters, context.tenant_id)
            elif query_type == 'users':
                query = self._build_users_query(filters, context.tenant_id)
            elif query_type == 'templates':
                query = self._build_templates_query(filters, context.tenant_id)
            else:
                query = filters.get('custom_query', 'SELECT 1 as test')
                
            async with self.pool_manager.get_connection() as conn:
                results = await conn.fetch(query)
                
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            data = [dict(row) for row in results]
            
            logger.info(f"✅ PostgreSQL query successful: {len(data)} rows in {execution_time:.2f}ms")
            
            return {
                "data": data,
                "row_count": len(data),
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL query failed: {e}")
            raise

    def _build_opportunities_query(self, filters: Dict, tenant_id: str) -> str:
        """Build Salesforce opportunities query for Fabric"""
        
        base_query = """
        SELECT 
            o.Id as opportunity_id,
            o.Name as opportunity_name,
            o.AccountId as account_id,
            a.Name as account_name,
            o.StageName as stage,
            o.Amount as amount,
            o.Probability as probability,
            o.CloseDate as close_date,
            o.CreatedDate as created_date,
            o.LastModifiedDate as modified_date,
            o.OwnerId as owner_id,
            u.Name as owner_name,
            o.Type as opportunity_type,
            o.LeadSource as lead_source,
            a.Industry as account_industry,
            a.AnnualRevenue as account_revenue
        FROM Opportunity o
        LEFT JOIN Account a ON o.AccountId = a.Id
        LEFT JOIN User u ON o.OwnerId = u.Id
        WHERE 1=1
        """
        
        # Add filters
        conditions = []
        
        if filters.get('stage'):
            if isinstance(filters['stage'], list):
                stages = "', '".join(filters['stage'])
                conditions.append(f"o.StageName IN ('{stages}')")
            else:
                conditions.append(f"o.StageName = '{filters['stage']}'")
        
        if filters.get('owner_id'):
            conditions.append(f"o.OwnerId = '{filters['owner_id']}'")
            
        if filters.get('account_id'):
            conditions.append(f"o.AccountId = '{filters['account_id']}'")
            
        if filters.get('close_date_from'):
            conditions.append(f"o.CloseDate >= '{filters['close_date_from']}'")
            
        if filters.get('close_date_to'):
            conditions.append(f"o.CloseDate <= '{filters['close_date_to']}'")
            
        if filters.get('min_amount'):
            conditions.append(f"o.Amount >= {filters['min_amount']}")
        
        # Add tenant isolation if available
        if filters.get('tenant_filter'):
            conditions.append(filters['tenant_filter'])
        
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
            
        # Add ordering and limits
        base_query += " ORDER BY o.LastModifiedDate DESC"
        
        if filters.get('limit'):
            base_query += f" LIMIT {filters['limit']}"
        else:
            base_query += " LIMIT 1000"  # Default limit
            
        return base_query

    def _build_accounts_query(self, filters: Dict, tenant_id: str) -> str:
        """Build Salesforce accounts query for Fabric"""
        
        base_query = """
        SELECT 
            a.Id as account_id,
            a.Name as account_name,
            a.Type as account_type,
            a.Industry as industry,
            a.AnnualRevenue as annual_revenue,
            a.NumberOfEmployees as employee_count,
            a.OwnerId as owner_id,
            u.Name as owner_name,
            a.CreatedDate as created_date,
            a.LastModifiedDate as modified_date,
            a.BillingCountry as country,
            a.BillingState as state,
            a.Website as website
        FROM Account a
        LEFT JOIN User u ON a.OwnerId = u.Id
        WHERE 1=1
        """
        
        # Add filters
        conditions = []
        
        if filters.get('account_type'):
            conditions.append(f"a.Type = '{filters['account_type']}'")
            
        if filters.get('industry'):
            conditions.append(f"a.Industry = '{filters['industry']}'")
            
        if filters.get('owner_id'):
            conditions.append(f"a.OwnerId = '{filters['owner_id']}'")
            
        if filters.get('min_revenue'):
            conditions.append(f"a.AnnualRevenue >= {filters['min_revenue']}")
        
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
            
        base_query += " ORDER BY a.LastModifiedDate DESC"
        
        if filters.get('limit'):
            base_query += f" LIMIT {filters['limit']}"
        else:
            base_query += " LIMIT 500"
            
        return base_query

    def _build_pipeline_query(self, filters: Dict, tenant_id: str) -> str:
        """Build pipeline analysis query for Fabric"""
        
        query = """
        SELECT 
            o.StageName as stage,
            COUNT(*) as opportunity_count,
            SUM(o.Amount) as total_amount,
            AVG(o.Amount) as avg_amount,
            AVG(o.Probability) as avg_probability,
            MIN(o.CloseDate) as earliest_close,
            MAX(o.CloseDate) as latest_close
        FROM Opportunity o
        WHERE o.IsClosed = false
        """
        
        # Add date filters for current quarter/period
        if filters.get('close_date_from'):
            query += f" AND o.CloseDate >= '{filters['close_date_from']}'"
        if filters.get('close_date_to'):
            query += f" AND o.CloseDate <= '{filters['close_date_to']}'"
            
        query += """
        GROUP BY o.StageName
        ORDER BY SUM(o.Amount) DESC
        """
        
        return query

    def _build_forecast_query(self, filters: Dict, tenant_id: str) -> str:
        """Build forecast data query for Fabric"""
        
        query = """
        SELECT 
            DATEPART(YEAR, o.CloseDate) as year,
            DATEPART(QUARTER, o.CloseDate) as quarter,
            DATEPART(MONTH, o.CloseDate) as month,
            o.StageName as stage,
            COUNT(*) as opportunity_count,
            SUM(o.Amount) as total_amount,
            SUM(o.Amount * o.Probability / 100) as weighted_amount,
            AVG(o.Probability) as avg_probability
        FROM Opportunity o
        WHERE o.CloseDate >= DATEADD(MONTH, -12, GETDATE())
        """
        
        if filters.get('owner_id'):
            query += f" AND o.OwnerId = '{filters['owner_id']}'"
            
        query += """
        GROUP BY 
            DATEPART(YEAR, o.CloseDate),
            DATEPART(QUARTER, o.CloseDate), 
            DATEPART(MONTH, o.CloseDate),
            o.StageName
        ORDER BY year DESC, quarter DESC, month DESC
        """
        
        return query

    def _build_planning_query(self, filters: Dict, tenant_id: str) -> str:
        """Build planning data query for PostgreSQL"""
        
        query = f"""
        SELECT 
            plan_id,
            plan_name,
            account_id,
            account_owner,
            annual_revenue,
            account_tier,
            status,
            created_at,
            created_by_user_id
        FROM strategic_account_plans
        WHERE tenant_id = '{tenant_id}'
        """
        
        if filters.get('status'):
            query += f" AND status = '{filters['status']}'"
            
        if filters.get('user_id'):
            query += f" AND created_by_user_id = {filters['user_id']}"
            
        query += " ORDER BY created_at DESC LIMIT 100"
        
        return query

    def _build_users_query(self, filters: Dict, tenant_id: str) -> str:
        """Build users query for PostgreSQL"""
        
        query = f"""
        SELECT 
            u.user_id,
            u.email,
            ur.role,
            ur.segment,
            ur.region,
            ur.crux_view,
            ur.crux_create
        FROM users u
        LEFT JOIN users_role ur ON u.user_id = ur.user_id
        WHERE u.tenant_id = '{tenant_id}'
        """
        
        if filters.get('role'):
            query += f" AND ur.role = '{filters['role']}'"
            
        query += " ORDER BY u.user_id LIMIT 50"
        
        return query

    def _build_templates_query(self, filters: Dict, tenant_id: str) -> str:
        """Build templates query for PostgreSQL"""
        
        query = f"""
        SELECT 
            template_id,
            template_name,
            description,
            status,
            version,
            created_at
        FROM account_planning_templates
        WHERE tenant_id = '{tenant_id}'
        """
        
        if filters.get('status'):
            query += f" AND status = '{filters['status']}'"
            
        query += " ORDER BY created_at DESC"
        
        return query

    async def _get_fallback_data(self, query_type: str) -> List[Dict]:
        """Provide fallback data when queries fail"""
        
        fallback_data = {
            'opportunities': [
                {
                    "opportunity_id": "sample_001",
                    "opportunity_name": "Sample Opportunity",
                    "account_name": "Sample Account",
                    "stage": "Negotiation",
                    "amount": 100000,
                    "probability": 75,
                    "close_date": "2024-12-31"
                }
            ],
            'accounts': [
                {
                    "account_id": "sample_acc_001",
                    "account_name": "Sample Account",
                    "industry": "Technology",
                    "annual_revenue": 5000000,
                    "account_type": "Customer"
                }
            ],
            'pipeline': [
                {
                    "stage": "Prospecting",
                    "opportunity_count": 10,
                    "total_amount": 500000,
                    "avg_probability": 25
                }
            ]
        }
        
        return fallback_data.get(query_type, [])

    async def get_schema_info(self) -> Dict[str, Any]:
        """Get available schema information for query building"""
        
        return {
            "fabric_sources": [
                {
                    "name": "opportunities",
                    "description": "Salesforce Opportunity data",
                    "key_fields": ["Id", "Name", "AccountId", "StageName", "Amount", "CloseDate", "Probability"]
                },
                {
                    "name": "accounts", 
                    "description": "Salesforce Account data",
                    "key_fields": ["Id", "Name", "Type", "Industry", "AnnualRevenue", "OwnerId"]
                },
                {
                    "name": "contacts",
                    "description": "Salesforce Contact data", 
                    "key_fields": ["Id", "Name", "AccountId", "Email", "Title"]
                },
                {
                    "name": "users",
                    "description": "Salesforce User data",
                    "key_fields": ["Id", "Name", "Email", "UserRole"]
                }
            ],
            "postgresql_sources": [
                {
                    "name": "strategic_account_plans",
                    "description": "Account planning data",
                    "key_fields": ["plan_id", "plan_name", "account_id", "status", "created_by_user_id"]
                },
                {
                    "name": "users",
                    "description": "Application users",
                    "key_fields": ["user_id", "email", "tenant_id"]
                }
            ]
        }
