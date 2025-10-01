"""
Database Service for Planning AI
Handles PostgreSQL and Fabric data access with RBAC enforcement
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncpg
import pandas as pd
from dataclasses import dataclass

@dataclass
class UserContext:
    user_id: int
    tenant_id: int
    role: str
    hierarchy_level: str
    assigned_level_id: str
    permissions: Dict[str, bool]
    segment: str
    region: str
    area: str
    district: str
    territory: str

class DatabaseService:
    """
    Optimized database service with shared connection pooling
    Uses centralized ConnectionPoolManager for efficient resource management
    """
    
    def __init__(self):
        from .connection_pool_manager import pool_manager
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.role_access_map = {
            'admin': ['global', 'region', 'country', 'area', 'segment', 'state', 'executive'],
            'cro': ['global', 'region', 'country', 'area', 'segment', 'state', 'executive'],
            'sales_leader': ['region', 'country', 'area', 'segment', 'state', 'executive'],
            'sales': ['segment', 'state', 'executive'],
            'sales_engineer': ['area', 'segment', 'state', 'executive'],
            'customer_success': ['segment', 'state', 'executive'],
            'default': ['executive']
        }
    
    async def initialize(self):
        """Initialize using shared connection pool manager"""
        try:
            if not self.initialized:
                # Ensure pool manager is initialized
                if not self.pool_manager._initialized:
                    await self.pool_manager.initialize()
                
                self.initialized = True
                self.logger.info("✅ Database service initialized with shared connection pool")
            
        except Exception as e:
            self.logger.error(f"❌ Database service initialization error: {str(e)}")
            raise
    
    @property
    def postgres_pool(self):
        """Access to shared PostgreSQL pool"""
        return self.pool_manager.postgres_pool
    
    @property
    def fabric_service(self):
        """Access to shared Fabric service"""
        return self.pool_manager.get_fabric_service()
    
    async def get_user_context(self, user_id: int, tenant_id: int) -> UserContext:
        """Get comprehensive user context with RBAC permissions"""
        try:
            # Use a simple approach to avoid event loop conflicts
            if not self.pool_manager.postgres_pool:
                raise RuntimeError("PostgreSQL pool not initialized")
            
            # Use timeout to prevent hanging connections
            conn = await asyncio.wait_for(
                self.pool_manager.postgres_pool.acquire(),
                timeout=5.0  # 5 second timeout
            )
            
            try:
                user_result = await conn.fetchrow("""
                    SELECT 
                        u.user_id, u.username, u.email, u.profile, u.reports_to, u.tenant_id,
                        ur.role, ur.segment, ur.region, ur.area, ur.district, ur.territory,
                        ur.crux_view, ur.crux_create, ur.crux_edit, ur.crux_assign, ur.crux_close,
                        ur.role_function, ur.business_function
                    FROM users u
                    LEFT JOIN users_role ur ON u.user_id = ur.user_id
                    WHERE u.user_id = $1 AND u.tenant_id = $2 AND u.is_activated = true
                """, user_id, tenant_id)
            finally:
                await self.pool_manager.postgres_pool.release(conn)
            
            if not user_result:
                raise ValueError(f"User {user_id} not found or not activated")
            
            # Parse user profile for additional context
            profile = json.loads(user_result['profile'] or '{}')
            
            # Determine hierarchy level and permissions
            hierarchy_level = self.determine_hierarchy_level(user_result)
            permissions = {
                'view': user_result['crux_view'] == 'true' if user_result['crux_view'] else False,
                'create': user_result['crux_create'] == 'true' if user_result['crux_create'] else False,
                'edit': user_result['crux_edit'] == 'true' if user_result['crux_edit'] else False,
                'assign': user_result['crux_assign'] == 'true' if user_result['crux_assign'] else False,
                'close': user_result['crux_close'] == 'true' if user_result['crux_close'] else False
            }
            
            return UserContext(
                user_id=user_id,
                tenant_id=tenant_id,
                role=user_result['role'] or 'default',
                hierarchy_level=hierarchy_level,
                assigned_level_id=profile.get('assignedLevelId', ''),
                permissions=permissions,
                segment=user_result['segment'] or '',
                region=user_result['region'] or '',
                area=user_result['area'] or '',
                district=user_result['district'] or '',
                territory=user_result['territory'] or ''
            )
                
        except Exception as e:
            self.logger.error(f"Error getting user context: {str(e)}")
            raise
    
    def determine_hierarchy_level(self, user_data: Dict) -> str:
        """Determine user's hierarchy level based on role and geographic assignment"""
        role = user_data.get('role', 'default')
        
        # Check geographic assignments to determine level
        if user_data.get('territory'):
            return 'executive'
        elif user_data.get('district'):
            return 'state'
        elif user_data.get('area'):
            return 'segment'
        elif user_data.get('region'):
            return 'area'
        else:
            # Fallback to role-based level
            if role in ['admin', 'cro']:
                return 'global'
            elif role in ['sales_leader']:
                return 'region'
            else:
                return 'executive'
    
    async def get_user_strategic_plans(self, user_context: UserContext, limit: int = 50) -> List[Dict]:
        """Get user's strategic account plans with RBAC filtering"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Build RBAC-compliant query based on user's permissions and hierarchy
                base_query = """
                SELECT 
                    sap.*,
                    u.username as created_by_username,
                    apt.template_name
                FROM strategic_account_plans sap
                LEFT JOIN users u ON sap.created_by_user_id = u.user_id
                LEFT JOIN account_planning_templates apt ON sap.template_id = apt.template_id
                WHERE sap.tenant_id = $1
                """
                
                params = [user_context.tenant_id]
                
                # Apply RBAC filtering
                if not user_context.permissions.get('view'):
                    base_query += " AND sap.created_by_user_id = $2"
                    params.append(user_context.user_id)
                elif user_context.hierarchy_level == 'executive':
                    # Executive level can only see their own plans
                    base_query += " AND sap.created_by_user_id = $2"
                    params.append(user_context.user_id)
                elif user_context.hierarchy_level in ['segment', 'state']:
                    # Segment/State can see plans in their geographic area
                    subordinate_users = await self.get_subordinate_users(user_context)
                    if subordinate_users:
                        user_ids = [user_context.user_id] + subordinate_users
                        placeholders = ','.join([f'${i+2}' for i in range(len(user_ids))])
                        base_query += f" AND sap.created_by_user_id IN ({placeholders})"
                        params.extend(user_ids)
                
                base_query += " ORDER BY sap.created_at DESC LIMIT $" + str(len(params) + 1)
                params.append(limit)
                
                plans = await conn.fetch(base_query, *params)
                
                return [dict(plan) for plan in plans]
                
        except Exception as e:
            self.logger.error(f"Error getting strategic plans: {str(e)}")
            return []
    
    async def get_subordinate_users(self, user_context: UserContext) -> List[int]:
        """Get list of subordinate user IDs based on hierarchy"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Get users who report to this user (direct and indirect)
                hierarchy_query = """
                WITH RECURSIVE user_hierarchy AS (
                    -- Start with direct reports
                    SELECT user_id, reports_to, 1 as level
                    FROM users 
                    WHERE reports_to = $1 AND tenant_id = $2 AND is_activated = true
                    
                    UNION ALL
                    
                    -- Get indirect reports
                    SELECT u.user_id, u.reports_to, uh.level + 1
                    FROM users u
                    JOIN user_hierarchy uh ON u.reports_to = uh.user_id
                    WHERE u.tenant_id = $2 AND u.is_activated = true AND uh.level < 5
                )
                SELECT DISTINCT user_id FROM user_hierarchy
                """
                
                subordinates = await conn.fetch(hierarchy_query, user_context.user_id, user_context.tenant_id)
                return [s['user_id'] for s in subordinates]
                
        except Exception as e:
            self.logger.error(f"Error getting subordinate users: {str(e)}")
            return []
    
    async def get_account_data_from_fabric(self, account_id: str, user_context: UserContext) -> Dict:
        """Get comprehensive account data from Fabric with RBAC"""
        try:
            # Check if user has permission to view this account
            if not await self.can_user_access_account(account_id, user_context):
                raise PermissionError(f"User {user_context.user_id} cannot access account {account_id}")
            
            # Get account data from Fabric (only safe columns)
            account_query = """
            SELECT 
                Id, Name, Type, Industry, AnnualRevenue, NumberOfEmployees,
                BillingCountry, BillingState,
                Phone, Website, Description, OwnerId, 
                CreatedDate, LastModifiedDate
            FROM dbo.accounts 
            WHERE Id = ?
            """
            
            account_data = await self.fabric_service.execute_query(account_query, [account_id])
            
            if not account_data:
                return {}
            
            account = account_data[0] if account_data else {}
            
            # Get related opportunities
            opportunities_query = """
            SELECT 
                Id, Name, AccountId, StageName, Amount, Probability, 
                CloseDate, Type, LeadSource, OwnerId, CreatedDate,
                ForecastCategory, HasOpportunityLineItem
            FROM dbo.opportunities 
            WHERE AccountId = ?
            ORDER BY CreatedDate DESC
            """
            
            opportunities = await self.fabric_service.execute_query(opportunities_query, [account_id])
            
            # Get account contacts
            contacts_query = """
            SELECT 
                Id, FirstName, LastName, Title, Email, Phone,
                Department, Level__c, AccountId
            FROM dbo.contacts 
            WHERE AccountId = ?
            ORDER BY LastName
            """
            
            contacts = await self.fabric_service.execute_query(contacts_query, [account_id])
            
            return {
                'account': account,
                'opportunities': opportunities or [],
                'contacts': contacts or [],
                'total_pipeline': sum(float(opp.get('Amount', 0) or 0) for opp in opportunities or []),
                'open_opportunities': len([opp for opp in opportunities or [] if not opp.get('IsClosed')]),
                'last_activity': max([opp.get('CreatedDate') for opp in opportunities or []], default=None)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account data from Fabric: {str(e)}")
            return {}
    
    async def can_user_access_account(self, account_id: str, user_context: UserContext) -> bool:
        """Check if user can access specific account based on RBAC"""
        try:
            # Admin and CRO can access all accounts
            if user_context.role in ['admin', 'cro']:
                return True
            
            # Get account owner information from Fabric
            owner_query = """
            SELECT a.OwnerId, u.Name as OwnerName, u.UserRoleId
            FROM dbo.accounts a
            LEFT JOIN dbo.users u ON a.OwnerId = u.Id
            WHERE a.Id = ?
            """
            
            owner_data = await self.fabric_service.execute_query(owner_query, [account_id])
            
            if not owner_data:
                return False
            
            owner_info = owner_data[0]
            
            # Check if user is the account owner
            if owner_info.get('OwnerId') == user_context.user_id:
                return True
            
            # Check if account owner is subordinate to current user
            subordinates = await self.get_subordinate_users(user_context)
            if owner_info.get('OwnerId') in subordinates:
                return True
            
            # Geographic-based access control
            # This would require mapping Salesforce territories to our hierarchy
            # For now, default to role-based access
            
            return user_context.permissions.get('view', False)
            
        except Exception as e:
            self.logger.error(f"Error checking account access: {str(e)}")
            return False
    
    async def create_strategic_plan(self, plan_data: Dict, user_context: UserContext) -> Dict:
        """Create a new strategic account plan with RBAC validation"""
        try:
            # Validate user permissions
            if not user_context.permissions.get('create'):
                raise PermissionError("User does not have permission to create strategic plans")
            
            # Validate account access
            account_id = plan_data.get('account_id')
            if account_id and not await self.can_user_access_account(account_id, user_context):
                raise PermissionError(f"User cannot create plan for account {account_id}")
            
            async with self.postgres_pool.acquire() as conn:
                # Insert strategic plan
                insert_query = """
                INSERT INTO strategic_account_plans (
                    plan_name, account_id, template_id, account_owner, industry,
                    annual_revenue, account_tier, region_territory, customer_since,
                    short_term_goals, long_term_goals, revenue_growth_target,
                    product_penetration_goals, customer_success_metrics,
                    key_opportunities, cross_sell_upsell_potential, known_risks,
                    risk_mitigation_strategies, communication_cadence, status,
                    created_by_user_id, tenant_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
                ) RETURNING plan_id, created_at
                """
                
                plan_result = await conn.fetchrow(
                    insert_query,
                    plan_data.get('plan_name'),
                    plan_data.get('account_id'),
                    plan_data.get('template_id'),
                    plan_data.get('account_owner'),
                    plan_data.get('industry'),
                    plan_data.get('annual_revenue'),
                    plan_data.get('account_tier'),
                    plan_data.get('region_territory'),
                    plan_data.get('customer_since'),
                    plan_data.get('short_term_goals'),
                    plan_data.get('long_term_goals'),
                    plan_data.get('revenue_growth_target'),
                    plan_data.get('product_penetration_goals'),
                    plan_data.get('customer_success_metrics'),
                    plan_data.get('key_opportunities'),
                    plan_data.get('cross_sell_upsell_potential'),
                    plan_data.get('known_risks'),
                    plan_data.get('risk_mitigation_strategies'),
                    plan_data.get('communication_cadence'),
                    plan_data.get('status', 'Draft'),
                    user_context.user_id,
                    user_context.tenant_id
                )
                
                plan_id = plan_result['plan_id']
                
                # Add stakeholders if provided
                stakeholders = plan_data.get('stakeholders', [])
                for stakeholder in stakeholders:
                    await conn.execute("""
                        INSERT INTO plan_stakeholders (
                            plan_id, name, role, influence_level, relationship_status, tenant_id
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """, plan_id, stakeholder.get('name'), stakeholder.get('role'),
                        stakeholder.get('influence_level'), stakeholder.get('relationship_status'),
                        user_context.tenant_id)
                
                # Add activities if provided
                activities = plan_data.get('activities', [])
                for activity in activities:
                    await conn.execute("""
                        INSERT INTO plan_activities (
                            plan_id, activity_title, planned_date, activity_type, 
                            description, status, tenant_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, plan_id, activity.get('activity_title'), activity.get('planned_date'),
                        activity.get('activity_type'), activity.get('description'),
                        activity.get('status', 'Planned'), user_context.tenant_id)
                
                return {
                    'plan_id': plan_id,
                    'created_at': plan_result['created_at'],
                    'success': True,
                    'message': 'Strategic plan created successfully'
                }
                
        except Exception as e:
            self.logger.error(f"Error creating strategic plan: {str(e)}")
            raise
    
    async def update_strategic_plan(self, plan_id: int, plan_data: Dict, user_context: UserContext) -> Dict:
        """Update existing strategic plan with RBAC validation"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Check if user can edit this plan
                plan_check = await conn.fetchrow("""
                    SELECT created_by_user_id, status FROM strategic_account_plans 
                    WHERE plan_id = $1 AND tenant_id = $2
                """, plan_id, user_context.tenant_id)
                
                if not plan_check:
                    raise ValueError(f"Plan {plan_id} not found")
                
                # Check permissions
                can_edit = (
                    user_context.permissions.get('edit', False) or
                    plan_check['created_by_user_id'] == user_context.user_id or
                    user_context.role in ['admin', 'cro']
                )
                
                if not can_edit:
                    raise PermissionError("User does not have permission to edit this plan")
                
                # Build dynamic update query
                update_fields = []
                params = []
                param_count = 1
                
                updatable_fields = [
                    'plan_name', 'account_owner', 'industry', 'annual_revenue',
                    'account_tier', 'region_territory', 'customer_since',
                    'short_term_goals', 'long_term_goals', 'revenue_growth_target',
                    'product_penetration_goals', 'customer_success_metrics',
                    'key_opportunities', 'cross_sell_upsell_potential', 'known_risks',
                    'risk_mitigation_strategies', 'communication_cadence', 'status'
                ]
                
                for field in updatable_fields:
                    if field in plan_data:
                        update_fields.append(f"{field} = ${param_count}")
                        params.append(plan_data[field])
                        param_count += 1
                
                if update_fields:
                    update_fields.append(f"updated_at = ${param_count}")
                    params.append(datetime.now())
                    param_count += 1
                    
                    params.extend([plan_id, user_context.tenant_id])
                    
                    update_query = f"""
                        UPDATE strategic_account_plans 
                        SET {', '.join(update_fields)}
                        WHERE plan_id = ${param_count-1} AND tenant_id = ${param_count}
                        RETURNING updated_at
                    """
                    
                    result = await conn.fetchrow(update_query, *params)
                    
                    return {
                        'plan_id': plan_id,
                        'updated_at': result['updated_at'],
                        'success': True,
                        'message': 'Strategic plan updated successfully'
                    }
                else:
                    return {
                        'plan_id': plan_id,
                        'success': True,
                        'message': 'No changes to update'
                    }
                
        except Exception as e:
            self.logger.error(f"Error updating strategic plan: {str(e)}")
            raise
    
    async def get_account_planning_templates(self, user_context: UserContext) -> List[Dict]:
        """Get available account planning templates"""
        try:
            async with self.postgres_pool.acquire() as conn:
                templates_query = """
                SELECT template_id, template_name, description, template_data, 
                       created_by_user_id, created_at, is_active
                FROM account_planning_templates 
                WHERE tenant_id = $1 AND is_active = true
                ORDER BY template_name
                """
                
                templates = await conn.fetch(templates_query, user_context.tenant_id)
                return [dict(template) for template in templates]
                
        except Exception as e:
            self.logger.error(f"Error getting templates: {str(e)}")
            return []
    
    async def get_intelligent_suggestions(self, user_context: UserContext, account_id: str = None) -> Dict:
        """Get intelligent suggestions based on user's historical data and account context"""
        try:
            suggestions = {}
            
            # Get user's historical plans for pattern analysis
            historical_plans = await self.get_user_strategic_plans(user_context, limit=10)
            
            # Analyze patterns in historical data
            if historical_plans:
                suggestions['revenue_targets'] = await self.analyze_revenue_patterns(historical_plans)
                suggestions['common_goals'] = await self.analyze_goal_patterns(historical_plans)
                suggestions['risk_patterns'] = await self.analyze_risk_patterns(historical_plans)
            
            # If account_id provided, get account-specific suggestions
            if account_id:
                account_data = await self.get_account_data_from_fabric(account_id, user_context)
                if account_data:
                    suggestions['account_specific'] = await self.generate_account_suggestions(account_data)
            
            # Get industry benchmarks and best practices
            suggestions['industry_benchmarks'] = await self.get_industry_benchmarks(user_context)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting intelligent suggestions: {str(e)}")
            return {}
    
    async def analyze_revenue_patterns(self, historical_plans: List[Dict]) -> Dict:
        """Analyze revenue growth patterns from historical plans"""
        try:
            revenue_targets = [
                float(plan.get('revenue_growth_target') or 0) 
                for plan in historical_plans 
                if plan.get('revenue_growth_target')
            ]
            
            if revenue_targets:
                avg_target = sum(revenue_targets) / len(revenue_targets)
                return {
                    'average_target': round(avg_target, 2),
                    'suggested_range': f"{round(avg_target * 0.8, 2)}-{round(avg_target * 1.2, 2)}%",
                    'confidence': 0.85 if len(revenue_targets) >= 3 else 0.6
                }
            
            return {'suggested_range': '15-25%', 'confidence': 0.3}
            
        except Exception as e:
            self.logger.error(f"Error analyzing revenue patterns: {str(e)}")
            return {}
    
    async def analyze_goal_patterns(self, historical_plans: List[Dict]) -> List[str]:
        """Extract common goals from historical plans"""
        try:
            all_goals = []
            for plan in historical_plans:
                short_term = plan.get('short_term_goals', '') or ''
                long_term = plan.get('long_term_goals', '') or ''
                all_goals.extend([goal.strip() for goal in (short_term + ' ' + long_term).split('\n') if goal.strip()])
            
            # Simple frequency analysis
            goal_frequency = {}
            for goal in all_goals:
                goal_lower = goal.lower()
                goal_frequency[goal_lower] = goal_frequency.get(goal_lower, 0) + 1
            
            # Return most common goals
            common_goals = sorted(goal_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            return [goal[0] for goal in common_goals]
            
        except Exception as e:
            self.logger.error(f"Error analyzing goal patterns: {str(e)}")
            return []
    
    async def analyze_risk_patterns(self, historical_plans: List[Dict]) -> List[str]:
        """Extract common risks from historical plans"""
        try:
            all_risks = []
            for plan in historical_plans:
                risks = plan.get('known_risks', '') or ''
                all_risks.extend([risk.strip() for risk in risks.split('\n') if risk.strip()])
            
            # Simple frequency analysis
            risk_frequency = {}
            for risk in all_risks:
                risk_lower = risk.lower()
                risk_frequency[risk_lower] = risk_frequency.get(risk_lower, 0) + 1
            
            # Return most common risks
            common_risks = sorted(risk_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            return [risk[0] for risk in common_risks]
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk patterns: {str(e)}")
            return []
    
    async def generate_account_suggestions(self, account_data: Dict) -> Dict:
        """Generate account-specific suggestions based on Salesforce data"""
        try:
            suggestions = {}
            account = account_data.get('account', {})
            opportunities = account_data.get('opportunities', [])
            
            # Revenue-based suggestions
            annual_revenue = account.get('AnnualRevenue', 0) or 0
            if annual_revenue:
                if annual_revenue >= 10000000:  # $10M+
                    suggestions['account_tier'] = 'Enterprise'
                    suggestions['revenue_growth_target'] = '15-25%'
                elif annual_revenue >= 1000000:  # $1M+
                    suggestions['account_tier'] = 'Key Account'
                    suggestions['revenue_growth_target'] = '20-35%'
                else:
                    suggestions['account_tier'] = 'Growth Account'
                    suggestions['revenue_growth_target'] = '30-50%'
            
            # Opportunity-based suggestions
            if opportunities:
                total_pipeline = sum(float(opp.get('Amount', 0) or 0) for opp in opportunities)
                suggestions['current_pipeline'] = f"${total_pipeline:,.0f}"
                
                # Stage analysis
                stages = [opp.get('StageName', '') for opp in opportunities]
                if any('Closed Won' in stage for stage in stages):
                    suggestions['relationship_status'] = 'Strong'
                elif any('Negotiation' in stage or 'Proposal' in stage for stage in stages):
                    suggestions['relationship_status'] = 'Engaged'
                else:
                    suggestions['relationship_status'] = 'Developing'
            
            # Industry-specific suggestions
            industry = account.get('Industry', '')
            if industry:
                suggestions['industry_focus'] = industry
                if 'Technology' in industry or 'Software' in industry:
                    suggestions['key_opportunities'] = 'Digital transformation, cloud migration, AI/ML initiatives'
                elif 'Financial' in industry:
                    suggestions['key_opportunities'] = 'Regulatory compliance, risk management, customer experience'
                elif 'Healthcare' in industry:
                    suggestions['key_opportunities'] = 'Patient engagement, operational efficiency, compliance'
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating account suggestions: {str(e)}")
            return {}
    
    async def get_industry_benchmarks(self, user_context: UserContext) -> Dict:
        """Get industry benchmarks and best practices"""
        try:
            # This would typically come from external data sources or industry databases
            # For now, return common benchmarks
            benchmarks = {
                'revenue_growth': {
                    'technology': '20-40%',
                    'financial_services': '15-25%',
                    'healthcare': '10-20%',
                    'manufacturing': '10-15%',
                    'retail': '15-30%'
                },
                'best_practices': [
                    'Quarterly business reviews with key stakeholders',
                    'Regular executive engagement and relationship building',
                    'Cross-functional collaboration with customer success teams',
                    'Data-driven decision making and performance tracking',
                    'Proactive risk identification and mitigation planning'
                ]
            }
            
            return benchmarks
            
        except Exception as e:
            self.logger.error(f"Error getting industry benchmarks: {str(e)}")
            return {}
    
    # ===============================================================
    # INTELLIGENT QUERY METHODS FOR AI AGENTS
    # ===============================================================
    
    async def intelligent_query(self, question: str, user_context: UserContext) -> Dict[str, Any]:
        """Route queries intelligently based on content and user context"""
        try:
            question_lower = question.lower()
            
            # Determine query type and route appropriately
            if any(keyword in question_lower for keyword in ['plan', 'plans', 'strategic']):
                return await self.query_strategic_plans(question, user_context)
            elif any(keyword in question_lower for keyword in ['opportunity', 'opportunities', 'deal', 'deals']):
                return await self.query_opportunities(question, user_context)
            elif any(keyword in question_lower for keyword in ['account', 'accounts', 'customer', 'customers']):
                return await self.query_accounts(question, user_context)
            elif any(keyword in question_lower for keyword in ['revenue', 'target', 'growth', 'pipeline']):
                return await self.query_revenue_data(question, user_context)
            else:
                return await self.general_data_query(question, user_context)
                
        except Exception as e:
            self.logger.error(f"Error in intelligent query: {str(e)}")
            return {"error": str(e), "data": []}
    
    async def query_strategic_plans(self, question: str, user_context: UserContext) -> Dict[str, Any]:
        """Query strategic plans with intelligent filtering"""
        try:
            async with self.postgres_pool.acquire() as conn:
                if 'how many' in question.lower() or 'count' in question.lower():
                    # Count query
                    count_query = """
                    SELECT COUNT(*) as total_plans,
                           COUNT(CASE WHEN status = 'Draft' THEN 1 END) as draft_plans,
                           COUNT(CASE WHEN status = 'Submitted' THEN 1 END) as submitted_plans,
                           COUNT(CASE WHEN status = 'Approved' THEN 1 END) as approved_plans
                    FROM strategic_account_plans 
                    WHERE tenant_id = $1
                    """
                    
                    if user_context.hierarchy_level == 'executive':
                        count_query += " AND created_by_user_id = $2"
                        result = await conn.fetchrow(count_query, user_context.tenant_id, user_context.user_id)
                    else:
                        result = await conn.fetchrow(count_query, user_context.tenant_id)
                    
                    return {
                        "answer": f"There are {result['total_plans']} strategic plans total ({result['draft_plans']} draft, {result['submitted_plans']} submitted, {result['approved_plans']} approved)",
                        "data": [dict(result)],
                        "query_type": "count",
                        "confidence": 0.95
                    }
                
                elif 'recent' in question.lower() or 'latest' in question.lower():
                    # Recent plans query
                    plans = await self.get_user_strategic_plans(user_context, limit=5)
                    
                    plan_summaries = []
                    for plan in plans:
                        plan_summaries.append({
                            'plan_name': plan['plan_name'],
                            'status': plan['status'],
                            'created_at': plan['created_at'].strftime('%Y-%m-%d') if plan['created_at'] else 'Unknown',
                            'revenue_target': plan.get('revenue_growth_target', 'Not set')
                        })
                    
                    return {
                        "answer": f"Your {len(plans)} most recent strategic plans",
                        "data": plan_summaries,
                        "query_type": "recent_plans",
                        "confidence": 0.9
                    }
                
                else:
                    # General plans query
                    plans = await self.get_user_strategic_plans(user_context, limit=10)
                    return {
                        "answer": f"Found {len(plans)} strategic plans",
                        "data": plans,
                        "query_type": "general_plans",
                        "confidence": 0.8
                    }
                    
        except Exception as e:
            self.logger.error(f"Error querying strategic plans: {str(e)}")
            return {"error": str(e), "data": []}
    
    async def query_opportunities(self, question: str, user_context: UserContext) -> Dict[str, Any]:
        """Query opportunities from Fabric with intelligent filtering"""
        try:
            if not self.fabric_service:
                return {"error": "Fabric service not available", "data": []}
            
            if 'how many' in question.lower() or 'count' in question.lower():
                # Count opportunities
                count_query = """
                SELECT COUNT(*) as total_opportunities,
                       COUNT(CASE WHEN IsClosed = 0 THEN 1 END) as open_opportunities,
                       COUNT(CASE WHEN IsWon = 1 THEN 1 END) as won_opportunities,
                       SUM(CASE WHEN IsClosed = 0 THEN Amount ELSE 0 END) as open_pipeline
                FROM dbo.opportunities
                """
                
                result = await self.fabric_service.execute_query(count_query)
                
                if result:
                    data = result[0]
                    return {
                        "answer": f"There are {data['total_opportunities']} opportunities total ({data['open_opportunities']} open, {data['won_opportunities']} won). Open pipeline: ${data['open_pipeline']:,.0f}",
                        "data": [data],
                        "query_type": "opportunity_count",
                        "confidence": 0.95
                    }
            
            elif 'pipeline' in question.lower():
                # Pipeline analysis
                pipeline_query = """
                SELECT StageName, COUNT(*) as count, SUM(Amount) as total_amount
                FROM dbo.opportunities 
                WHERE IsClosed = 0
                GROUP BY StageName
                ORDER BY total_amount DESC
                """
                
                results = await self.fabric_service.execute_query(pipeline_query)
                
                if results:
                    total_pipeline = sum(r['total_amount'] or 0 for r in results)
                    return {
                        "answer": f"Current pipeline breakdown: ${total_pipeline:,.0f} total across {len(results)} stages",
                        "data": results,
                        "query_type": "pipeline_analysis",
                        "confidence": 0.9
                    }
            
            elif 'closing' in question.lower() or 'quarter' in question.lower():
                # Opportunities closing this quarter
                closing_query = """
                SELECT Id, Name, StageName, Amount, CloseDate, AccountId
                FROM dbo.opportunities 
                WHERE IsClosed = 0 
                  AND CloseDate >= DATEADD(quarter, DATEDIFF(quarter, 0, GETDATE()), 0)
                  AND CloseDate < DATEADD(quarter, DATEDIFF(quarter, 0, GETDATE()) + 1, 0)
                ORDER BY Amount DESC
                """
                
                results = await self.fabric_service.execute_query(closing_query)
                
                if results:
                    total_value = sum(r['Amount'] or 0 for r in results)
                    return {
                        "answer": f"{len(results)} opportunities closing this quarter worth ${total_value:,.0f}",
                        "data": results,
                        "query_type": "quarterly_opportunities",
                        "confidence": 0.9
                    }
            
            return {"answer": "Could not find specific opportunity data", "data": [], "confidence": 0.3}
            
        except Exception as e:
            self.logger.error(f"Error querying opportunities: {str(e)}")
            return {"error": str(e), "data": []}
    
    async def query_accounts(self, question: str, user_context: UserContext) -> Dict[str, Any]:
        """Query accounts from Fabric with intelligent filtering"""
        try:
            if not self.fabric_service:
                return {"error": "Fabric service not available", "data": []}
            
            if 'enterprise' in question.lower() or 'large' in question.lower():
                # Enterprise accounts
                query = """
                SELECT Id, Name, Industry, AnnualRevenue, NumberOfEmployees
                FROM dbo.accounts 
                WHERE AnnualRevenue >= 10000000
                ORDER BY AnnualRevenue DESC
                """
                
                results = await self.fabric_service.execute_query(query)
                
                if results:
                    total_revenue = sum(r['AnnualRevenue'] or 0 for r in results)
                    return {
                        "answer": f"Found {len(results)} enterprise accounts with combined revenue of ${total_revenue:,.0f}",
                        "data": results,
                        "query_type": "enterprise_accounts",
                        "confidence": 0.9
                    }
            
            elif 'industry' in question.lower():
                # Industry breakdown
                query = """
                SELECT Industry, COUNT(*) as account_count, AVG(AnnualRevenue) as avg_revenue
                FROM dbo.accounts 
                WHERE Industry IS NOT NULL
                GROUP BY Industry
                ORDER BY account_count DESC
                """
                
                results = await self.fabric_service.execute_query(query)
                
                if results:
                    return {
                        "answer": f"Account distribution across {len(results)} industries",
                        "data": results,
                        "query_type": "industry_breakdown",
                        "confidence": 0.9
                    }
            
            return {"answer": "Could not find specific account data", "data": [], "confidence": 0.3}
            
        except Exception as e:
            self.logger.error(f"Error querying accounts: {str(e)}")
            return {"error": str(e), "data": []}
    
    async def query_revenue_data(self, question: str, user_context: UserContext) -> Dict[str, Any]:
        """Query revenue and target data from both PostgreSQL and Fabric"""
        try:
            combined_data = {}
            
            # Get plan revenue targets from PostgreSQL
            async with self.postgres_pool.acquire() as conn:
                plan_targets_query = """
                SELECT 
                    AVG(CAST(revenue_growth_target AS FLOAT)) as avg_target,
                    COUNT(*) as plan_count,
                    MIN(CAST(revenue_growth_target AS FLOAT)) as min_target,
                    MAX(CAST(revenue_growth_target AS FLOAT)) as max_target
                FROM strategic_account_plans 
                WHERE tenant_id = $1 
                  AND revenue_growth_target IS NOT NULL 
                  AND revenue_growth_target ~ '^[0-9]+\\.?[0-9]*$'
                """
                
                plan_targets = await conn.fetchrow(plan_targets_query, user_context.tenant_id)
                combined_data['plan_targets'] = dict(plan_targets)
            
            # Get actual revenue from Fabric
            if self.fabric_service:
                revenue_query = """
                SELECT 
                    SUM(Amount) as total_opportunity_value,
                    COUNT(*) as opportunity_count,
                    AVG(Amount) as avg_opportunity_size
                FROM dbo.opportunities 
                WHERE Amount IS NOT NULL
                """
                
                revenue_results = await self.fabric_service.execute_query(revenue_query)
                if revenue_results:
                    combined_data['actual_revenue'] = revenue_results[0]
            
            if combined_data:
                avg_target = combined_data.get('plan_targets', {}).get('avg_target', 0) or 0
                total_opp_value = combined_data.get('actual_revenue', {}).get('total_opportunity_value', 0) or 0
                
                return {
                    "answer": f"Average revenue growth target: {avg_target:.1f}%. Total opportunity value: ${total_opp_value:,.0f}",
                    "data": [combined_data],
                    "query_type": "revenue_analysis",
                    "confidence": 0.9
                }
            
            return {"answer": "Could not find revenue data", "data": [], "confidence": 0.3}
            
        except Exception as e:
            self.logger.error(f"Error querying revenue data: {str(e)}")
            return {"error": str(e), "data": []}
    
    async def general_data_query(self, question: str, user_context: UserContext) -> Dict[str, Any]:
        """Handle general data queries that don't fit specific categories"""
        try:
            # This could be enhanced with NLP to better understand intent
            return {
                "answer": "I can help you with questions about strategic plans, opportunities, accounts, and revenue data. Could you be more specific?",
                "data": [],
                "query_type": "general",
                "confidence": 0.5,
                "suggestions": [
                    "How many strategic plans do I have?",
                    "Show me opportunities closing this quarter",
                    "What's my average revenue growth target?",
                    "List my recent strategic plans"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in general data query: {str(e)}")
            return {"error": str(e), "data": []}
    
    async def close(self):
        """Close database connections"""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            if self.fabric_service:
                await self.fabric_service.close()
            self.logger.info("✅ Database service connections closed")
        except Exception as e:
            self.logger.error(f"❌ Error closing database service: {str(e)}")
