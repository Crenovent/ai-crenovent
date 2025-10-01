#!/usr/bin/env python3
"""
Production CRM Data Sync for Knowledge Graph
============================================

This script shows how REAL production data will be synced from Salesforce/CRM
into the Knowledge Graph, replacing the sample data we populated for testing.

In production, this runs:
1. Every 15 minutes for real-time data sync
2. On-demand when users trigger automations
3. During nightly batch processes for historical data
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncpg
from config.database import DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class SalesforceAccount:
    """Real Salesforce Account data structure"""
    id: str
    name: str
    industry: str
    annual_revenue: float
    owner_id: str
    health_score: float
    region: str
    tier: str
    last_activity: datetime

@dataclass
class SalesforceOpportunity:
    """Real Salesforce Opportunity data structure"""
    id: str
    name: str
    account_id: str
    stage_name: str
    amount: float
    probability: float
    close_date: datetime
    owner_id: str
    forecast_category: str

class ProductionCRMSync:
    """
    Syncs real production data from CRM systems into Knowledge Graph
    """
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        
    async def sync_salesforce_accounts(self, tenant_id: str) -> int:
        """
        Sync real Salesforce accounts into Knowledge Graph
        In production, this connects to actual Salesforce API
        """
        logger.info(f"🔄 Syncing Salesforce accounts for tenant {tenant_id}")
        
        # In production, this would be:
        # salesforce_client = SalesforceClient(tenant_config)
        # accounts = await salesforce_client.query("SELECT Id, Name, Industry, AnnualRevenue, OwnerId FROM Account WHERE IsDeleted = false")
        
        # For demo, simulating real Salesforce data structure
        real_accounts = [
            SalesforceAccount(
                id="001XX000004C7g2YAC",
                name="Microsoft Corporation", 
                industry="Technology",
                annual_revenue=168088000000,  # Real Microsoft revenue
                owner_id="005XX000001SvKHYA0",
                health_score=0.92,
                region="Americas",
                tier="Strategic",
                last_activity=datetime.now() - timedelta(days=2)
            ),
            SalesforceAccount(
                id="001XX000004C7g3YAC",
                name="Salesforce Inc",
                industry="Software",
                annual_revenue=26492000000,  # Real Salesforce revenue
                owner_id="005XX000001SvKHYA0", 
                health_score=0.88,
                region="Americas",
                tier="Strategic",
                last_activity=datetime.now() - timedelta(days=1)
            ),
            # ... hundreds more in production
        ]
        
        synced_count = 0
        
        async with asyncpg.create_pool(self.db_config.postgres_url) as pool:
            async with pool.acquire() as conn:
                for account in real_accounts:
                    # Convert to Knowledge Graph entity
                    await conn.execute("""
                        INSERT INTO kg_entities (entity_type, entity_id, entity_name, metadata, tenant_id)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (entity_id, tenant_id) DO UPDATE SET
                            entity_name = EXCLUDED.entity_name,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    """, 
                    'Account',
                    f'sf_account_{account.id}',
                    account.name,
                    json.dumps({
                        'salesforce_id': account.id,
                        'industry': account.industry,
                        'annual_revenue': account.annual_revenue,
                        'owner_id': account.owner_id,
                        'health_score': account.health_score,
                        'region': account.region,
                        'tier': account.tier,
                        'last_activity': account.last_activity.isoformat(),
                        'data_source': 'salesforce',
                        'sync_timestamp': datetime.now().isoformat()
                    }),
                    tenant_id)
                    
                    synced_count += 1
        
        logger.info(f"✅ Synced {synced_count} Salesforce accounts")
        return synced_count
    
    async def sync_salesforce_opportunities(self, tenant_id: str) -> int:
        """
        Sync real Salesforce opportunities into Knowledge Graph
        """
        logger.info(f"🔄 Syncing Salesforce opportunities for tenant {tenant_id}")
        
        # In production: salesforce_client.query("SELECT Id, Name, AccountId, StageName, Amount, Probability, CloseDate FROM Opportunity WHERE IsWon = false AND IsClosed = false")
        
        real_opportunities = [
            SalesforceOpportunity(
                id="006XX000008WyUPYA0",
                name="Microsoft - RevAI Enterprise License",
                account_id="001XX000004C7g2YAC",
                stage_name="Negotiation/Review",
                amount=450000.00,
                probability=85.0,
                close_date=datetime(2024, 12, 31),
                owner_id="005XX000001SvKHYA0",
                forecast_category="Pipeline"
            ),
            SalesforceOpportunity(
                id="006XX000008WyUQYA0", 
                name="Salesforce - Annual Renewal",
                account_id="001XX000004C7g3YAC",
                stage_name="Proposal/Price Quote",
                amount=380000.00,
                probability=75.0,
                close_date=datetime(2024, 11, 30),
                owner_id="005XX000001SvKHYA0",
                forecast_category="Pipeline"
            ),
            # ... thousands more in production
        ]
        
        synced_count = 0
        
        async with asyncpg.create_pool(self.db_config.postgres_url) as pool:
            async with pool.acquire() as conn:
                for opp in real_opportunities:
                    await conn.execute("""
                        INSERT INTO kg_entities (entity_type, entity_id, entity_name, metadata, tenant_id)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (entity_id, tenant_id) DO UPDATE SET
                            entity_name = EXCLUDED.entity_name,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    """,
                    'Opportunity',
                    f'sf_opportunity_{opp.id}',
                    opp.name,
                    json.dumps({
                        'salesforce_id': opp.id,
                        'account_id': opp.account_id,
                        'stage_name': opp.stage_name,
                        'amount': opp.amount,
                        'probability': opp.probability / 100.0,
                        'close_date': opp.close_date.isoformat(),
                        'owner_id': opp.owner_id,
                        'forecast_category': opp.forecast_category,
                        'data_source': 'salesforce',
                        'sync_timestamp': datetime.now().isoformat()
                    }),
                    tenant_id)
                    
                    synced_count += 1
        
        logger.info(f"✅ Synced {synced_count} Salesforce opportunities")
        return synced_count
    
    async def sync_user_profiles(self, tenant_id: str) -> int:
        """
        Sync real user profiles from internal systems
        """
        logger.info(f"🔄 Syncing user profiles for tenant {tenant_id}")
        
        # In production, this queries the actual users table + Salesforce User object
        async with asyncpg.create_pool(self.db_config.postgres_url) as pool:
            async with pool.acquire() as conn:
                # Get real users from the users table
                users = await conn.fetch("""
                    SELECT u.user_id, u.email, u.profile, ur.role_name, ur.permissions,
                           ur.segment, ur.region, ur.area, ur.district, ur.territory
                    FROM users u
                    LEFT JOIN users_role ur ON u.user_id = ur.user_id
                    WHERE u.tenant_id = $1 AND u.is_active = true
                """, int(tenant_id) if tenant_id.isdigit() else tenant_id)
                
                synced_count = 0
                for user in users:
                    profile_data = json.loads(user['profile']) if user['profile'] else {}
                    
                    await conn.execute("""
                        INSERT INTO kg_entities (entity_type, entity_id, entity_name, metadata, tenant_id)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (entity_id, tenant_id) DO UPDATE SET
                            entity_name = EXCLUDED.entity_name,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    """,
                    'User',
                    f'user_{user["user_id"]}',
                    profile_data.get('name', user['email'].split('@')[0]),
                    json.dumps({
                        'user_id': user['user_id'],
                        'email': user['email'],
                        'role': user['role_name'],
                        'permissions': user['permissions'] or [],
                        'segment': user['segment'],
                        'region': user['region'], 
                        'area': user['area'],
                        'district': user['district'],
                        'territory': user['territory'],
                        'profile': profile_data,
                        'data_source': 'internal_users',
                        'sync_timestamp': datetime.now().isoformat()
                    }),
                    tenant_id)
                    
                    synced_count += 1
        
        logger.info(f"✅ Synced {synced_count} user profiles")
        return synced_count
    
    async def create_entity_relationships(self, tenant_id: str) -> int:
        """
        Create relationships between synced entities based on real data
        """
        logger.info(f"🔄 Creating entity relationships for tenant {tenant_id}")
        
        relationship_count = 0
        
        async with asyncpg.create_pool(self.db_config.postgres_url) as pool:
            async with pool.acquire() as conn:
                # Account -> Opportunity relationships (from Salesforce Account.Id -> Opportunity.AccountId)
                account_opp_relationships = await conn.fetch("""
                    SELECT 
                        a.id as account_kg_id,
                        o.id as opportunity_kg_id,
                        a.metadata->>'salesforce_id' as account_sf_id,
                        o.metadata->>'account_id' as opp_account_id,
                        o.metadata->>'amount' as amount,
                        o.metadata->>'probability' as probability
                    FROM kg_entities a
                    JOIN kg_entities o ON a.metadata->>'salesforce_id' = o.metadata->>'account_id'
                    WHERE a.entity_type = 'Account' 
                      AND o.entity_type = 'Opportunity'
                      AND a.tenant_id = $1
                """, tenant_id)
                
                for rel in account_opp_relationships:
                    await conn.execute("""
                        INSERT INTO kg_relationships (source_id, target_id, relationship_type, properties, tenant_id, confidence_score, evidence_pack_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT DO NOTHING
                    """,
                    rel['account_kg_id'],
                    rel['opportunity_kg_id'], 
                    'has_opportunity',
                    json.dumps({
                        'amount': float(rel['amount']) if rel['amount'] else 0,
                        'probability': float(rel['probability']) if rel['probability'] else 0,
                        'relationship_source': 'salesforce_sync',
                        'created_at': datetime.now().isoformat()
                    }),
                    tenant_id,
                    0.98,  # High confidence for direct Salesforce relationships
                    f"sf_sync_{datetime.now().strftime('%Y%m%d')}")
                    
                    relationship_count += 1
        
        logger.info(f"✅ Created {relationship_count} entity relationships")
        return relationship_count
    
    async def log_execution_trace(self, workflow_id: str, tenant_id: str, trace_data: Dict[str, Any]) -> str:
        """
        Log execution trace - this is called after EVERY automation execution in production
        """
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{workflow_id}"
        
        async with asyncpg.create_pool(self.db_config.postgres_url) as pool:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO kg_execution_traces (tenant_id, execution_id, workflow_type, trace_data, entities_involved, outcome_type, impact_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                tenant_id,
                execution_id,
                trace_data.get('workflow_type', 'RBA'),
                json.dumps(trace_data),
                trace_data.get('entities_involved', []),
                trace_data.get('outcome_type', 'success'),
                trace_data.get('impact_score', 0.5))
        
        logger.info(f"✅ Logged execution trace: {execution_id}")
        return execution_id

class ProductionDataManager:
    """
    Main manager for production data synchronization
    """
    
    def __init__(self):
        self.crm_sync = ProductionCRMSync()
    
    async def full_tenant_sync(self, tenant_id: str) -> Dict[str, int]:
        """
        Perform full data sync for a tenant (run this on tenant onboarding)
        """
        logger.info(f"🚀 Starting full sync for tenant {tenant_id}")
        
        results = {}
        
        # Sync core entities
        results['accounts'] = await self.crm_sync.sync_salesforce_accounts(tenant_id)
        results['opportunities'] = await self.crm_sync.sync_salesforce_opportunities(tenant_id)
        results['users'] = await self.crm_sync.sync_user_profiles(tenant_id)
        
        # Create relationships
        results['relationships'] = await self.crm_sync.create_entity_relationships(tenant_id)
        
        logger.info(f"🎉 Full sync completed for tenant {tenant_id}: {results}")
        return results
    
    async def incremental_sync(self, tenant_id: str, since: datetime) -> Dict[str, int]:
        """
        Perform incremental sync (run this every 15 minutes in production)
        """
        logger.info(f"🔄 Starting incremental sync for tenant {tenant_id} since {since}")
        
        # In production, this would query Salesforce for records modified since 'since' timestamp
        # For now, just run full sync
        return await self.full_tenant_sync(tenant_id)

# =============================================================================
# PRODUCTION USAGE EXAMPLE
# =============================================================================

async def production_example():
    """
    Example of how this works in production
    """
    data_manager = ProductionDataManager()
    
    print("🏭 PRODUCTION DATA SYNC EXAMPLE")
    print("=" * 50)
    
    # Tenant onboarding: Full sync
    tenant_id = 1300
    
    print(f"📊 Syncing real data for tenant {tenant_id}")
    results = await data_manager.full_tenant_sync(tenant_id)
    
    print(f"✅ Synced: {results['accounts']} accounts, {results['opportunities']} opportunities, {results['users']} users, {results['relationships']} relationships")
    
    # Simulate automation execution trace logging
    trace_data = {
        'workflow_id': 'FORECAST_ANALYSIS_PRODUCTION',
        'workflow_type': 'RBA',
        'input': {'region': 'Americas', 'period': 'Q4 2024'},
        'output': {'forecast_amount': 2800000, 'confidence': 0.82},
        'entities_involved': ['sf_account_001XX000004C7g2YAC', 'sf_opportunity_006XX000008WyUPYA0'],
        'governance': {'policy_pack': 'SOX_Compliance', 'compliance_score': 1.0},
        'outcome_type': 'success',
        'impact_score': 0.85
    }
    
    execution_id = await data_manager.crm_sync.log_execution_trace('FORECAST_ANALYSIS', tenant_id, trace_data)
    print(f"📝 Logged execution trace: {execution_id}")
    
    print("\n🎯 KEY INSIGHT:")
    print("In production, this sync runs:")
    print("• Every 15 minutes for incremental updates")
    print("• On-demand when users trigger automations") 
    print("• During nightly batch for historical data")
    print("• Every automation execution logs traces for ML training")

if __name__ == "__main__":
    asyncio.run(production_example())


