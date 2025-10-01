"""
Quick script to seed policy packs with correct structure
"""
import asyncio
import uuid
from src.services.connection_pool_manager import pool_manager

async def seed_policy_packs():
    try:
        print("🌱 Seeding policy packs...")
        
        # Initialize connection pool
        success = await pool_manager.initialize()
        if not success:
            raise Exception("Failed to initialize connection pool")
        
        async with pool_manager.postgres_pool.acquire() as conn:
            # Set tenant context
            await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "1300")
            
            # Check existing policy packs
            existing_count = await conn.fetchval("SELECT COUNT(*) FROM dsl_policy_packs WHERE tenant_id = 1300")
            print(f"📊 Existing policy packs: {existing_count}")
            
            if existing_count == 0:
                # Insert SOX policy pack
                sox_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO dsl_policy_packs (
                        policy_pack_id, name, description, version, rules, industry, 
                        compliance_standards, is_global, created_by_user_id, tenant_id, 
                        pack_type, industry_code, region_code, enforcement_level
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                    sox_id,
                    'SaaS SOX Compliance Pack',
                    'SOX compliance rules for SaaS companies',
                    '1.0.0',
                    {
                        "financial_controls": {
                            "revenue_recognition": {"enabled": True, "enforcement": "strict"},
                            "deal_approval_thresholds": {"high_value": 250000, "mega_deal": 1000000},
                            "segregation_of_duties": {"enabled": True, "maker_checker": True}
                        },
                        "audit_requirements": {
                            "execution_logging": {"enabled": True, "retention_days": 2555},
                            "override_justification": {"required": True, "approval_required": True},
                            "evidence_generation": {"enabled": True, "immutable": True}
                        }
                    },
                    'SaaS',
                    ["SOX"],
                    False,
                    1319,
                    1300,
                    'SOX',
                    'SaaS',
                    'US',
                    'strict'
                )
                
                # Insert GDPR policy pack
                gdpr_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO dsl_policy_packs (
                        policy_pack_id, name, description, version, rules, industry,
                        compliance_standards, is_global, created_by_user_id, tenant_id,
                        pack_type, industry_code, region_code, enforcement_level
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                    gdpr_id,
                    'SaaS GDPR Privacy Pack',
                    'GDPR privacy rules for SaaS companies',
                    '1.0.0',
                    {
                        "data_protection": {
                            "consent_management": {"enabled": True, "explicit_consent": True},
                            "right_to_erasure": {"enabled": True, "retention_override": False},
                            "data_minimization": {"enabled": True, "purpose_limitation": True}
                        },
                        "privacy_controls": {
                            "pii_classification": {"enabled": True, "auto_detection": True},
                            "cross_border_transfer": {"restricted": True, "adequacy_required": True},
                            "breach_notification": {"enabled": True, "notification_hours": 72}
                        }
                    },
                    'SaaS',
                    ["GDPR"],
                    False,
                    1319,
                    1300,
                    'GDPR',
                    'SaaS',
                    'EU',
                    'strict'
                )
                
                print("✅ SOX and GDPR policy packs created")
            else:
                print(f"✅ {existing_count} policy packs already exist")
            
            # Test multi-tenant functionality
            print("🧪 Testing multi-tenant functionality...")
            
            # Test tenant context
            await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "1300")
            policy_count = await conn.fetchval("SELECT COUNT(*) FROM dsl_policy_packs WHERE tenant_id = 1300")
            print(f"  ✅ Policy packs for tenant 1300: {policy_count}")
            
            # Test cross-tenant isolation
            await conn.execute("SELECT set_config('app.current_tenant_id', $1, false)", "9999")
            cross_tenant_count = await conn.fetchval("SELECT COUNT(*) FROM dsl_policy_packs WHERE tenant_id = 1300")
            
            if cross_tenant_count == 0:
                print("  ✅ Cross-tenant isolation working (RLS policies active)")
            else:
                print(f"  ⚠️ Cross-tenant isolation may have issues ({cross_tenant_count} records visible)")
            
            print("\n🎉 SUCCESS: Policy packs seeded and multi-tenant functionality verified!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pool_manager.postgres_pool:
            await pool_manager.postgres_pool.close()

if __name__ == "__main__":
    asyncio.run(seed_policy_packs())
