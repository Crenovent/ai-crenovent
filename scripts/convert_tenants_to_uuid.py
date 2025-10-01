import asyncio
import uuid
from src.services.connection_pool_manager import ConnectionPoolManager
from dotenv import load_dotenv

load_dotenv()

async def convert_tenants_to_uuid():
    """Convert existing integer tenant IDs to UUIDs across the system"""
    pool = ConnectionPoolManager()
    await pool.initialize()
    
    print("🔄 Converting existing integer tenant IDs to UUIDs...")
    
    # Create mapping of integer tenant IDs to UUIDs
    tenant_uuid_mapping = {}
    
    async with pool.postgres_pool.acquire() as conn:
        # Step 1: Get all existing tenant IDs
        print("\n📊 Step 1: Discovering existing tenant IDs...")
        
        existing_tenants = await conn.fetch("""
            SELECT DISTINCT tenant_id 
            FROM users 
            WHERE tenant_id IS NOT NULL 
            ORDER BY tenant_id
        """)
        
        print(f"Found {len(existing_tenants)} unique tenant IDs:")
        for row in existing_tenants:
            tenant_id = row['tenant_id']
            # Generate deterministic UUID based on tenant ID for consistency
            tenant_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"tenant-{tenant_id}"))
            tenant_uuid_mapping[tenant_id] = tenant_uuid
            print(f"  {tenant_id} → {tenant_uuid}")
        
        # Step 2: Create tenant mapping table for reference
        print(f"\n🗃️  Step 2: Creating tenant ID mapping table...")
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tenant_id_mapping (
                old_tenant_id INTEGER PRIMARY KEY,
                new_tenant_uuid UUID NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Insert mappings
        for old_id, new_uuid in tenant_uuid_mapping.items():
            try:
                await conn.execute("""
                    INSERT INTO tenant_id_mapping (old_tenant_id, new_tenant_uuid)
                    VALUES ($1, $2::uuid)
                    ON CONFLICT (old_tenant_id) DO UPDATE SET new_tenant_uuid = EXCLUDED.new_tenant_uuid
                """, old_id, new_uuid)
            except Exception as e:
                print(f"  ⚠️  Warning: {e}")
        
        print(f"✅ Tenant mapping table created with {len(tenant_uuid_mapping)} mappings")
        
        # Step 3: Update existing tables that use tenant_id
        print(f"\n🔧 Step 3: Updating existing tables to use UUID tenant_id...")
        
        # Tables to update (checking if columns exist first)
        tables_to_update = [
            'users',
            'strategic_account_plans',
            'account_planning_templates',
            'users_role'
        ]
        
        for table_name in tables_to_update:
            try:
                print(f"  📋 Updating {table_name}...")
                
                # Check if table has tenant_id column
                column_check = await conn.fetchrow("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = $1 AND column_name = 'tenant_id'
                """, table_name)
                
                if not column_check:
                    print(f"    ⏭️  Skipping {table_name} (no tenant_id column)")
                    continue
                
                current_type = column_check['data_type']
                print(f"    Current type: {current_type}")
                
                if current_type == 'uuid':
                    print(f"    ✅ {table_name} already uses UUID tenant_id")
                    continue
                
                # Add new UUID column
                await conn.execute(f"""
                    ALTER TABLE {table_name} 
                    ADD COLUMN IF NOT EXISTS tenant_uuid UUID
                """)
                
                # Update with mapped UUIDs
                for old_id, new_uuid in tenant_uuid_mapping.items():
                    await conn.execute(f"""
                        UPDATE {table_name} 
                        SET tenant_uuid = $1::uuid 
                        WHERE tenant_id = $2
                    """, new_uuid, old_id)
                
                # Count updated rows
                count = await conn.fetchval(f"""
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE tenant_uuid IS NOT NULL
                """)
                
                print(f"    ✅ Updated {count} rows in {table_name}")
                
            except Exception as e:
                print(f"    ❌ Error updating {table_name}: {e}")
        
        # Step 4: Show the mapping for DSL setup
        print(f"\n🎯 Step 4: Tenant UUID mapping for DSL setup:")
        print(f"For tenant 1300 (your main tenant): {tenant_uuid_mapping.get(1300, 'NOT FOUND')}")
        
        # Store the main tenant UUID for easy access
        main_tenant_uuid = tenant_uuid_mapping.get(1300)
        if main_tenant_uuid:
            print(f"\n💾 Saving main tenant UUID to .env file...")
            with open('.env', 'a') as f:
                f.write(f"\n# Main tenant UUID for DSL\n")
                f.write(f"MAIN_TENANT_UUID={main_tenant_uuid}\n")
            print(f"✅ Added MAIN_TENANT_UUID={main_tenant_uuid} to .env")
    
    await pool.close()
    return tenant_uuid_mapping

if __name__ == "__main__":
    mapping = asyncio.run(convert_tenants_to_uuid())
    print(f"\n🎉 Conversion complete! Generated {len(mapping)} UUID mappings.")
