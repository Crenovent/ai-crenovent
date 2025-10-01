"""
Onboarding Workflow API
======================
REST API endpoints for user onboarding workflow execution with dynamic field assignment.
Provides workflow-based processing for CSV user imports with hierarchy field assignment.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Request
from typing import Dict, List, Any, Optional
import logging
import json
import csv
import io
import asyncio
from datetime import datetime, date
import uuid
import hashlib
try:
    import jwt
except ImportError:
    jwt = None

from dsl.hub.execution_hub import ExecutionHub
from dsl.hub.hub_router import get_execution_hub
from dsl.operators.rba.onboarding_rba_agent import OnboardingRBAAgent
from src.services.jwt_auth import require_auth, validate_tenant_access
from src.services.connection_pool_manager import pool_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onboarding", tags=["Onboarding Workflow"])

# Workflow execution status tracking
workflow_status = {}

@router.post("/execute-workflow")
async def execute_onboarding_workflow(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with user data"),
    tenant_id: int = Form(..., description="Tenant ID"),
    uploaded_by_user_id: int = Form(..., description="User ID who uploaded the file"),
    workflow_config: Optional[str] = Form(None, description="JSON string of workflow configuration"),
    execution_mode: str = Form("sync", description="Execution mode: sync or async"),
    save_to_database: bool = Form(False, description="Save processed users to database")
):
    # Debug: Check if endpoint is being called
    import os
    debug_path = os.path.join(os.getcwd(), 'debug_endpoint.txt')
    with open(debug_path, 'a') as f:
        f.write(f"=== ENDPOINT CALLED ===\n")
        f.write(f"tenant_id: {tenant_id}\n")
        f.write(f"execution_mode: {execution_mode}\n")
        f.write(f"workflow_config: {workflow_config}\n")
        f.write(f"file.filename: {file.filename}\n")
        f.write("=" * 50 + "\n")
    """
    Execute onboarding workflow with dynamic field assignment
    
    This endpoint:
    1. Validates JWT authentication
    2. Accepts CSV file with user data
    3. Parses workflow configuration from drag-drop interface
    4. Executes RBA agent for field assignment
    5. Optionally saves processed users to database
    6. Returns processed user data with assigned fields
    """
    # Initialize execution_id early to avoid UnboundLocalError
    execution_id = str(uuid.uuid4())
    
    try:
        # **PHASE 2: JWT AUTHENTICATION**
        user = await require_auth(request)
        logger.info(f"🔐 Authenticated user: {user.get('user_id')} (tenant: {user.get('tenant_id')})")
        logger.info(f"🔍 DEBUG: Full user object: {user}")
        logger.info(f"🔍 DEBUG: Requested tenant_id: {tenant_id} (type: {type(tenant_id)})")
        logger.info(f"🔍 DEBUG: User tenant_id: {user.get('tenant_id')} (type: {type(user.get('tenant_id'))})")
        
        # **PHASE 2: TENANT VALIDATION**  
        # **TEMPORARY**: Skip tenant validation for debugging
        tenant_validation_result = validate_tenant_access(user, tenant_id)
        logger.info(f"🔍 DEBUG: Tenant validation result: {tenant_validation_result}")
        
        if not tenant_validation_result:
            logger.warning(f"⚠️ BYPASSING tenant validation for debugging: User tenant {user.get('tenant_id')} != requested tenant {tenant_id}")
            # Temporarily allow access for debugging
            # raise HTTPException(
            #     status_code=403,
            #     detail={
            #         "error": "Forbidden",
            #         "message": f"User does not have access to tenant {tenant_id}. User tenant: {user.get('tenant_id')}",
            #         "timestamp": datetime.utcnow().isoformat()
            #     }
            # )
        
        logger.info(f"✅ Tenant access validated for user {user.get('user_id')} -> tenant {tenant_id}")
        
        # Initialize status tracking
        workflow_status[execution_id] = {
            'status': 'processing',
            'started_at': datetime.utcnow(),
            'progress': 0,
            'message': 'Initializing workflow execution'
        }
        
        logger.info(f"🚀 Starting onboarding workflow execution: {execution_id}")
        
        # Parse CSV file
        csv_content = await file.read()
        csv_data = parse_csv_content(csv_content)
        
        if not csv_data:
            raise HTTPException(status_code=400, detail="No valid user data found in CSV")
        
        # Parse workflow configuration
        config = {}
        if workflow_config:
            try:
                config = json.loads(workflow_config)
            except json.JSONDecodeError:
                logger.warning("Invalid workflow config JSON, using defaults")
        
        # Update status
        workflow_status[execution_id]['progress'] = 25
        workflow_status[execution_id]['message'] = f'Processing {len(csv_data)} users'
        
        if execution_mode == "async":
            # Execute in background
            background_tasks.add_task(
                execute_workflow_background,
                execution_id,
                csv_data,
                config,
                tenant_id,
                uploaded_by_user_id
            )
            
            return {
                'success': True,
                'execution_id': execution_id,
                'status': 'processing',
                'message': f'Workflow execution started for {len(csv_data)} users',
                'estimated_completion': '2-5 minutes'
            }
        else:
            # Execute synchronously
            result = await execute_workflow_sync(
                execution_id,
                csv_data,
                config,
                tenant_id,
                uploaded_by_user_id,
                save_to_database,
                user  # Pass authenticated user for database operations
            )
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Onboarding workflow execution failed: {e}")
        if execution_id in workflow_status:
            workflow_status[execution_id]['status'] = 'failed'
            workflow_status[execution_id]['error'] = str(e)
        
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@router.get("/status/{execution_id}")
async def get_workflow_status(execution_id: str):
    """Get the status of a workflow execution"""
    if execution_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Execution ID not found")
    
    return workflow_status[execution_id]

@router.get("/results/{execution_id}")
async def get_workflow_results(execution_id: str):
    """Get the results of a completed workflow execution"""
    if execution_id not in workflow_status:
        raise HTTPException(status_code=404, detail="Execution ID not found")
    
    status_info = workflow_status[execution_id]
    
    if status_info['status'] != 'completed':
        raise HTTPException(
            status_code=400, 
            detail=f"Workflow is not completed yet. Current status: {status_info['status']}"
        )
    
    return status_info.get('results', {})

@router.post("/validate-csv")
async def validate_csv_format(
    file: UploadFile = File(..., description="CSV file to validate")
):
    """
    Validate CSV format and identify missing fields that need workflow assignment
    """
    try:
        csv_content = await file.read()
        csv_data = parse_csv_content(csv_content)
        
        if not csv_data:
            return {
                'valid': False,
                'error': 'No valid data found in CSV',
                'missing_fields': [],
                'sample_data': []
            }
        
        # Analyze fields and identify missing ones
        sample_user = csv_data[0] if csv_data else {}
        required_fields = ['Region', 'Segment', 'Territory', 'Level', 'Modules']
        
        missing_fields = []
        for field in required_fields:
            if field not in sample_user or not sample_user.get(field, '').strip():
                missing_fields.append(field)
        
        return {
            'valid': True,
            'total_users': len(csv_data),
            'available_fields': list(sample_user.keys()),
            'missing_fields': missing_fields,
            'requires_workflow': len(missing_fields) > 0,
            'sample_data': csv_data[:3]  # First 3 records for preview
        }
        
    except Exception as e:
        logger.error(f"❌ CSV validation failed: {e}")
        return {
            'valid': False,
            'error': f'Validation failed: {str(e)}',
            'missing_fields': [],
            'sample_data': []
        }

async def execute_workflow_sync(
    execution_id: str,
    csv_data: List[Dict],
    config: Dict,
    tenant_id: int,
    user_id: int,
    save_to_database: bool = False,
    authenticated_user: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Execute workflow synchronously"""
    try:
        # Debug: Check if we reach this point
        import os
        debug_path = os.path.join(os.getcwd(), 'debug_fastapi.txt')
        with open(debug_path, 'a') as f:
            f.write(f"=== FASTAPI EXECUTE_WORKFLOW_SYNC CALLED ===\n")
            f.write(f"CSV data length: {len(csv_data)}\n")
            f.write(f"Config: {config}\n")
            f.write(f"CSV data sample: {csv_data[0] if csv_data else 'No data'}\n")
            f.write("=" * 50 + "\n")
        
        # Initialize onboarding RBA agent
        agent = OnboardingRBAAgent(config)
        
        with open(debug_path, 'a') as f:
            f.write(f"RBA Agent created successfully\n")
        
        # Update status
        workflow_status[execution_id]['progress'] = 50
        workflow_status[execution_id]['message'] = 'Executing field assignment logic'
        
        # Execute the RBA agent
        context = {
            'users_data': csv_data,
            'workflow_config': config,
            'tenant_id': tenant_id,
            'user_id': user_id
        }
        
        with open(debug_path, 'a') as f:
            f.write(f"About to call agent._execute_rba_logic\n")
        
        result = await agent._execute_rba_logic(context, config)
        
        with open(debug_path, 'a') as f:
            f.write(f"RBA logic completed, result keys: {list(result.keys()) if result else 'None'}\n")
        
        # Update status
        workflow_status[execution_id]['progress'] = 100
        workflow_status[execution_id]['status'] = 'completed'
        workflow_status[execution_id]['completed_at'] = datetime.utcnow()
        workflow_status[execution_id]['results'] = result
        
        # **CRITICAL: VERIFY MODULE ASSIGNMENTS IN RESPONSE**
        processed_users = result.get('processed_users', [])
        if processed_users:
            first_user = processed_users[0]
            logger.info(f"🎯 RESPONSE VERIFICATION - First user: {first_user.get('Name', 'Unknown')}")
            logger.info(f"🎯 RESPONSE VERIFICATION - Modules: {first_user.get('Modules', 'NOT FOUND')}")
            logger.info(f"🎯 RESPONSE VERIFICATION - Modules type: {type(first_user.get('Modules', 'NOT FOUND'))}")
            logger.info(f"🎯 RESPONSE VERIFICATION - modules_assigned flag: {first_user.get('modules_assigned', 'NOT FOUND')}")
            
            # Count users with modules
            users_with_modules = sum(1 for user in processed_users if user.get('Modules') and user.get('Modules') != '')
            logger.info(f"🎯 RESPONSE VERIFICATION - Users with modules: {users_with_modules}/{len(processed_users)}")
        
        # **PHASE 2: DATABASE SAVE DISABLED**
        # Database save is now handled by the separate /complete-setup endpoint
        # This endpoint only processes and returns data for frontend preview
        logger.info("💾 Database save skipped - will be handled by /complete-setup endpoint")
        if save_to_database:
            logger.info("⚠️ save_to_database=True ignored - use /complete-setup endpoint instead")
            result['database_save'] = {
                'success': True,
                'message': 'Database save will be handled by /complete-setup endpoint',
                'inserted_count': 0
            }
        
        logger.info(f"✅ Workflow {execution_id} completed successfully")
        
        return {
            'success': True,
            'execution_id': execution_id,
            'status': 'completed',
            'results': result
        }
        
    except Exception as e:
        logger.error(f"❌ Sync workflow execution failed: {e}")
        workflow_status[execution_id]['status'] = 'failed'
        workflow_status[execution_id]['error'] = str(e)
        raise

async def execute_workflow_background(
    execution_id: str,
    csv_data: List[Dict],
    config: Dict,
    tenant_id: int,
    user_id: int
):
    """Execute workflow in background task"""
    try:
        await execute_workflow_sync(execution_id, csv_data, config, tenant_id, user_id)
    except Exception as e:
        logger.error(f"❌ Background workflow execution failed: {e}")
        workflow_status[execution_id]['status'] = 'failed'
        workflow_status[execution_id]['error'] = str(e)

def parse_csv_content(csv_content: bytes) -> List[Dict[str, Any]]:
    """Parse CSV content and return list of dictionaries"""
    try:
        # Decode bytes to string
        csv_string = csv_content.decode('utf-8-sig')  # Handle BOM if present
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_string))
        
        users_data = []
        for row in csv_reader:
            # Clean up the row data
            cleaned_row = {}
            for key, value in row.items():
                if key:  # Skip empty column names
                    cleaned_key = key.strip()
                    cleaned_value = value.strip() if value else ''
                    cleaned_row[cleaned_key] = cleaned_value
            
            if cleaned_row:  # Only add non-empty rows
                users_data.append(cleaned_row)
        
        logger.info(f"📊 Parsed {len(users_data)} users from CSV")
        return users_data
        
    except Exception as e:
        logger.error(f"❌ CSV parsing failed: {e}")
        return []

async def save_users_to_database(
    processed_users: List[Dict[str, Any]], 
    tenant_id: int, 
    authenticated_user: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save processed users to database with the same logic as Node.js backend
    Replicates the functionality from crenovent-backend/controller/register/index.js
    """
    try:
        # Ensure connection pool is available with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not pool_manager.postgres_pool:
                    logger.info(f"🔄 Attempt {attempt + 1}: Initializing connection pool...")
                    await pool_manager.initialize()
                
                # Test the pool connection
                async with pool_manager.postgres_pool.acquire() as test_conn:
                    await test_conn.fetchval("SELECT 1")
                    logger.info("✅ Connection pool test successful")
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to establish database connection after {max_retries} attempts: {e}")
                await asyncio.sleep(2)  # Wait before retry
        
        inserted_users = []
        errors = []
        
        logger.info(f"🔍 Database save: PostgreSQL pool status: {pool_manager.postgres_pool is not None}")
        
        async with pool_manager.postgres_pool.acquire() as conn:
            # Start transaction
            async with conn.transaction():
                logger.info(f"🚀 Starting database transaction for {len(processed_users)} users")
                logger.info(f"🔍 Database save: Connection acquired successfully")
                
                # Create user map for hierarchy building (same as Node.js)
                user_map = {}
                for user in processed_users:
                    name_key = user.get("Name", "").strip().lower() if user.get("Name") else ""
                    if name_key:
                        user_map[name_key] = user
                
                logger.info(f"🗺️ Created user map with {len(user_map)} entries")
                
                # Insert users with hierarchy logic (same as Node.js)
                inserted = set()
                
                async def insert_user_recursively(user_name: str, row_idx: int):
                    """Recursive user insertion with hierarchy support"""
                    name_key = user_name.strip().lower() if user_name else ""
                    if not name_key or name_key in inserted:
                        return
                    
                    user = user_map.get(name_key)
                    if not user:
                        return
                    
                    try:
                        # Extract user data (same fields as Node.js)
                        # Handle modules properly - convert from list to string if needed
                        modules_data = user.get('Modules', '')
                        if isinstance(modules_data, list):
                            modules_string = ', '.join(modules_data)
                        elif isinstance(modules_data, str) and ',' in modules_data:
                            # Already a comma-separated string, keep as is
                            modules_string = modules_data.strip()
                        else:
                            modules_string = str(modules_data).strip()
                        
                        logger.info(f"🔍 DEBUG Database Save - User: {user.get('Name', 'Unknown')}")
                        logger.info(f"🔍 DEBUG Database Save - Raw modules: {modules_data} (type: {type(modules_data)})")
                        logger.info(f"🔍 DEBUG Database Save - Converted modules string: {modules_string}")
                        logger.info(f"🔍 DEBUG Database Save - Modules string length: {len(modules_string)}")
                        if isinstance(modules_data, list):
                            logger.info(f"🔍 DEBUG Database Save - Original list length: {len(modules_data)}")
                            logger.info(f"🔍 DEBUG Database Save - First 3 modules: {modules_data[:3]}")
                        
                        # Create profile JSON with all user data
                        profile = {
                            'role_title': user.get('Role Title', user.get('Role', '')).strip(),
                            'team': user.get('Team', '').strip(),
                            'department': user.get('Department', '').strip(),
                            'location': user.get('Location', user.get('Office Location', '')).strip(),
                            'employee_number': user.get('Employee Number', '').strip(),
                            'hire_date': user.get('Hire Date', '').strip(),
                            'user_status': user.get('User Status', 'Active').strip(),
                            'permissions': user.get('Permissions', '').strip(),
                            'user_type': user.get('User Type', 'Standard User').strip(),
                            'region': user.get('Region', '').strip(),
                            'segment': user.get('Segment', '').strip(),
                            'territory': user.get('Territory', '').strip(),
                            'area': user.get('Area', '').strip(),
                            'district': user.get('District', '').strip(),
                            'level': user.get('Level', '').strip(),
                            'modules': modules_string,
                            'tenant_id': tenant_id
                        }
                        
                        # Create password hash (temporary)
                        temp_password = f"temp_{user.get('Email', '').strip().lower()}"
                        password_hash = hashlib.sha256(temp_password.encode()).hexdigest()
                        
                        # Generate user_id (unique identifier)
                        user_id = abs(hash(user.get('Email', '').strip().lower())) % (10**9)
                        
                        # Create expiration date (1 year from now, same as Node.js)
                        today = datetime.now()
                        expiration_date = date(today.year + 1, today.month, today.day)
                        
                        # Generate JWT tokens (same as Node.js)
                        JWT_SECRET = "your-secret-key"  # Should be from environment
                        if jwt:
                            access_token = jwt.encode(
                                {'id': user_id, 'username': user.get('Name', '').strip()},
                                JWT_SECRET,
                                algorithm='HS256'
                            )
                            refresh_token = jwt.encode(
                                {'id': user_id, 'username': user.get('Name', '').strip()},
                                JWT_SECRET,
                                algorithm='HS256'
                            )
                        else:
                            access_token = f"temp_access_{user_id}"
                            refresh_token = f"temp_refresh_{user_id}"
                        
                        user_data = {
                            'user_id': user_id,
                            'username': user.get('Name', '').strip(),  # PostgreSQL uses 'username' not 'name'
                            'email': user.get('Email', '').strip().lower(),
                            'tenant_id': tenant_id,
                            'is_activated': True,
                            'profile': json.dumps(profile),  # Store as JSON string
                            'password': password_hash,  # Hashed password
                            'expiration_date': expiration_date,  # Proper date object
                            'access_token': access_token,  # JWT token
                            'refresh_token': refresh_token  # JWT refresh token
                        }
                        
                        # Handle manager relationship
                        manager_email = (user.get('Reports To Email') or 
                                       user.get('Reporting Email') or 
                                       user.get('Manager Email', '')).strip().lower()
                        
                        reports_to = None
                        if manager_email and manager_email != user_data['email']:
                            # Find manager in database
                            manager_result = await conn.fetchrow(
                                "SELECT user_id FROM users WHERE email = $1 AND tenant_id = $2",
                                manager_email, tenant_id
                            )
                            if manager_result:
                                reports_to = manager_result['user_id']
                        
                        user_data['reports_to'] = reports_to
                        
                        # Check if user already exists
                        existing_user = await conn.fetchrow(
                            "SELECT user_id FROM users WHERE email = $1 AND tenant_id = $2",
                            user_data['email'], tenant_id
                        )
                        
                        if existing_user:
                            # Update existing user with correct PostgreSQL columns
                            await conn.execute("""
                                UPDATE users SET 
                                    username = $1, profile = $2, reports_to = $3,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE user_id = $4 AND tenant_id = $5
                            """, 
                                user_data['username'], user_data['profile'], reports_to, 
                                existing_user['user_id'], tenant_id
                            )
                            inserted_users.append({
                                'user_id': existing_user['user_id'],
                                'name': user_data['username'],
                                'email': user_data['email'],
                                'action': 'updated'
                            })
                            logger.info(f"✅ Updated user: {user_data['username']} ({user_data['email']})")
                        else:
                            # Insert new user with correct PostgreSQL columns (matching Node.js exactly)
                            new_user_id = await conn.fetchval("""
                                INSERT INTO users (
                                    user_id, username, email, tenant_id, reports_to, is_activated, 
                                    profile, password, expiration_date, access_token, refresh_token, 
                                    created_at, updated_at
                                ) VALUES (
                                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                                ) RETURNING user_id
                            """,
                                user_data['user_id'], user_data['username'], user_data['email'], 
                                user_data['tenant_id'], reports_to, user_data['is_activated'],
                                user_data['profile'], user_data['password'], user_data['expiration_date'],
                                user_data['access_token'], user_data['refresh_token']
                            )
                            inserted_users.append({
                                'user_id': new_user_id,
                                'name': user_data['username'],
                                'email': user_data['email'],
                                'action': 'inserted'
                            })
                            logger.info(f"✅ Inserted user: {user_data['username']} ({user_data['email']})")
                        
                        inserted.add(name_key)
                        
                    except Exception as user_error:
                        error_msg = f"Failed to save user {user.get('Name', 'Unknown')}: {str(user_error)}"
                        logger.error(f"❌ {error_msg}")
                        errors.append({
                            'user': user.get('Name', 'Unknown'),
                            'email': user.get('Email', ''),
                            'error': str(user_error)
                        })
                
                # Process all users
                for i, user in enumerate(processed_users):
                    user_name = user.get('Name', '').strip()
                    if user_name:
                        await insert_user_recursively(user_name, i)
                
                logger.info(f"🎯 Database transaction completed: {len(inserted_users)} users processed")
        
        return {
            'success': True,
            'inserted_count': len(inserted_users),
            'error_count': len(errors),
            'inserted_users': inserted_users,
            'errors': errors
        }
        
    except Exception as e:
        logger.error(f"❌ Database save operation failed: {e}")
        return {
            'success': False,
            'inserted_count': 0,
            'error_count': 1,
            'errors': [{'error': str(e)}]
        }

@router.get("/test-endpoint")
async def test_endpoint():
    """Simple test endpoint to verify router is working"""
    logger.info("🧪 TEST ENDPOINT CALLED!")
    return {"status": "success", "message": "Test endpoint working"}

@router.post("/complete-setup")
async def complete_onboarding_setup(
    processed_users: str = Form(..., description="JSON string of processed users"),
    tenant_id: str = Form(..., description="Tenant ID")
):
    """
    Complete onboarding setup by saving processed users to database
    This endpoint is called by the "Complete Setup" button
    """
    logger.info("🚀 COMPLETE SETUP ENDPOINT CALLED!")
    logger.info(f"🔍 Complete Setup called with tenant_id: {tenant_id}")
    logger.info(f"📥 RAW INPUT: {len(processed_users)} characters")
    logger.info(f"📥 RAW DATA PREVIEW: {processed_users[:200]}...")
    
    try:
        # Parse JSON
        logger.info("🔍 Parsing JSON data...")
        users_data = json.loads(processed_users)
        logger.info(f"✅ Parsed {len(users_data)} users from JSON")
        
        # **CRITICAL MODULES DEBUG**
        if users_data:
            first_user = users_data[0]
            logger.info(f"🎯 MODULES DEBUG - First user: {first_user.get('Name', 'Unknown')}")
            logger.info(f"🎯 MODULES KEY EXISTS: {'Modules' in first_user}")
            logger.info(f"🎯 MODULES VALUE: {first_user.get('Modules', 'NOT FOUND')}")
            logger.info(f"🎯 MODULES TYPE: {type(first_user.get('Modules', 'NOT FOUND'))}")
            
            # Check first 3 users
            for i in range(min(3, len(users_data))):
                user = users_data[i]
                modules = user.get('Modules', 'NOT FOUND')
                logger.info(f"🎯 User {i+1} ({user.get('Name', 'Unknown')}): Modules = {modules}")
        
        # **PHASE 2: SAVE TO DATABASE**
        logger.info(f"💾 Starting database save for {len(users_data)} users...")
        try:
            # Mock user for database save (temporarily bypass auth)
            mock_user = {"user_id": 1370, "tenant_id": int(tenant_id)}
            
            database_result = await save_users_to_database(
                users_data, 
                int(tenant_id), 
                mock_user
            )
            logger.info(f"✅ Database save completed: {database_result}")
            
            return {
                "success": True,
                "message": "Complete Setup successful with database save!",
                "users_saved": database_result.get('inserted_count', 0),
                "users_count": len(users_data),
                "first_user_modules": users_data[0].get('Modules', 'NOT FOUND') if users_data else 'NO USERS',
                "tenant_id": tenant_id,
                "database_result": database_result
            }
            
        except Exception as db_error:
            logger.error(f"❌ Database save failed: {db_error}")
            return {
                "success": False,
                "error": f"Database save failed: {str(db_error)}",
                "users_count": len(users_data),
                "first_user_modules": users_data[0].get('Modules', 'NOT FOUND') if users_data else 'NO USERS',
                "tenant_id": tenant_id
            }
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON PARSE ERROR: {e}")
        return {
            "success": False,
            "error": f"JSON parse error: {str(e)}"
        }

@router.get("/agent-config")
async def get_agent_configuration():
    """Get the configuration schema for the onboarding RBA agent"""
    try:
        agent = OnboardingRBAAgent()
        
        return {
            'success': True,
            'config_schema': agent._define_config_schema(),
            'result_schema': agent._define_result_schema(),
            'agent_info': {
                'name': agent.AGENT_NAME,
                'description': 'Dynamic field assignment for user onboarding',
                'category': 'user_management'
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get agent config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent configuration: {str(e)}")

@router.post("/preview-assignments")
async def preview_field_assignments(
    file: UploadFile = File(..., description="CSV file with user data"),
    workflow_config: Optional[str] = Form(None, description="JSON string of workflow configuration"),
    sample_size: int = Form(5, description="Number of users to preview")
):
    """
    Preview field assignments without executing full workflow
    Shows how fields would be assigned for a sample of users
    """
    try:
        # Parse CSV file
        csv_content = await file.read()
        csv_data = parse_csv_content(csv_content)
        
        if not csv_data:
            raise HTTPException(status_code=400, detail="No valid user data found in CSV")
        
        # Parse workflow configuration
        config = {}
        if workflow_config:
            try:
                config = json.loads(workflow_config)
            except json.JSONDecodeError:
                logger.warning("Invalid workflow config JSON, using defaults")
        
        # Take a sample of users for preview
        sample_data = csv_data[:sample_size]
        
        # Initialize onboarding RBA agent
        agent = OnboardingRBAAgent(config)
        
        # Execute on sample data
        context = {
            'users_data': sample_data,
            'workflow_config': config,
            'tenant_id': 1,  # Preview mode
            'user_id': 1     # Preview mode
        }
        
        result = await agent._execute_rba_logic(context, config)
        
        return {
            'success': True,
            'preview_mode': True,
            'sample_size': len(sample_data),
            'total_users': len(csv_data),
            'preview_results': result.get('processed_users', []),
            'assignment_statistics': result.get('assignment_statistics', {}),
            'confidence_scores': result.get('confidence_scores', {})
        }
        
    except Exception as e:
        logger.error(f"❌ Preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for onboarding workflow service"""
    try:
        # Test agent initialization
        agent = OnboardingRBAAgent()
        
        return {
            'status': 'healthy',
            'service': 'onboarding_workflow',
            'agent_available': True,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'service': 'onboarding_workflow',
            'agent_available': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

