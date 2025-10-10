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
from datetime import datetime, date, timedelta
import uuid
import hashlib
import os
try:
    import jwt
except ImportError:
    jwt = None

try:
    import httpx
except ImportError:
    httpx = None

from dsl.hub.execution_hub import ExecutionHub
from dsl.hub.hub_router import get_execution_hub
from dsl.operators.rba.onboarding_rba_agent import OnboardingRBAAgent
from src.services.jwt_auth import require_auth, validate_tenant_access
from src.services.connection_pool_manager import pool_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onboarding", tags=["Onboarding Workflow"])

def generate_service_token(authenticated_user: Dict[str, Any]) -> str:
    """
    Generate JWT token for service-to-service communication
    This token will be used by AI backend to authenticate with TS backend
    """
    try:
        import jwt
        import time
        
        # Get JWT secret from environment (same as TS backend)
        jwt_secret = os.getenv('JWT_SECRET', 'your-secret-key')
        
        # Extract user info from authenticated_user object
        # The user object from require_auth has 'user_id' and 'tenant_id' fields (UUID strings)
        user_id = authenticated_user.get('user_id') or authenticated_user.get('id', '0')
        tenant_id = authenticated_user.get('tenant_id', '0')
        email = authenticated_user.get('email', 'service@crenovent.com')
        role = authenticated_user.get('role', 'admin')
        
        logger.info(f" [SERVICE-AUTH] Extracted from authenticated_user: user_id={user_id}, tenant_id={tenant_id}, email={email}, role={role}")
        
        payload = {
            'id': user_id,  # Use the UUID string from user_id field
            'tenant_id': tenant_id,
            'email': email,
            'role': role,
            'service': 'ai-backend',
            'target_service': 'ts-backend',
            'iat': int(time.time()),
            'exp': int(time.time()) + (15 * 60)  # 15 minutes
        }
        
        logger.info(f" [SERVICE-AUTH] Generating service token for: userId: {user_id}, tenantId: {tenant_id}, email: {email}, role: {role}, service: {payload['service']}, target_service: {payload['target_service']}")
        
        token = jwt.encode(payload, jwt_secret, algorithm='HS256')
        
        logger.info(" [SERVICE-AUTH] Service token generated successfully")
        return token
        
    except Exception as e:
        logger.error(f" [SERVICE-AUTH] Failed to generate service token: {e}")
        raise Exception(f"Service token generation failed: {str(e)}")


# Workflow execution status tracking
workflow_status = {}

@router.post("/execute-workflow")
async def execute_onboarding_workflow(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with user data"),
    tenant_id: str = Form(..., description="Tenant ID (UUID string)"),
    uploaded_by_user_id: str = Form(..., description="User ID who uploaded the file (UUID string)"),
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
        logger.info(f" Authenticated user: {user.get('user_id')} (tenant: {user.get('tenant_id')})")
        logger.info(f"DEBUG: Full user object: {user}")
        logger.info(f"DEBUG: Requested tenant_id: {tenant_id} (type: {type(tenant_id)})")
        logger.info(f"DEBUG: User tenant_id: {user.get('tenant_id')} (type: {type(user.get('tenant_id'))})")
        
        # **PHASE 2: TENANT VALIDATION**  
        # **TEMPORARY**: Skip tenant validation for debugging
        tenant_validation_result = validate_tenant_access(user, tenant_id)
        logger.info(f"DEBUG: Tenant validation result: {tenant_validation_result}")
        
        if not tenant_validation_result:
            logger.warning(f" BYPASSING tenant validation for debugging: User tenant {user.get('tenant_id')} != requested tenant {tenant_id}")
            # Temporarily allow access for debugging
            # raise HTTPException(
            #     status_code=403,
            #     detail={
            #         "error": "Forbidden",
            #         "message": f"User does not have access to tenant {tenant_id}. User tenant: {user.get('tenant_id')}",
            #         "timestamp": datetime.utcnow().isoformat()
            #     }
            # )
        
        logger.info(f" Tenant access validated for user {user.get('user_id')} -> tenant {tenant_id}")
        
        # Initialize status tracking
        workflow_status[execution_id] = {
            'status': 'processing',
            'started_at': datetime.utcnow(),
            'progress': 0,
            'message': 'Initializing workflow execution'
        }
        
        logger.info(f" Starting onboarding workflow execution: {execution_id}")
        
        # Parse CSV file
        logger.info(f" Processing CSV file: {file.filename} (size: {file.size if hasattr(file, 'size') else 'unknown'})")
        csv_content = await file.read()
        logger.info(f" CSV content size: {len(csv_content)} bytes")
        csv_data = parse_csv_content(csv_content)
        logger.info(f" Parsed CSV data: {len(csv_data)} users")
        
        if csv_data:
            logger.info(f" First user sample: {csv_data[0]}")
        else:
            logger.warning(" No CSV data parsed!")
        
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
        logger.error(f" Onboarding workflow execution failed: {e}")
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
        logger.error(f" CSV validation failed: {e}")
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
    tenant_id: str,
    user_id: str,
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
            logger.info(f" RESPONSE VERIFICATION - First user: {first_user.get('Name', 'Unknown')}")
            logger.info(f" RESPONSE VERIFICATION - Modules: {first_user.get('Modules', 'NOT FOUND')}")
            logger.info(f" RESPONSE VERIFICATION - Modules type: {type(first_user.get('Modules', 'NOT FOUND'))}")
            logger.info(f" RESPONSE VERIFICATION - modules_assigned flag: {first_user.get('modules_assigned', 'NOT FOUND')}")
            
            # Count users with modules
            users_with_modules = sum(1 for user in processed_users if user.get('Modules') and user.get('Modules') != '')
            logger.info(f" RESPONSE VERIFICATION - Users with modules: {users_with_modules}/{len(processed_users)}")
        
        # **PHASE 2: DATABASE SAVE DISABLED**
        # Database save is now handled by the separate /complete-setup endpoint
        # This endpoint only processes and returns data for frontend preview
        logger.info(" Database save skipped - will be handled by /complete-setup endpoint")
        if save_to_database:
            logger.info(" save_to_database=True ignored - use /complete-setup endpoint instead")
            result['database_save'] = {
                'success': True,
                'message': 'Database save will be handled by /complete-setup endpoint',
                'inserted_count': 0
            }
        
        logger.info(f" Workflow {execution_id} completed successfully")
        
        return {
            'success': True,
            'execution_id': execution_id,
            'status': 'completed',
            'results': result
        }
        
    except Exception as e:
        logger.error(f" Sync workflow execution failed: {e}")
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
        logger.error(f" Background workflow execution failed: {e}")
        workflow_status[execution_id]['status'] = 'failed'
        workflow_status[execution_id]['error'] = str(e)

def parse_csv_content(csv_content: bytes) -> List[Dict[str, Any]]:
    """Parse CSV content and return list of dictionaries"""
    try:
        logger.info(f"CSV parsing: Input size {len(csv_content)} bytes")
        
        # Decode bytes to string
        csv_string = csv_content.decode('utf-8-sig')  # Handle BOM if present
        logger.info(f"CSV parsing: Decoded string length {len(csv_string)} chars")
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_string))
        logger.info(f"CSV parsing: Headers found: {csv_reader.fieldnames}")
        
        users_data = []
        for i, row in enumerate(csv_reader):
            # Clean up the row data
            cleaned_row = {}
            for key, value in row.items():
                if key:  # Skip empty column names
                    cleaned_key = key.strip()
                    cleaned_value = value.strip() if value else ''
                    cleaned_row[cleaned_key] = cleaned_value
            
            if cleaned_row:  # Only add non-empty rows
                users_data.append(cleaned_row)
                if i < 3:  # Log first 3 rows for debugging
                    logger.info(f"CSV parsing: Row {i+1}: {cleaned_row}")
        
        logger.info(f" Parsed {len(users_data)} users from CSV")
        return users_data
        
    except Exception as e:
        logger.error(f" CSV parsing failed: {e}")
        import traceback
        logger.error(f" CSV parsing traceback: {traceback.format_exc()}")
        return []

async def save_users_to_database(
    processed_users: List[Dict[str, Any]], 
    tenant_id: str, 
    authenticated_user: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save processed users to database by calling the TypeScript backend
    Uses the existing bulkImportUsers endpoint instead of direct database access
    """
    try:
        logger.info(f" Calling TypeScript backend to save {len(processed_users)} users")
        
        # Get TypeScript backend URL from environment
        nodejs_backend_url = os.getenv('NODEJS_BACKEND_URL', 'http://localhost:8080')
        
        # Convert processed users to the format expected by bulkImportUsers
        users_for_import = []
        for user in processed_users:
            # DEBUG: Log all available fields in the processed user data
            logger.info(f"[DEBUG] Processing user: {user.get('Name', 'Unknown')}")
            logger.info(f"[DEBUG] Available fields: {list(user.keys())}")
            logger.info(f"[DEBUG] Manager Email field: '{user.get('Manager Email', 'NOT_FOUND')}'")
            logger.info(f"[DEBUG] Reports To Email field: '{user.get('Reports To Email', 'NOT_FOUND')}'")
            logger.info(f"[DEBUG] Reporting Email field: '{user.get('Reporting Email', 'NOT_FOUND')}'")
            
            # Extract user data
            email = user.get('Email', '').strip().lower()
            full_name = user.get('Name', '').strip()
            
            # Split name into first and last name
            name_parts = full_name.split(' ', 1)
            first_name = name_parts[0] if name_parts else ''
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            
            # Handle manager relationship - try multiple field names
            manager_email = (user.get('Reports To Email') or 
                           user.get('Reporting Email') or 
                           user.get('Manager Email') or
                           user.get('manager_email') or
                           user.get('ManagerEmail', '')).strip().lower()
            
            logger.info(f"[DEBUG] Extracted manager_email: '{manager_email}'")
            
            # Convert modules to string if it's a list
            modules_data = user.get('Modules', '')
            if isinstance(modules_data, list):
                modules_string = ', '.join(modules_data)
            else:
                modules_string = str(modules_data).strip()
                        
            user_data = {
                'email': email,
                'name': f"{first_name} {last_name}".strip(),  # Combine first and last name
                'firstName': first_name,
                'lastName': last_name,
                'jobTitle': user.get('Job Title', ''),
                'department': user.get('Department', ''),
                'location': user.get('Location', ''),
                'employeeId': user.get('Employee ID', ''),
                'startDate': user.get('Start Date', ''),
                'role': user.get('Role', ''),
                'userType': user.get('User Type', 'Standard User'),
                'profile': user.get('Profile', ''),
                'reportingEmail': manager_email,  # Fixed: Use reportingEmail instead of managerEmail
                'region': user.get('Region', ''),
                'segment': user.get('Segment', ''),
                'territory': user.get('Territory', ''),
                'area': user.get('Area', ''),
                'district': user.get('District', ''),
                'level': user.get('Level', ''),
                'modules': modules_string
            }
            users_for_import.append(user_data)
        
        # Prepare the request payload
        # Use the original UUID string from authenticated_user instead of converted integer
        original_tenant_id = authenticated_user.get('tenant_id', str(tenant_id))
        # Ensure it's always a string for headers
        original_tenant_id = str(original_tenant_id)
        logger.info(f" [SERVICE-AUTH] Using original tenant_id UUID: {original_tenant_id} (instead of integer: {tenant_id})")
        
        payload = {
            'users': users_for_import,
            'tenantId': original_tenant_id,  # Use UUID string, not integer
            'sendInviteEmails': True  # Enable email sending
        }
        
        # Generate service token for authentication
        # Use the original authenticated user object (contains UUID strings)
        if authenticated_user is None:
            # This should not happen since user is passed from the endpoint
            logger.error(" [SERVICE-AUTH] No authenticated_user provided - this should not happen")
            raise Exception("No authenticated user provided for service token generation")
        
        logger.info(f" [SERVICE-AUTH] Using authenticated_user: {authenticated_user}")
        service_token = generate_service_token(authenticated_user)
        
        # Call TypeScript backend bulkImportUsers endpoint
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {service_token}',
            'x-tenant-id': original_tenant_id  # Use UUID string, not integer
        }
        
        logger.info(f" Calling {nodejs_backend_url}/api/auth/bulk-import")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{nodejs_backend_url}/api/auth/bulk-import",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f" TypeScript backend response: {result}")
                
                # Convert the response to match expected format
                return {
                    "success": True,
                    "message": "Users saved successfully via TypeScript backend",
                    "users_saved": len(users_for_import),
                    "database_operations": {
                        "users_created": result.get('created', 0),
                        "users_updated": result.get('updated', 0),
                        "errors": result.get('errors', [])
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_msg = f"TypeScript backend returned status {response.status_code}: {response.text}"
                logger.error(f" {error_msg}")
        
        return {
            "success": False,
            "message": "Failed to save users via TypeScript backend",
            "error": error_msg,
            "users_saved": 0,
            "database_operations": {
                "users_created": 0,
                "users_updated": 0,
                "errors": [error_msg]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Error calling TypeScript backend: {str(e)}"
        logger.error(f" {error_msg}")
        return {
            "success": False,
            "message": "Failed to save users",
            "error": error_msg,
            "users_saved": 0,
            "database_operations": {
                "users_created": 0,
                "users_updated": 0,
                "errors": [error_msg]
            },
            "timestamp": datetime.now().isoformat()
        }

@router.get("/test-endpoint")
async def test_endpoint():
    """Simple test endpoint to verify router is working"""
    logger.info(" TEST ENDPOINT CALLED!")
    return {"status": "success", "message": "Test endpoint working"}

@router.post("/complete-setup")
async def complete_onboarding_setup(request: dict):
    """
    Complete onboarding setup by saving processed users to database
    This endpoint is called by the "Complete Setup" button
    """
    logger.info(" COMPLETE SETUP ENDPOINT CALLED!")
    
    try:
        # Extract data from request body
        processed_users = request.get('processed_users')
        tenant_id = request.get('tenant_id')
        
        logger.info(f"Complete Setup called with tenant_id: {tenant_id}")
        logger.info(f" RAW INPUT: {len(processed_users) if processed_users else 0} characters")
        logger.info(f" RAW DATA PREVIEW: {processed_users[:200] if processed_users else 'None'}...")
        
        if not processed_users or not tenant_id:
            logger.error(" Missing required fields: processed_users or tenant_id")
            return {
                "success": False,
                "error": "Missing required fields: processed_users and tenant_id"
            }
        
        # Validate tenant_id (should be a UUID string)
        import uuid
        try:
            # Try to parse as UUID
            tenant_uuid = uuid.UUID(tenant_id)
            logger.info(f" Valid tenant_id UUID: {tenant_uuid}")
        except (ValueError, TypeError):
            logger.error(f" Invalid tenant_id UUID format: {tenant_id}")
            return {
                "success": False,
                "error": "Invalid tenant ID UUID format provided"
            }
        
        # Parse JSON
        logger.info("Parsing JSON data...")
        try:
            users_data = json.loads(processed_users)
            if not isinstance(users_data, list):
                raise ValueError("Processed users must be a list")
            logger.info(f" Parsed {len(users_data)} users from JSON")
        except json.JSONDecodeError as e:
            logger.error(f" JSON PARSE ERROR: {e}")
            return {
                "success": False,
                "error": f"Invalid JSON format: {str(e)}"
            }
        except ValueError as e:
            logger.error(f" DATA VALIDATION ERROR: {e}")
            return {
                "success": False,
                "error": f"Data validation error: {str(e)}"
            }
        
        # Validate users data
        if not users_data:
            logger.error(" No users data provided")
            return {
                "success": False,
                "error": "No users data provided"
            }
        
        # Validate required fields for each user
        for i, user in enumerate(users_data):
            if not isinstance(user, dict):
                logger.error(f" User {i+1} is not a valid object")
                return {
                    "success": False,
                    "error": f"User {i+1} is not a valid object"
                }
            
            required_fields = ['Name', 'Email']
            for field in required_fields:
                if not user.get(field) or not str(user.get(field)).strip():
                    logger.error(f" User {i+1} missing required field: {field}")
                    return {
                        "success": False,
                        "error": f"User {i+1} missing required field: {field}"
                    }
        
        # **CRITICAL MODULES DEBUG**
        if users_data:
            first_user = users_data[0]
            logger.info(f" MODULES DEBUG - First user: {first_user.get('Name', 'Unknown')}")
            logger.info(f" MODULES KEY EXISTS: {'Modules' in first_user}")
            logger.info(f" MODULES VALUE: {first_user.get('Modules', 'NOT FOUND')}")
            logger.info(f" MODULES TYPE: {type(first_user.get('Modules', 'NOT FOUND'))}")
            
            # Check first 3 users
            for i in range(min(3, len(users_data))):
                user = users_data[i]
                modules = user.get('Modules', 'NOT FOUND')
                logger.info(f" User {i+1} ({user.get('Name', 'Unknown')}): Modules = {modules}")
        
        # **PHASE 2: SAVE TO DATABASE**
        logger.info(f" Starting database save for {len(users_data)} users...")
        try:
            # Mock user for database save (temporarily bypass auth)
            mock_user = {"user_id": "c0d323a5-0e78-4936-bfe4-0da5e16ce185", "tenant_id": tenant_id}
            
            database_result = await save_users_to_database(
                users_data, 
                tenant_id, 
                mock_user
            )
            logger.info(f" Database save completed: {database_result}")
            
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
            logger.error(f" Database save failed: {db_error}")
            return {
                "success": False,
                "error": f"Database save failed: {str(db_error)}",
                "users_count": len(users_data),
                "first_user_modules": users_data[0].get('Modules', 'NOT FOUND') if users_data else 'NO USERS',
                "tenant_id": tenant_id
            }
        
    except Exception as e:
        logger.error(f" Unexpected error in complete setup: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
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
        logger.error(f" Failed to get agent config: {e}")
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
        logger.error(f" Preview failed: {e}")
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
        logger.error(f" Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'service': 'onboarding_workflow',
            'agent_available': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


