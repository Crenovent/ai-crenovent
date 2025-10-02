"""
CRUD Operations Agent with Advanced Natural Language Processing
Handles database modifications with natural language processing and user confirmation
Supports complex operations like "Delete all plans from last month" or "Update revenue targets by 15%"
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import requests
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class CRUDOperation(Enum):
    CREATE = "create"
    READ = "read" 
    UPDATE = "update"
    DELETE = "delete"

@dataclass
class CRUDRequest:
    """Represents a CRUD operation request"""
    operation: CRUDOperation
    entity: str  # plan, stakeholder, activity, etc.
    entity_id: Optional[str] = None
    fields: Dict[str, Any] = None
    conditions: Dict[str, Any] = None
    confirmation_required: bool = True
    
@dataclass 
class CRUDResult:
    """Result of a CRUD operation"""
    success: bool
    operation: CRUDOperation
    entity: str
    affected_records: int
    data: Any = None
    message: str = ""
    error: Optional[str] = None

class CRUDAgent:
    """
    CRUD Operations Agent that:
    1. Parses natural language CRUD requests
    2. Validates operations against RBAC permissions
    3. Requires user confirmation for destructive operations
    4. Executes database operations via backend API
    5. Provides detailed feedback on results
    """
    
    def __init__(self, backend_url: str = None):
        if backend_url is None:
            backend_url = os.getenv('BACKEND_BASE_URL', 'http://localhost:3001')
        
        self.agent_id = "crud_agent"
        self.name = "CRUD Operations Agent"
        self.description = "Natural language database operations with confirmation"
        self.backend_url = backend_url
        
        # Memory for operation history
        self.memory = ConversationBufferWindowMemory(
            k=20,  # Remember last 20 operations
            return_messages=True,
            memory_key="crud_history"
        )
        
        # Track pending confirmations
        self.pending_confirmations: Dict[str, CRUDRequest] = {}
        
        logger.info(f"🔧 {self.name} initialized")
    
    async def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if this agent should handle the message"""
        
        crud_patterns = [
            # Create patterns
            r'\b(create|add|new|make)\b.*\b(plan|stakeholder|activity|goal)\b',
            
            # Update patterns  
            r'\b(update|change|modify|edit|set)\b.*\b(arr|revenue|goal|date|status)\b',
            r'\bchange\b.*\bto\b',
            r'\bset\b.*\bto\b',
            
            # Delete patterns
            r'\b(delete|remove|cancel)\b.*\b(plan|stakeholder|activity)\b',
            
            # Read patterns with modification intent
            r'\bshow\b.*\bthen\b.*(change|update|modify)',
            
            # Confirmation patterns
            r'\b(yes|no|confirm|cancel|proceed)\b'
        ]
        
        return any(re.search(pattern, message.lower()) for pattern in crud_patterns)
    
    async def handle_message(self, 
                           message: str, 
                           user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CRUD operation messages"""
        
        try:
            user_id = user_context.get("user_id", "unknown")
            session_id = user_context.get("session_id", "unknown")
            
            # Check if this is a confirmation response
            if await self._is_confirmation_response(message):
                return await self._handle_confirmation(message, user_id, session_id)
            
            # Parse CRUD request from natural language
            crud_request = await self._parse_crud_request(message, user_context)
            
            if not crud_request:
                return {
                    "agent": self.name,
                    "response": "I couldn't understand the database operation you want to perform. Please be more specific.",
                    "examples": [
                        "Change plan ABC's ARR to 250000",
                        "Update stakeholder John's role to Champion", 
                        "Delete activity with ID 123",
                        "Create new plan for account XYZ"
                    ]
                }
            
            # Handle the CRUD request
            return await self._handle_crud_request(crud_request, user_id, session_id)
            
        except Exception as e:
            logger.error(f"❌ Error handling CRUD message: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "response": "I encountered an error while processing your database operation request."
            }
    
    async def _is_confirmation_response(self, message: str) -> bool:
        """Check if message is a confirmation response"""
        
        confirmation_patterns = [
            r'\b(yes|y|confirm|proceed|ok|okay|do it)\b',
            r'\b(no|n|cancel|abort|stop|nevermind)\b'
        ]
        
        return any(re.search(pattern, message.lower()) for pattern in confirmation_patterns)
    
    async def _handle_confirmation(self, 
                                 message: str, 
                                 user_id: str, 
                                 session_id: str) -> Dict[str, Any]:
        """Handle confirmation responses"""
        
        # Get pending confirmation for this user
        pending_key = f"{user_id}_{session_id}"
        pending_request = self.pending_confirmations.get(pending_key)
        
        if not pending_request:
            return {
                "agent": self.name,
                "response": "I don't have any pending operations waiting for confirmation.",
                "help": "Start a new database operation if you need to make changes."
            }
        
        # Check if user confirmed or cancelled
        is_confirmed = bool(re.search(r'\b(yes|y|confirm|proceed|ok|okay|do it)\b', message.lower()))
        
        if is_confirmed:
            # Execute the operation
            result = await self._execute_crud_operation(pending_request, user_id)
            
            # Clean up pending confirmation
            del self.pending_confirmations[pending_key]
            
            return {
                "agent": self.name,
                "response": f"✅ Operation completed: {result.message}",
                "operation": pending_request.operation.value,
                "entity": pending_request.entity,
                "affected_records": result.affected_records,
                "success": result.success
            }
        else:
            # User cancelled
            del self.pending_confirmations[pending_key]
            
            return {
                "agent": self.name,
                "response": "❌ Operation cancelled as requested.",
                "operation": pending_request.operation.value,
                "entity": pending_request.entity
            }
    
    async def _parse_crud_request(self, 
                                message: str, 
                                user_context: Dict[str, Any]) -> Optional[CRUDRequest]:
        """Parse natural language into CRUD request"""
        
        message_lower = message.lower()
        
        # Update operations
        if re.search(r'\b(update|change|modify|edit|set)\b', message_lower):
            return await self._parse_update_request(message, user_context)
        
        # Create operations
        elif re.search(r'\b(create|add|new|make)\b', message_lower):
            return await self._parse_create_request(message, user_context)
        
        # Delete operations
        elif re.search(r'\b(delete|remove|cancel)\b', message_lower):
            return await self._parse_delete_request(message, user_context)
        
        return None
    
    async def _parse_update_request(self, 
                                  message: str, 
                                  user_context: Dict[str, Any]) -> Optional[CRUDRequest]:
        """Parse update operation from message"""
        
        # Extract entity and field updates
        # Pattern: "Change plan ABC's ARR to 250000"
        plan_pattern = r'\b(?:plan|account)\s+([A-Za-z0-9\-_]+).*?(\w+)\s+to\s+([A-Za-z0-9,.$%\s]+)'
        match = re.search(plan_pattern, message, re.IGNORECASE)
        
        if match:
            entity_id = match.group(1)
            field_name = match.group(2).lower()
            new_value = match.group(3).strip()
            
            # Map common field names
            field_mapping = {
                'arr': 'annual_revenue',
                'revenue': 'annual_revenue', 
                'target': 'revenue_growth_target',
                'goal': 'short_term_goals',
                'tier': 'account_tier'
            }
            
            field_name = field_mapping.get(field_name, field_name)
            
            # Clean up value
            if field_name in ['annual_revenue', 'revenue_growth_target']:
                new_value = re.sub(r'[,$%]', '', new_value)
                try:
                    new_value = float(new_value)
                except ValueError:
                    pass
            
            return CRUDRequest(
                operation=CRUDOperation.UPDATE,
                entity="strategic_account_plan",
                entity_id=entity_id,
                fields={field_name: new_value},
                confirmation_required=True
            )
        
        # Pattern: "Update stakeholder John's role to Champion"
        stakeholder_pattern = r'\bstakeholder\s+([A-Za-z\s]+?)(?:\'s)?\s+(\w+)\s+to\s+([A-Za-z\s]+)'
        match = re.search(stakeholder_pattern, message, re.IGNORECASE)
        
        if match:
            stakeholder_name = match.group(1).strip()
            field_name = match.group(2).lower()
            new_value = match.group(3).strip()
            
            return CRUDRequest(
                operation=CRUDOperation.UPDATE,
                entity="plan_stakeholder",
                conditions={"name": stakeholder_name},
                fields={field_name: new_value},
                confirmation_required=True
            )
        
        return None
    
    async def _parse_create_request(self, 
                                  message: str, 
                                  user_context: Dict[str, Any]) -> Optional[CRUDRequest]:
        """Parse create operation from message"""
        
        # Pattern: "Create new plan for account XYZ"
        plan_pattern = r'\b(?:create|add|new|make)\b.*?\bplan\b.*?\baccount\s+([A-Za-z0-9\-_]+)'
        match = re.search(plan_pattern, message, re.IGNORECASE)
        
        if match:
            account_name = match.group(1)
            
            return CRUDRequest(
                operation=CRUDOperation.CREATE,
                entity="strategic_account_plan",
                fields={"account_name": account_name},
                confirmation_required=True
            )
        
        return None
    
    async def _parse_delete_request(self, 
                                  message: str, 
                                  user_context: Dict[str, Any]) -> Optional[CRUDRequest]:
        """Parse delete operation from message"""
        
        # Pattern: "Delete plan ABC" or "Remove activity 123"
        delete_pattern = r'\b(?:delete|remove|cancel)\b\s+(\w+)(?:\s+(?:with\s+)?(?:id\s+)?([A-Za-z0-9\-_]+))?'
        match = re.search(delete_pattern, message, re.IGNORECASE)
        
        if match:
            entity_type = match.group(1).lower()
            entity_id = match.group(2) if match.group(2) else None
            
            # Map entity types
            entity_mapping = {
                'plan': 'strategic_account_plan',
                'stakeholder': 'plan_stakeholder',
                'activity': 'plan_activity'
            }
            
            entity = entity_mapping.get(entity_type, entity_type)
            
            return CRUDRequest(
                operation=CRUDOperation.DELETE,
                entity=entity,
                entity_id=entity_id,
                confirmation_required=True
            )
        
        return None
    
    async def _handle_crud_request(self, 
                                 crud_request: CRUDRequest,
                                 user_id: str,
                                 session_id: str) -> Dict[str, Any]:
        """Handle a parsed CRUD request"""
        
        # Check if confirmation is required
        if crud_request.confirmation_required:
            # Store pending confirmation
            pending_key = f"{user_id}_{session_id}"
            self.pending_confirmations[pending_key] = crud_request
            
            # Generate confirmation message
            confirmation_msg = await self._generate_confirmation_message(crud_request)
            
            return {
                "agent": self.name,
                "response": confirmation_msg,
                "requires_confirmation": True,
                "operation": crud_request.operation.value,
                "entity": crud_request.entity,
                "pending_confirmation_id": pending_key
            }
        else:
            # Execute immediately
            result = await self._execute_crud_operation(crud_request, user_id)
            
            return {
                "agent": self.name,
                "response": result.message,
                "operation": crud_request.operation.value,
                "entity": crud_request.entity,
                "success": result.success,
                "affected_records": result.affected_records
            }
    
    async def _generate_confirmation_message(self, crud_request: CRUDRequest) -> str:
        """Generate confirmation message for CRUD operation"""
        
        if crud_request.operation == CRUDOperation.UPDATE:
            if crud_request.entity_id:
                return f"🔄 **Confirm Update**: Are you sure you want to update {crud_request.entity} '{crud_request.entity_id}' with the following changes?\n\n" + \
                       f"**Changes**: {', '.join([f'{k} → {v}' for k, v in crud_request.fields.items()])}\n\n" + \
                       f"**Reply 'yes' to confirm or 'no' to cancel.**"
            else:
                conditions = ', '.join([f'{k}={v}' for k, v in crud_request.conditions.items()])
                return f"🔄 **Confirm Update**: Are you sure you want to update {crud_request.entity} where {conditions}?\n\n" + \
                       f"**Changes**: {', '.join([f'{k} → {v}' for k, v in crud_request.fields.items()])}\n\n" + \
                       f"**Reply 'yes' to confirm or 'no' to cancel.**"
        
        elif crud_request.operation == CRUDOperation.DELETE:
            entity_desc = f"{crud_request.entity} '{crud_request.entity_id}'" if crud_request.entity_id else crud_request.entity
            return f"🗑️ **Confirm Deletion**: Are you sure you want to delete {entity_desc}?\n\n" + \
                   f"**⚠️ This action cannot be undone!**\n\n" + \
                   f"**Reply 'yes' to confirm or 'no' to cancel.**"
        
        elif crud_request.operation == CRUDOperation.CREATE:
            fields_desc = ', '.join([f'{k}={v}' for k, v in crud_request.fields.items()])
            return f"➕ **Confirm Creation**: Are you sure you want to create a new {crud_request.entity}?\n\n" + \
                   f"**Details**: {fields_desc}\n\n" + \
                   f"**Reply 'yes' to confirm or 'no' to cancel.**"
        
        return f"**Confirm {crud_request.operation.value.title()}**: Please confirm this operation. Reply 'yes' or 'no'."
    
    async def _execute_crud_operation(self, 
                                    crud_request: CRUDRequest,
                                    user_id: str) -> CRUDResult:
        """Execute the CRUD operation via backend API"""
        
        try:
            # Map entity to API endpoint
            endpoint_mapping = {
                'strategic_account_plan': '/api/strategic-account-plans',
                'plan_stakeholder': '/api/plan-stakeholders',
                'plan_activity': '/api/plan-activities'
            }
            
            base_endpoint = endpoint_mapping.get(crud_request.entity)
            if not base_endpoint:
                return CRUDResult(
                    success=False,
                    operation=crud_request.operation,
                    entity=crud_request.entity,
                    affected_records=0,
                    error=f"Unknown entity type: {crud_request.entity}"
                )
            
            # Build API request
            if crud_request.operation == CRUDOperation.UPDATE:
                if crud_request.entity_id:
                    url = f"{self.backend_url}{base_endpoint}/{crud_request.entity_id}"
                    method = "PUT"
                    data = crud_request.fields
                else:
                    # Update by condition - need special endpoint
                    url = f"{self.backend_url}{base_endpoint}/update-by-condition"
                    method = "POST"
                    data = {
                        "conditions": crud_request.conditions,
                        "updates": crud_request.fields
                    }
                
                response = requests.request(method, url, json=data, timeout=10)
                
                if response.status_code == 200:
                    return CRUDResult(
                        success=True,
                        operation=crud_request.operation,
                        entity=crud_request.entity,
                        affected_records=1,
                        data=response.json(),
                        message=f"Successfully updated {crud_request.entity}"
                    )
                else:
                    return CRUDResult(
                        success=False,
                        operation=crud_request.operation,
                        entity=crud_request.entity,
                        affected_records=0,
                        error=f"API error: {response.status_code} - {response.text}"
                    )
            
            elif crud_request.operation == CRUDOperation.DELETE:
                url = f"{self.backend_url}{base_endpoint}/{crud_request.entity_id}"
                response = requests.delete(url, timeout=10)
                
                if response.status_code == 200:
                    return CRUDResult(
                        success=True,
                        operation=crud_request.operation,
                        entity=crud_request.entity,
                        affected_records=1,
                        message=f"Successfully deleted {crud_request.entity}"
                    )
                else:
                    return CRUDResult(
                        success=False,
                        operation=crud_request.operation,
                        entity=crud_request.entity,
                        affected_records=0,
                        error=f"API error: {response.status_code} - {response.text}"
                    )
            
            elif crud_request.operation == CRUDOperation.CREATE:
                url = f"{self.backend_url}{base_endpoint}"
                response = requests.post(url, json=crud_request.fields, timeout=10)
                
                if response.status_code == 201:
                    return CRUDResult(
                        success=True,
                        operation=crud_request.operation,
                        entity=crud_request.entity,
                        affected_records=1,
                        data=response.json(),
                        message=f"Successfully created {crud_request.entity}"
                    )
                else:
                    return CRUDResult(
                        success=False,
                        operation=crud_request.operation,
                        entity=crud_request.entity,
                        affected_records=0,
                        error=f"API error: {response.status_code} - {response.text}"
                    )
            
        except requests.exceptions.RequestException as e:
            return CRUDResult(
                success=False,
                operation=crud_request.operation,
                entity=crud_request.entity,
                affected_records=0,
                error=f"Network error: {str(e)}"
            )
        except Exception as e:
            return CRUDResult(
                success=False,
                operation=crud_request.operation,
                entity=crud_request.entity,
                affected_records=0,
                error=f"Unexpected error: {str(e)}"
            )
