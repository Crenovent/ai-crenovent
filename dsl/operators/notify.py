"""
Notify Operator - Handles notifications and user responses
"""

import json
from typing import Dict, List, Any, Optional
from .base import BaseOperator, OperatorContext, OperatorResult
import logging

logger = logging.getLogger(__name__)

class NotifyOperator(BaseOperator):
    """
    Notify operator for sending notifications and responses
    Supports:
    - UI responses (immediate user feedback)
    - Email notifications
    - Slack messages
    - Webhook calls
    - System logs
    """
    
    def __init__(self, config=None):
        super().__init__("notify_operator")
        self.config = config or {}
    
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate notify operator configuration"""
        errors = []
        
        # Check required fields
        if 'channel' not in config:
            errors.append("'channel' is required")
        elif config['channel'] not in ['ui_response', 'email', 'slack', 'webhook', 'log']:
            errors.append(f"Unsupported channel: {config['channel']}")
        
        if 'template' not in config and 'message' not in config:
            errors.append("Either 'template' or 'message' is required")
        
        # Channel-specific validation
        channel = config.get('channel')
        if channel == 'email':
            if 'to' not in config:
                errors.append("'to' field required for email channel")
        elif channel == 'slack':
            if 'to' not in config and 'channel_id' not in config:
                errors.append("Either 'to' or 'channel_id' required for slack")
        elif channel == 'webhook':
            if 'url' not in config:
                errors.append("'url' required for webhook channel")
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute the notification"""
        channel = config['channel']
        
        try:
            if channel == 'ui_response':
                return await self._send_ui_response(context, config)
            elif channel == 'email':
                return await self._send_email(context, config)
            elif channel == 'slack':
                return await self._send_slack(context, config)
            elif channel == 'webhook':
                return await self._send_webhook(context, config)
            elif channel == 'log':
                return await self._send_log(context, config)
            else:
                return OperatorResult(
                    success=False,
                    error_message=f"Unsupported channel: {channel}"
                )
                
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Notification failed: {e}"
            )
    
    async def _send_ui_response(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Send immediate UI response to user"""
        try:
            # Process message template
            message = self._process_message_template(config, context)
            
            # Prepare attachments
            attachments = self._process_attachments(config, context)
            
            # Create UI response
            ui_response = {
                'message': message,
                'type': config.get('message_type', 'success'),
                'attachments': attachments,
                'timestamp': context.input_data.get('timestamp'),
                'user_id': context.user_id,
                'session_id': context.session_id
            }
            
            # Add any custom UI elements
            if 'ui_elements' in config:
                ui_response['ui_elements'] = config['ui_elements']
            
            return OperatorResult(
                success=True,
                output_data={
                    'ui_response': ui_response,
                    'channel': 'ui_response',
                    'message_sent': True
                },
                confidence_score=1.0
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"UI response failed: {e}"
            )
    
    async def _send_email(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Send email notification"""
        try:
            # This would integrate with your email service
            # For now, log the email that would be sent
            
            message = self._process_message_template(config, context)
            recipients = config.get('to', [])
            subject = config.get('subject', 'RevAI Pro Notification')
            
            # Log the email (in production, this would actually send)
            logger.info(f"Email notification: To={recipients}, Subject={subject}, Message={message}")
            
            return OperatorResult(
                success=True,
                output_data={
                    'email_sent': True,
                    'recipients': recipients,
                    'subject': subject,
                    'message': message,
                    'channel': 'email'
                },
                confidence_score=0.95  # Slightly lower due to external dependency
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Email notification failed: {e}"
            )
    
    async def _send_slack(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Send Slack notification"""
        try:
            # This would integrate with Slack API
            # For now, log the Slack message that would be sent
            
            message = self._process_message_template(config, context)
            channel_id = config.get('channel_id', config.get('to', '#general'))
            
            # Log the Slack message (in production, this would actually send)
            logger.info(f"Slack notification: Channel={channel_id}, Message={message}")
            
            return OperatorResult(
                success=True,
                output_data={
                    'slack_sent': True,
                    'channel': channel_id,
                    'message': message,
                    'notification_type': 'slack'
                },
                confidence_score=0.90
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Slack notification failed: {e}"
            )
    
    async def _send_webhook(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Send webhook notification"""
        try:
            # This would make HTTP POST to webhook URL
            # For now, log the webhook that would be sent
            
            message = self._process_message_template(config, context)
            url = config['url']
            payload = {
                'message': message,
                'user_id': context.user_id,
                'tenant_id': context.tenant_id,
                'workflow_id': context.workflow_id,
                'timestamp': context.input_data.get('timestamp')
            }
            
            # Add custom payload data
            if 'payload' in config:
                payload.update(config['payload'])
            
            # Log the webhook (in production, this would actually send)
            logger.info(f"Webhook notification: URL={url}, Payload={json.dumps(payload)}")
            
            return OperatorResult(
                success=True,
                output_data={
                    'webhook_sent': True,
                    'url': url,
                    'payload': payload,
                    'channel': 'webhook'
                },
                confidence_score=0.85
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Webhook notification failed: {e}"
            )
    
    async def _send_log(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Send log notification"""
        try:
            message = self._process_message_template(config, context)
            log_level = config.get('level', 'info').upper()
            
            # Log the message at specified level
            if log_level == 'DEBUG':
                logger.debug(message)
            elif log_level == 'INFO':
                logger.info(message)
            elif log_level == 'WARNING':
                logger.warning(message)
            elif log_level == 'ERROR':
                logger.error(message)
            else:
                logger.info(message)
            
            return OperatorResult(
                success=True,
                output_data={
                    'log_sent': True,
                    'message': message,
                    'level': log_level,
                    'channel': 'log'
                },
                confidence_score=1.0
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Log notification failed: {e}"
            )
    
    def _process_message_template(self, config: Dict[str, Any], context: OperatorContext) -> str:
        """Process message template with variable substitution"""
        template = config.get('template', config.get('message', ''))
        
        # Replace template variables
        template_vars = {
            'user_id': context.user_id,
            'tenant_id': context.tenant_id,
            'workflow_id': context.workflow_id,
            'step_id': context.step_id,
            **context.previous_outputs
        }
        
        # Simple template variable replacement
        for var_name, var_value in template_vars.items():
            placeholder = f"{{{{{var_name}}}}}"
            if placeholder in template:
                template = template.replace(placeholder, str(var_value))
        
        # Handle nested object access like {{result.field}}
        import re
        nested_pattern = r'\{\{([^}]+)\}\}'
        
        def replace_nested(match):
            var_path = match.group(1).strip()
            try:
                value = template_vars
                for part in var_path.split('.'):
                    if isinstance(value, dict):
                        value = value.get(part, '')
                    elif hasattr(value, part):
                        value = getattr(value, part)
                    else:
                        return match.group(0)  # Return original if not found
                return str(value)
            except:
                return match.group(0)  # Return original if error
        
        processed_template = re.sub(nested_pattern, replace_nested, template)
        
        return processed_template
    
    def _process_attachments(self, config: Dict[str, Any], context: OperatorContext) -> List[Dict[str, Any]]:
        """Process notification attachments"""
        attachments = config.get('attachments', [])
        processed_attachments = []
        
        for attachment in attachments:
            if isinstance(attachment, dict):
                processed_attachment = {
                    'name': attachment.get('name', 'attachment'),
                    'type': attachment.get('type', 'data'),
                    'data': attachment.get('data', {}),
                    'description': attachment.get('description', '')
                }
                
                # Process data references
                if 'ref' in attachment:
                    ref_path = attachment['ref']
                    # Try to resolve reference from previous outputs
                    try:
                        value = context.previous_outputs
                        for part in ref_path.split('.'):
                            value = value.get(part, {})
                        processed_attachment['data'] = value
                    except:
                        processed_attachment['data'] = {}
                
                processed_attachments.append(processed_attachment)
        
        return processed_attachments
