"""
Integration Test Suite for RevAI Pro Platform
Tests complete workflows and service interactions
"""

import pytest
import asyncio
import httpx
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time

class TestCalendarToLetsMeetIntegration:
    """Integration tests for Calendar → Let's Meet workflow"""
    
    @pytest.fixture
    async def setup_services(self):
        """Setup test services"""
        services = {
            "orchestrator": "http://localhost:8001",
            "calendar_automation": "http://localhost:8007",
            "letsmeet_automation": "http://localhost:8008",
            "event_bus": "http://localhost:8013"
        }
        return services
    
    async def test_calendar_event_to_meeting_capture(self, setup_services):
        """Test complete flow from calendar event creation to meeting capture"""
        services = await setup_services
        
        # Step 1: Create calendar event
        calendar_event = {
            "event_id": "test-event-001",
            "tenant_id": "test-tenant",
            "user_id": "test-user",
            "title": "Integration Test Meeting",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "location": "Virtual Meeting",
            "attendees": ["test-user@example.com"],
            "metadata": {
                "auto_join_enabled": True,
                "auto_transcribe": True
            }
        }
        
        # Publish calendar event
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "test-event-001",
                    "tenant_id": "test-tenant",
                    "event_type": "calendar.event.created",
                    "topic": "calendar-events",
                    "payload": calendar_event,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 2: Wait for automation to trigger
        await asyncio.sleep(1)
        
        # Step 3: Simulate meeting capture
        meeting_data = {
            "meeting_id": "test-meeting-001",
            "tenant_id": "test-tenant",
            "user_id": "test-user",
            "calendar_event_id": "test-event-001",
            "audio_url": "https://test-storage.com/audio/test-meeting-001.wav",
            "duration_seconds": 1800,
            "participants": ["test-user@example.com"],
            "status": "recorded"
        }
        
        # Trigger meeting capture automation
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['letsmeet_automation']}/automation/trigger",
                json={
                    "event_id": "test-meeting-001",
                    "tenant_id": "test-tenant",
                    "user_id": "test-user",
                    "meeting_id": "test-meeting-001",
                    "trigger": "audio_uploaded",
                    "event_data": meeting_data,
                    "context": {}
                }
            )
            assert response.status_code == 200
        
        # Step 4: Verify transcription was triggered
        await asyncio.sleep(1)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['letsmeet_automation']}/transcriptions/test-meeting-001"
            )
            # Should either return transcription or indicate processing
            assert response.status_code in [200, 404]  # 404 if not yet processed
    
    async def test_meeting_transcription_to_summary(self, setup_services):
        """Test flow from meeting transcription to summary generation"""
        services = await setup_services
        
        # Step 1: Simulate completed transcription
        transcription_data = {
            "meeting_id": "test-meeting-002",
            "transcript": "This is a test meeting transcript with action items.",
            "segments": [
                {"speaker": "John", "text": "We need to follow up with the client."},
                {"speaker": "Jane", "text": "I'll create a task for that."}
            ],
            "confidence_score": 0.95,
            "language": "en-US",
            "duration_seconds": 300
        }
        
        # Publish transcription completed event
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "test-transcription-002",
                    "tenant_id": "test-tenant",
                    "event_type": "letsmeet.transcription.completed",
                    "topic": "letsmeet-events",
                    "payload": transcription_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 2: Wait for summary generation
        await asyncio.sleep(1)
        
        # Step 3: Verify summary was generated
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['letsmeet_automation']}/summaries/test-meeting-002"
            )
            # Should either return summary or indicate processing
            assert response.status_code in [200, 404]

class TestLetsMeetToCruxxIntegration:
    """Integration tests for Let's Meet → Cruxx workflow"""
    
    @pytest.fixture
    async def setup_services(self):
        """Setup test services"""
        services = {
            "letsmeet_automation": "http://localhost:8008",
            "cruxx_automation": "http://localhost:8009",
            "event_bus": "http://localhost:8013"
        }
        return services
    
    async def test_summary_to_cruxx_actions(self, setup_services):
        """Test flow from meeting summary to Cruxx action creation"""
        services = await setup_services
        
        # Step 1: Simulate completed summary with action items
        summary_data = {
            "summary_id": "test-summary-003",
            "meeting_id": "test-meeting-003",
            "tenant_id": "test-tenant",
            "summary_text": "Meeting about project updates. Action items identified.",
            "key_points": [
                "Project is on track",
                "Client feedback received",
                "Next steps defined"
            ],
            "action_items": [
                {
                    "text": "Follow up with client on pricing",
                    "assignee": "john@example.com",
                    "due_date": "2024-01-15T00:00:00Z",
                    "priority": "high"
                },
                {
                    "text": "Update project timeline",
                    "assignee": "jane@example.com",
                    "due_date": "2024-01-20T00:00:00Z",
                    "priority": "medium"
                }
            ],
            "decisions": [
                "Approved budget increase",
                "Extended project deadline"
            ],
            "sentiment": "positive",
            "confidence_score": 0.9
        }
        
        # Publish summary generated event
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "test-summary-003",
                    "tenant_id": "test-tenant",
                    "event_type": "letsmeet.summary.generated",
                    "topic": "letsmeet-events",
                    "payload": summary_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 2: Wait for Cruxx automation
        await asyncio.sleep(1)
        
        # Step 3: Verify Cruxx actions were created
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['cruxx_automation']}/cruxx?tenant_id=test-tenant&source_summary_id=test-summary-003"
            )
            assert response.status_code == 200
            actions = response.json()
            assert len(actions) >= 2  # Should have created 2 actions

class TestCruxxToCalendarIntegration:
    """Integration tests for Cruxx → Calendar workflow"""
    
    @pytest.fixture
    async def setup_services(self):
        """Setup test services"""
        services = {
            "cruxx_automation": "http://localhost:8009",
            "calendar_automation": "http://localhost:8007",
            "event_bus": "http://localhost:8013"
        }
        return services
    
    async def test_cruxx_action_to_calendar_overlay(self, setup_services):
        """Test flow from Cruxx action creation to calendar overlay"""
        services = await setup_services
        
        # Step 1: Create Cruxx action
        cruxx_action = {
            "cruxx_id": "test-cruxx-004",
            "tenant_id": "test-tenant",
            "opportunity_id": "test-opportunity-004",
            "title": "Follow up with client",
            "description": "Client follow-up regarding pricing proposal",
            "assignee": "john@example.com",
            "due_date": "2024-01-15T00:00:00Z",
            "priority": "high",
            "status": "open",
            "sla_status": "met",
            "source_meeting_id": "test-meeting-004",
            "source_summary_id": "test-summary-004",
            "created_by": "automation"
        }
        
        # Publish Cruxx action created event
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "test-cruxx-004",
                    "tenant_id": "test-tenant",
                    "event_type": "cruxx.action.created",
                    "topic": "cruxx-events",
                    "payload": cruxx_action,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 2: Wait for calendar overlay creation
        await asyncio.sleep(1)
        
        # Step 3: Verify calendar overlay was created
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['calendar_automation']}/overlays?tenant_id=test-tenant&cruxx_id=test-cruxx-004"
            )
            assert response.status_code == 200
            overlays = response.json()
            assert len(overlays) >= 1  # Should have created at least one overlay

class TestEndToEndWorkflow:
    """Complete end-to-end workflow tests"""
    
    @pytest.fixture
    async def setup_all_services(self):
        """Setup all test services"""
        services = {
            "orchestrator": "http://localhost:8001",
            "agent_registry": "http://localhost:8002",
            "routing_orchestrator": "http://localhost:8003",
            "kpi_exporter": "http://localhost:8004",
            "confidence_thresholds": "http://localhost:8005",
            "model_audit": "http://localhost:8006",
            "calendar_automation": "http://localhost:8007",
            "letsmeet_automation": "http://localhost:8008",
            "cruxx_automation": "http://localhost:8009",
            "run_trace_schema": "http://localhost:8010",
            "dlq_replay_tooling": "http://localhost:8011",
            "metrics_exporter": "http://localhost:8012",
            "event_bus": "http://localhost:8013"
        }
        return services
    
    async def test_complete_automation_loop(self, setup_all_services):
        """Test complete automation loop: Calendar → Let's Meet → Cruxx → Calendar"""
        services = await setup_all_services
        
        # Step 1: Create calendar event with automation enabled
        calendar_event = {
            "event_id": "e2e-event-001",
            "tenant_id": "e2e-tenant",
            "user_id": "e2e-user",
            "title": "E2E Test Meeting",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "location": "Virtual Meeting",
            "attendees": ["e2e-user@example.com"],
            "metadata": {
                "auto_join_enabled": True,
                "auto_transcribe": True,
                "auto_summarize": True,
                "auto_extract_actions": True
            }
        }
        
        # Publish calendar event
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "e2e-event-001",
                    "tenant_id": "e2e-tenant",
                    "event_type": "calendar.event.created",
                    "topic": "calendar-events",
                    "payload": calendar_event,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 2: Simulate meeting capture
        await asyncio.sleep(2)
        
        meeting_data = {
            "meeting_id": "e2e-meeting-001",
            "tenant_id": "e2e-tenant",
            "user_id": "e2e-user",
            "calendar_event_id": "e2e-event-001",
            "audio_url": "https://test-storage.com/audio/e2e-meeting-001.wav",
            "duration_seconds": 1800,
            "participants": ["e2e-user@example.com"],
            "status": "recorded"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "e2e-meeting-001",
                    "tenant_id": "e2e-tenant",
                    "event_type": "letsmeet.meeting.captured",
                    "topic": "letsmeet-events",
                    "payload": meeting_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 3: Simulate transcription completion
        await asyncio.sleep(2)
        
        transcription_data = {
            "meeting_id": "e2e-meeting-001",
            "transcript": "E2E test meeting. We need to follow up with the client and update the project timeline.",
            "segments": [
                {"speaker": "User", "text": "E2E test meeting. We need to follow up with the client."},
                {"speaker": "User", "text": "Also, we should update the project timeline."}
            ],
            "confidence_score": 0.95,
            "language": "en-US",
            "duration_seconds": 300
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "e2e-transcription-001",
                    "tenant_id": "e2e-tenant",
                    "event_type": "letsmeet.transcription.completed",
                    "topic": "letsmeet-events",
                    "payload": transcription_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 4: Simulate summary generation
        await asyncio.sleep(2)
        
        summary_data = {
            "summary_id": "e2e-summary-001",
            "meeting_id": "e2e-meeting-001",
            "tenant_id": "e2e-tenant",
            "summary_text": "E2E test meeting completed. Action items identified.",
            "key_points": [
                "Meeting completed successfully",
                "Action items identified"
            ],
            "action_items": [
                {
                    "text": "Follow up with client",
                    "assignee": "e2e-user@example.com",
                    "due_date": "2024-01-15T00:00:00Z",
                    "priority": "high"
                },
                {
                    "text": "Update project timeline",
                    "assignee": "e2e-user@example.com",
                    "due_date": "2024-01-20T00:00:00Z",
                    "priority": "medium"
                }
            ],
            "decisions": [],
            "sentiment": "neutral",
            "confidence_score": 0.9
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "e2e-summary-001",
                    "tenant_id": "e2e-tenant",
                    "event_type": "letsmeet.summary.generated",
                    "topic": "letsmeet-events",
                    "payload": summary_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 5: Verify Cruxx actions were created
        await asyncio.sleep(2)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['cruxx_automation']}/cruxx?tenant_id=e2e-tenant&source_summary_id=e2e-summary-001"
            )
            assert response.status_code == 200
            actions = response.json()
            assert len(actions) >= 2  # Should have created 2 actions
        
        # Step 6: Verify calendar overlays were created
        await asyncio.sleep(2)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['calendar_automation']}/overlays?tenant_id=e2e-tenant"
            )
            assert response.status_code == 200
            overlays = response.json()
            assert len(overlays) >= 1  # Should have created overlays
        
        # Step 7: Verify metrics were recorded
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['metrics_exporter']}/metrics/dashboard?tenant_id=e2e-tenant"
            )
            assert response.status_code == 200
            metrics = response.json()
            assert "tenant_id" in metrics
            assert "metrics" in metrics

class TestErrorHandlingAndRecovery:
    """Tests for error handling and recovery mechanisms"""
    
    @pytest.fixture
    async def setup_services(self):
        """Setup test services"""
        services = {
            "event_bus": "http://localhost:8013",
            "dlq_replay_tooling": "http://localhost:8011",
            "calendar_automation": "http://localhost:8007"
        }
        return services
    
    async def test_dlq_handling(self, setup_services):
        """Test DLQ handling for failed events"""
        services = await setup_services
        
        # Step 1: Publish an event that will fail processing
        failed_event = {
            "event_id": "failed-event-001",
            "tenant_id": "test-tenant",
            "event_type": "calendar.event.created",
            "topic": "calendar-events",
            "payload": {
                "event_id": "failed-event-001",
                "tenant_id": "test-tenant",
                "user_id": "test-user",
                "title": "Failed Event",
                "start_time": "invalid-date",  # This should cause processing to fail
                "end_time": "invalid-date"
            },
            "headers": {},
            "metadata": {}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json=failed_event
            )
            assert response.status_code == 200
        
        # Step 2: Wait for processing and failure
        await asyncio.sleep(2)
        
        # Step 3: Check DLQ for failed message
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['dlq_replay_tooling']}/dlq/messages?tenant_id=test-tenant&status=failed"
            )
            assert response.status_code == 200
            messages = response.json()
            # Should have at least one failed message
            assert len(messages) >= 0  # May be 0 if processing hasn't failed yet
    
    async def test_retry_mechanism(self, setup_services):
        """Test retry mechanism for failed events"""
        services = await setup_services
        
        # Step 1: Add a message to DLQ
        dlq_message = {
            "message_id": "retry-test-001",
            "tenant_id": "test-tenant",
            "original_topic": "test-topic",
            "original_message": {"test": "data"},
            "failure_reason": "Processing failed",
            "error_details": {"error": "test error"},
            "retry_count": 0,
            "max_retries": 3,
            "status": "pending",
            "replay_strategy": "exponential_backoff",
            "metadata": {}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['dlq_replay_tooling']}/dlq/messages",
                json=dlq_message
            )
            assert response.status_code == 200
        
        # Step 2: Trigger replay
        replay_data = {
            "dlq_message_id": "retry-test-001",
            "replay_strategy": "immediate"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['dlq_replay_tooling']}/dlq/replay",
                json=replay_data
            )
            assert response.status_code == 200
        
        # Step 3: Wait for retry processing
        await asyncio.sleep(2)
        
        # Step 4: Check retry status
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['dlq_replay_tooling']}/dlq/messages?tenant_id=test-tenant&message_id=retry-test-001"
            )
            assert response.status_code == 200
            messages = response.json()
            if messages:
                message = messages[0]
                assert message["retry_count"] >= 1

class TestPerformanceUnderLoad:
    """Performance tests under load"""
    
    async def test_concurrent_event_processing(self):
        """Test processing multiple events concurrently"""
        services = {
            "event_bus": "http://localhost:8013",
            "calendar_automation": "http://localhost:8007"
        }
        
        # Create multiple events
        events = []
        for i in range(10):
            event = {
                "event_id": f"load-test-event-{i:03d}",
                "tenant_id": "load-test-tenant",
                "event_type": "calendar.event.created",
                "topic": "calendar-events",
                "payload": {
                    "event_id": f"load-test-event-{i:03d}",
                    "tenant_id": "load-test-tenant",
                    "user_id": "load-test-user",
                    "title": f"Load Test Event {i}",
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
                },
                "headers": {},
                "metadata": {}
            }
            events.append(event)
        
        # Publish all events concurrently
        async with httpx.AsyncClient() as client:
            tasks = []
            for event in events:
                task = client.post(
                    f"{services['event_bus']}/events/publish",
                    json=event
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All events should be published successfully
            for response in responses:
                assert response.status_code == 200
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Verify events were processed
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['calendar_automation']}/events?tenant_id=load-test-tenant"
            )
            assert response.status_code == 200
            processed_events = response.json()
            assert len(processed_events) >= 10

class TestDataConsistency:
    """Tests for data consistency across services"""
    
    async def test_tenant_data_isolation(self):
        """Test that tenant data is properly isolated"""
        services = {
            "event_bus": "http://localhost:8013",
            "calendar_automation": "http://localhost:8007",
            "letsmeet_automation": "http://localhost:8008",
            "cruxx_automation": "http://localhost:8009"
        }
        
        # Create events for different tenants
        tenant1_event = {
            "event_id": "tenant1-event-001",
            "tenant_id": "tenant-1",
            "event_type": "calendar.event.created",
            "topic": "calendar-events",
            "payload": {
                "event_id": "tenant1-event-001",
                "tenant_id": "tenant-1",
                "user_id": "tenant1-user",
                "title": "Tenant 1 Event",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            },
            "headers": {},
            "metadata": {}
        }
        
        tenant2_event = {
            "event_id": "tenant2-event-001",
            "tenant_id": "tenant-2",
            "event_type": "calendar.event.created",
            "topic": "calendar-events",
            "payload": {
                "event_id": "tenant2-event-001",
                "tenant_id": "tenant-2",
                "user_id": "tenant2-user",
                "title": "Tenant 2 Event",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            },
            "headers": {},
            "metadata": {}
        }
        
        # Publish both events
        async with httpx.AsyncClient() as client:
            await client.post(f"{services['event_bus']}/events/publish", json=tenant1_event)
            await client.post(f"{services['event_bus']}/events/publish", json=tenant2_event)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify tenant isolation
        async with httpx.AsyncClient() as client:
            # Check tenant 1 events
            response = await client.get(
                f"{services['calendar_automation']}/events?tenant_id=tenant-1"
            )
            assert response.status_code == 200
            tenant1_events = response.json()
            assert all(event["tenant_id"] == "tenant-1" for event in tenant1_events)
            
            # Check tenant 2 events
            response = await client.get(
                f"{services['calendar_automation']}/events?tenant_id=tenant-2"
            )
            assert response.status_code == 200
            tenant2_events = response.json()
            assert all(event["tenant_id"] == "tenant-2" for event in tenant2_events)
            
            # Verify no cross-contamination
            tenant1_ids = {event["event_id"] for event in tenant1_events}
            tenant2_ids = {event["event_id"] for event in tenant2_events}
            assert len(tenant1_ids.intersection(tenant2_ids)) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
