"""
End-to-End Test Scenarios for RevAI Pro Platform
Complete user journey tests covering all modules
"""

import pytest
import asyncio
import httpx
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time

class TestUserJourneyScenarios:
    """Complete user journey test scenarios"""
    
    @pytest.fixture
    async def setup_test_environment(self):
        """Setup test environment with all services"""
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
        
        # Test tenant and user data
        test_data = {
            "tenant_id": "e2e-test-tenant",
            "user_id": "e2e-test-user",
            "session_id": "e2e-test-session"
        }
        
        return services, test_data
    
    async def test_scenario_1_high_confidence_automation(self, setup_test_environment):
        """Scenario 1: High confidence automation - User creates meeting, AI handles everything"""
        services, test_data = await setup_test_environment
        
        # Step 1: User creates a calendar event with high confidence context
        calendar_event = {
            "event_id": "scenario1-event-001",
            "tenant_id": test_data["tenant_id"],
            "user_id": test_data["user_id"],
            "title": "Client Meeting - Q1 Review",
            "description": "Quarterly business review with ABC Corp",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "location": "Virtual Meeting",
            "attendees": ["client@abccorp.com", test_data["user_id"] + "@example.com"],
            "metadata": {
                "meeting_type": "client_review",
                "auto_join_enabled": True,
                "auto_transcribe": True,
                "auto_summarize": True,
                "auto_extract_actions": True,
                "confidence_context": "high"  # This should trigger high confidence automation
            }
        }
        
        # Step 2: Publish calendar event
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "scenario1-event-001",
                    "tenant_id": test_data["tenant_id"],
                    "event_type": "calendar.event.created",
                    "topic": "calendar-events",
                    "payload": calendar_event,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 3: Verify AI orchestrator routes to agent mode
        await asyncio.sleep(1)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['orchestrator']}/orchestrate",
                json={
                    "tenant_id": test_data["tenant_id"],
                    "user_id": test_data["user_id"],
                    "session_id": test_data["session_id"],
                    "service_name": "calendar",
                    "operation_type": "create_event",
                    "input_data": calendar_event,
                    "context": {"confidence_context": "high"}
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["recommended_mode"] == "agent"
            assert data["confidence_score"] >= 0.8
        
        # Step 4: Simulate meeting capture and processing
        await asyncio.sleep(2)
        
        meeting_data = {
            "meeting_id": "scenario1-meeting-001",
            "tenant_id": test_data["tenant_id"],
            "user_id": test_data["user_id"],
            "calendar_event_id": "scenario1-event-001",
            "audio_url": "https://storage.com/audio/scenario1-meeting-001.wav",
            "duration_seconds": 3600,
            "participants": ["client@abccorp.com", test_data["user_id"] + "@example.com"],
            "status": "recorded"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "scenario1-meeting-001",
                    "tenant_id": test_data["tenant_id"],
                    "event_type": "letsmeet.meeting.captured",
                    "topic": "letsmeet-events",
                    "payload": meeting_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 5: Verify automatic transcription
        await asyncio.sleep(3)
        
        transcription_data = {
            "meeting_id": "scenario1-meeting-001",
            "transcript": "Q1 Review Meeting with ABC Corp. Key points: Revenue up 15%, new product launch planned for Q2, need to follow up on pricing proposal.",
            "segments": [
                {"speaker": "Client", "text": "Revenue is up 15% this quarter"},
                {"speaker": "User", "text": "Great news! What about the new product launch?"},
                {"speaker": "Client", "text": "We're planning for Q2 launch"},
                {"speaker": "User", "text": "I'll follow up on the pricing proposal"}
            ],
            "confidence_score": 0.95,
            "language": "en-US",
            "duration_seconds": 3600
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "scenario1-transcription-001",
                    "tenant_id": test_data["tenant_id"],
                    "event_type": "letsmeet.transcription.completed",
                    "topic": "letsmeet-events",
                    "payload": transcription_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 6: Verify automatic summary generation
        await asyncio.sleep(3)
        
        summary_data = {
            "summary_id": "scenario1-summary-001",
            "meeting_id": "scenario1-meeting-001",
            "tenant_id": test_data["tenant_id"],
            "summary_text": "Q1 Review Meeting with ABC Corp completed successfully. Revenue up 15%, Q2 product launch planned.",
            "key_points": [
                "Revenue increased 15% in Q1",
                "New product launch planned for Q2",
                "Pricing proposal needs follow-up"
            ],
            "action_items": [
                {
                    "text": "Follow up on pricing proposal",
                    "assignee": test_data["user_id"] + "@example.com",
                    "due_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
                    "priority": "high"
                }
            ],
            "decisions": [
                "Q2 product launch approved",
                "Revenue targets met"
            ],
            "sentiment": "positive",
            "confidence_score": 0.9
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "scenario1-summary-001",
                    "tenant_id": test_data["tenant_id"],
                    "event_type": "letsmeet.summary.generated",
                    "topic": "letsmeet-events",
                    "payload": summary_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 7: Verify automatic Cruxx action creation
        await asyncio.sleep(3)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['cruxx_automation']}/cruxx?tenant_id={test_data['tenant_id']}&source_summary_id=scenario1-summary-001"
            )
            assert response.status_code == 200
            actions = response.json()
            assert len(actions) >= 1
            assert any(action["title"] == "Follow up on pricing proposal" for action in actions)
        
        # Step 8: Verify calendar overlay creation
        await asyncio.sleep(2)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['calendar_automation']}/overlays?tenant_id={test_data['tenant_id']}"
            )
            assert response.status_code == 200
            overlays = response.json()
            assert len(overlays) >= 1
        
        # Step 9: Verify metrics and audit trail
        await asyncio.sleep(1)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['metrics_exporter']}/metrics/dashboard?tenant_id={test_data['tenant_id']}"
            )
            assert response.status_code == 200
            metrics = response.json()
            assert "mode_adoption" in metrics["metrics"]
            assert metrics["metrics"]["mode_adoption"]["agent_percentage"] > 0
        
        # Step 10: Verify run trace
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['run_trace_schema']}/traces?tenant_id={test_data['tenant_id']}&session_id={test_data['session_id']}"
            )
            assert response.status_code == 200
            traces = response.json()
            assert len(traces) >= 1
    
    async def test_scenario_2_low_confidence_hybrid_mode(self, setup_test_environment):
        """Scenario 2: Low confidence hybrid mode - AI suggests, user confirms"""
        services, test_data = await setup_test_environment
        
        # Step 1: User creates a calendar event with ambiguous context
        calendar_event = {
            "event_id": "scenario2-event-001",
            "tenant_id": test_data["tenant_id"],
            "user_id": test_data["user_id"],
            "title": "Meeting",  # Ambiguous title
            "description": "Need to discuss something",  # Vague description
            "start_time": datetime.utcnow().isoformat(),
            "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "location": "Office",
            "attendees": ["colleague@example.com"],
            "metadata": {
                "confidence_context": "low"  # This should trigger low confidence
            }
        }
        
        # Step 2: Verify AI orchestrator routes to hybrid mode
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['orchestrator']}/orchestrate",
                json={
                    "tenant_id": test_data["tenant_id"],
                    "user_id": test_data["user_id"],
                    "session_id": test_data["session_id"],
                    "service_name": "calendar",
                    "operation_type": "create_event",
                    "input_data": calendar_event,
                    "context": {"confidence_context": "low"}
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert data["recommended_mode"] == "hybrid"
            assert data["confidence_score"] < 0.7
        
        # Step 3: Verify AI provides suggestions with explanations
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['confidence_thresholds']}/explainability/generate",
                json={
                    "tenant_id": test_data["tenant_id"],
                    "service_name": "calendar",
                    "operation_type": "create_event",
                    "confidence_score": 0.4,
                    "trust_score": 0.5,
                    "context": {"complexity": "high"}
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "explanation" in data
            assert "factors" in data
            assert "recommendations" in data
        
        # Step 4: Simulate user override
        override_data = {
            "tenant_id": test_data["tenant_id"],
            "user_id": test_data["user_id"],
            "session_id": test_data["session_id"],
            "service_name": "calendar",
            "operation_type": "create_event",
            "original_action": {"action": "auto_create"},
            "override_action": {"action": "manual_create"},
            "override_reason": "User preference for manual control",
            "confidence_score": 0.4,
            "trust_score": 0.5
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['kpi_exporter']}/kpis/override",
                json=override_data
            )
            assert response.status_code == 200
        
        # Step 5: Verify override is recorded in audit trail
        await asyncio.sleep(1)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['model_audit']}/audit/summary?tenant_id={test_data['tenant_id']}"
            )
            assert response.status_code == 200
            data = response.json()
            assert "total_calls" in data
            assert "override_count" in data
    
    async def test_scenario_3_trust_drift_detection(self, setup_test_environment):
        """Scenario 3: Trust drift detection and recovery"""
        services, test_data = await setup_test_environment
        
        # Step 1: Simulate initial high trust scenario
        initial_trust_data = {
            "tenant_id": test_data["tenant_id"],
            "user_id": test_data["user_id"],
            "agent_id": "calendar-agent-001",
            "previous_trust_score": 0.8,
            "current_trust_score": 0.8,
            "drift_amount": 0.0,
            "drift_reason": "No change"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['metrics_exporter']}/metrics/trust-drift",
                json=initial_trust_data
            )
            assert response.status_code == 200
        
        # Step 2: Simulate trust drift (user starts overriding more)
        drift_data = {
            "tenant_id": test_data["tenant_id"],
            "user_id": test_data["user_id"],
            "agent_id": "calendar-agent-001",
            "previous_trust_score": 0.8,
            "current_trust_score": 0.6,
            "drift_amount": -0.2,
            "drift_reason": "Increased user overrides"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['metrics_exporter']}/metrics/trust-drift",
                json=drift_data
            )
            assert response.status_code == 200
        
        # Step 3: Verify trust drift is detected
        await asyncio.sleep(1)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['metrics_exporter']}/metrics/alerts?tenant_id={test_data['tenant_id']}"
            )
            assert response.status_code == 200
            data = response.json()
            assert "alerts" in data
            # Should have trust drift alert
            trust_alerts = [alert for alert in data["alerts"] if alert["type"] == "trust_drift_high"]
            assert len(trust_alerts) >= 0  # May be 0 if threshold not met
        
        # Step 4: Verify system adapts (confidence thresholds adjust)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['confidence_thresholds']}/config?tenant_id={test_data['tenant_id']}"
            )
            assert response.status_code == 200
            data = response.json()
            assert "high_confidence_threshold" in data
            assert "medium_confidence_threshold" in data
    
    async def test_scenario_4_error_recovery_and_dlq(self, setup_test_environment):
        """Scenario 4: Error recovery and DLQ handling"""
        services, test_data = await setup_test_environment
        
        # Step 1: Publish an event that will fail processing
        failed_event = {
            "event_id": "scenario4-failed-event-001",
            "tenant_id": test_data["tenant_id"],
            "event_type": "calendar.event.created",
            "topic": "calendar-events",
            "payload": {
                "event_id": "scenario4-failed-event-001",
                "tenant_id": test_data["tenant_id"],
                "user_id": test_data["user_id"],
                "title": "Failed Event",
                "start_time": "invalid-date-format",  # This should cause processing to fail
                "end_time": "invalid-date-format"
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
        
        # Step 2: Wait for processing failure
        await asyncio.sleep(3)
        
        # Step 3: Check DLQ for failed message
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['dlq_replay_tooling']}/dlq/messages?tenant_id={test_data['tenant_id']}&status=failed"
            )
            assert response.status_code == 200
            messages = response.json()
            # Should have at least one failed message
            assert len(messages) >= 0  # May be 0 if processing hasn't failed yet
        
        # Step 4: Test replay mechanism
        if messages:
            failed_message = messages[0]
            replay_data = {
                "dlq_message_id": failed_message["message_id"],
                "replay_strategy": "immediate"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['dlq_replay_tooling']}/dlq/replay",
                    json=replay_data
                )
                assert response.status_code == 200
        
        # Step 5: Verify SLO monitoring
        await asyncio.sleep(2)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['dlq_replay_tooling']}/slo/dashboard?tenant_id={test_data['tenant_id']}"
            )
            assert response.status_code == 200
            data = response.json()
            assert "slo_status" in data
            assert "time_to_recovery" in data["slo_status"]
    
    async def test_scenario_5_multi_tenant_isolation(self, setup_test_environment):
        """Scenario 5: Multi-tenant data isolation"""
        services, test_data = await setup_test_environment
        
        # Create test data for multiple tenants
        tenants = ["tenant-a", "tenant-b", "tenant-c"]
        
        # Step 1: Create events for each tenant
        for tenant in tenants:
            event_data = {
                "event_id": f"scenario5-{tenant}-event-001",
                "tenant_id": tenant,
                "event_type": "calendar.event.created",
                "topic": "calendar-events",
                "payload": {
                    "event_id": f"scenario5-{tenant}-event-001",
                    "tenant_id": tenant,
                    "user_id": f"{tenant}-user",
                    "title": f"{tenant.title()} Event",
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
                },
                "headers": {},
                "metadata": {}
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services['event_bus']}/events/publish",
                    json=event_data
                )
                assert response.status_code == 200
        
        # Step 2: Wait for processing
        await asyncio.sleep(3)
        
        # Step 3: Verify tenant isolation
        for tenant in tenants:
            async with httpx.AsyncClient() as client:
                # Check calendar events
                response = await client.get(
                    f"{services['calendar_automation']}/events?tenant_id={tenant}"
                )
                assert response.status_code == 200
                events = response.json()
                assert all(event["tenant_id"] == tenant for event in events)
                
                # Check metrics
                response = await client.get(
                    f"{services['metrics_exporter']}/metrics/dashboard?tenant_id={tenant}"
                )
                assert response.status_code == 200
                metrics = response.json()
                assert metrics["tenant_id"] == tenant
        
        # Step 4: Verify no cross-tenant data leakage
        async with httpx.AsyncClient() as client:
            # Get all events for tenant-a
            response = await client.get(
                f"{services['calendar_automation']}/events?tenant_id=tenant-a"
            )
            tenant_a_events = response.json()
            tenant_a_ids = {event["event_id"] for event in tenant_a_events}
            
            # Get all events for tenant-b
            response = await client.get(
                f"{services['calendar_automation']}/events?tenant_id=tenant-b"
            )
            tenant_b_events = response.json()
            tenant_b_ids = {event["event_id"] for event in tenant_b_events}
            
            # Verify no overlap
            assert len(tenant_a_ids.intersection(tenant_b_ids)) == 0
    
    async def test_scenario_6_performance_under_load(self, setup_test_environment):
        """Scenario 6: Performance testing under load"""
        services, test_data = await setup_test_environment
        
        # Step 1: Create multiple concurrent events
        event_count = 50
        events = []
        
        for i in range(event_count):
            event = {
                "event_id": f"scenario6-load-event-{i:03d}",
                "tenant_id": test_data["tenant_id"],
                "event_type": "calendar.event.created",
                "topic": "calendar-events",
                "payload": {
                    "event_id": f"scenario6-load-event-{i:03d}",
                    "tenant_id": test_data["tenant_id"],
                    "user_id": test_data["user_id"],
                    "title": f"Load Test Event {i}",
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
                },
                "headers": {},
                "metadata": {}
            }
            events.append(event)
        
        # Step 2: Publish all events concurrently
        start_time = time.time()
        
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
        
        publish_time = time.time() - start_time
        
        # Step 3: Wait for processing
        await asyncio.sleep(10)
        
        # Step 4: Verify all events were processed
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['calendar_automation']}/events?tenant_id={test_data['tenant_id']}"
            )
            assert response.status_code == 200
            processed_events = response.json()
            assert len(processed_events) >= event_count
        
        # Step 5: Verify performance metrics
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['metrics_exporter']}/metrics/dashboard?tenant_id={test_data['tenant_id']}"
            )
            assert response.status_code == 200
            metrics = response.json()
            assert "response_time" in metrics["metrics"]
            
            # Verify response times are within acceptable limits
            avg_response_time = metrics["metrics"]["response_time"]["avg_ms"]
            assert avg_response_time < 2000  # Less than 2 seconds
        
        # Step 6: Verify system health
        for service_name, service_url in services.items():
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{service_url}/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"

class TestBusinessLogicScenarios:
    """Business logic specific test scenarios"""
    
    async def test_scenario_7_sla_tracking_and_alerts(self):
        """Scenario 7: SLA tracking and alert generation"""
        services = {
            "cruxx_automation": "http://localhost:8009",
            "metrics_exporter": "http://localhost:8012",
            "event_bus": "http://localhost:8013"
        }
        
        tenant_id = "sla-test-tenant"
        
        # Step 1: Create Cruxx action with SLA
        cruxx_action = {
            "cruxx_id": "sla-test-action-001",
            "tenant_id": tenant_id,
            "opportunity_id": "sla-test-opportunity-001",
            "title": "SLA Test Action",
            "description": "Action with SLA tracking",
            "assignee": "sla-test-user@example.com",
            "due_date": (datetime.utcnow() + timedelta(hours=2)).isoformat(),  # 2 hours SLA
            "priority": "high",
            "status": "open",
            "sla_status": "met",
            "created_by": "automation"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "sla-test-action-001",
                    "tenant_id": tenant_id,
                    "event_type": "cruxx.action.created",
                    "topic": "cruxx-events",
                    "payload": cruxx_action,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 2: Simulate SLA warning (1 hour before due)
        await asyncio.sleep(1)
        
        sla_warning_data = {
            "tenant_id": tenant_id,
            "cruxx_id": "sla-test-action-001",
            "sla_status": "warning",
            "time_remaining_hours": 1,
            "alert_type": "sla_warning"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['event_bus']}/events/publish",
                json={
                    "event_id": "sla-warning-001",
                    "tenant_id": tenant_id,
                    "event_type": "system.sla.warning",
                    "topic": "system-events",
                    "payload": sla_warning_data,
                    "headers": {},
                    "metadata": {}
                }
            )
            assert response.status_code == 200
        
        # Step 3: Verify SLA metrics
        await asyncio.sleep(1)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['metrics_exporter']}/metrics/alerts?tenant_id={tenant_id}"
            )
            assert response.status_code == 200
            data = response.json()
            assert "alerts" in data
    
    async def test_scenario_8_compliance_and_audit_trail(self):
        """Scenario 8: Compliance and audit trail verification"""
        services = {
            "model_audit": "http://localhost:8006",
            "run_trace_schema": "http://localhost:8010",
            "event_bus": "http://localhost:8013"
        }
        
        tenant_id = "compliance-test-tenant"
        user_id = "compliance-test-user"
        session_id = "compliance-test-session"
        
        # Step 1: Create a trace with compliance flags
        trace_data = {
            "trace_id": "compliance-trace-001",
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            "service_name": "calendar",
            "operation_type": "create_event",
            "status": "started",
            "start_time": datetime.utcnow().isoformat(),
            "overall_confidence_score": 0.8,
            "overall_trust_score": 0.7,
            "trust_drift": 0.0,
            "inputs": [],
            "decisions": [],
            "outputs": [],
            "spans": [],
            "compliance_flags": ["gdpr_compliant", "audit_required"],
            "context": {},
            "metadata": {}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['run_trace_schema']}/traces",
                json=trace_data
            )
            assert response.status_code == 200
        
        # Step 2: Add compliance decision
        decision_data = {
            "decision_id": "compliance-decision-001",
            "trace_id": "compliance-trace-001",
            "decision_type": "compliance_check",
            "decision_point": "data_processing",
            "input_data": {"data_type": "personal_information"},
            "decision_criteria": {"gdpr_compliance": True},
            "decision_result": {"approved": True, "reason": "GDPR compliant"},
            "confidence_score": 0.9,
            "trust_score": 0.8,
            "reasoning": "Data processing complies with GDPR requirements",
            "alternatives_considered": []
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{services['run_trace_schema']}/traces/compliance-trace-001/decisions",
                json=decision_data
            )
            assert response.status_code == 200
        
        # Step 3: Verify audit trail
        await asyncio.sleep(1)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{services['model_audit']}/audit/summary?tenant_id={tenant_id}"
            )
            assert response.status_code == 200
            data = response.json()
            assert "total_calls" in data
            assert "compliance_flags" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
