"""
Task 3.2.27: CAB workflow & minutes auto-logging to evidence
- Traceable approvals
- CAB app/integration
- Time-stamped
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import json
import hashlib

app = FastAPI(title="RBIA CAB Minutes & Evidence Logging Service")
logger = logging.getLogger(__name__)

class MeetingStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VoteType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DEFER = "defer"

class AgendaItemType(str, Enum):
    MODEL_APPROVAL = "model_approval"
    POLICY_REVIEW = "policy_review"
    INCIDENT_REVIEW = "incident_review"
    RISK_ASSESSMENT = "risk_assessment"
    GENERAL_DISCUSSION = "general_discussion"

class CABMember(BaseModel):
    member_id: str
    name: str
    role: str
    department: str
    email: str
    approval_authority: List[str] = []  # Risk levels they can approve
    present: bool = True

class AgendaItem(BaseModel):
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    item_type: AgendaItemType
    title: str
    description: str
    presenter: str
    estimated_duration_minutes: int = 15
    
    # Related artifacts
    model_id: Optional[str] = None
    approval_request_id: Optional[str] = None
    policy_id: Optional[str] = None
    risk_level: Optional[str] = None
    
    # Supporting documents
    documents: List[str] = []  # URLs or file paths
    evidence_pack_id: Optional[str] = None

class Vote(BaseModel):
    member_id: str
    vote: VoteType
    comments: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgendaItemDiscussion(BaseModel):
    item_id: str
    discussion_points: List[str] = []
    concerns_raised: List[str] = []
    questions_asked: List[str] = []
    votes: List[Vote] = []
    decision: Optional[str] = None
    action_items: List[str] = []
    follow_up_required: bool = False
    next_review_date: Optional[datetime] = None

class CABMeeting(BaseModel):
    meeting_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    meeting_date: datetime
    tenant_id: str
    
    # Meeting details
    meeting_title: str
    meeting_type: str = "regular"  # regular, emergency, special
    chair_person: str
    secretary: str
    location: str = "virtual"
    
    # Attendees
    members: List[CABMember] = []
    guests: List[str] = []  # Non-voting attendees
    
    # Agenda
    agenda_items: List[AgendaItem] = []
    
    # Meeting status
    status: MeetingStatus = MeetingStatus.SCHEDULED
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Minutes
    opening_remarks: Optional[str] = None
    closing_remarks: Optional[str] = None
    next_meeting_date: Optional[datetime] = None

class CABMinutes(BaseModel):
    minutes_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    meeting_id: str
    tenant_id: str
    
    # Meeting summary
    meeting_summary: str
    key_decisions: List[str] = []
    action_items: List[str] = []
    
    # Detailed discussions
    item_discussions: List[AgendaItemDiscussion] = []
    
    # Attendance
    attendees_present: List[str] = []
    attendees_absent: List[str] = []
    
    # Administrative
    minutes_prepared_by: str
    minutes_approved_by: Optional[str] = None
    minutes_approved_at: Optional[datetime] = None
    
    # Evidence and audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    evidence_hash: str = ""
    immutable_copy_stored: bool = False
    auto_logged_to_evidence: bool = False

class EvidenceLogEntry(BaseModel):
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str
    event_type: str = "cab_minutes_logged"
    
    # CAB-specific data
    meeting_id: str
    minutes_id: str
    decisions_made: List[str] = []
    approvals_granted: List[str] = []
    approvals_rejected: List[str] = []
    
    # Evidence integrity
    content_hash: str
    previous_hash: Optional[str] = None
    hash_chain_verified: bool = True
    
    # Metadata
    logged_by: str = "cab_minutes_service"
    metadata: Dict[str, Any] = {}

# In-memory stores (replace with proper database in production)
cab_meetings: Dict[str, CABMeeting] = {}
cab_minutes: Dict[str, CABMinutes] = {}
evidence_log: List[EvidenceLogEntry] = []

def _generate_content_hash(content: Dict[str, Any]) -> str:
    """Generate SHA256 hash of content for evidence integrity."""
    content_str = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(content_str.encode()).hexdigest()

def _get_previous_hash(tenant_id: str) -> Optional[str]:
    """Get the hash of the most recent evidence entry for hash chaining."""
    tenant_entries = [e for e in evidence_log if e.tenant_id == tenant_id]
    return tenant_entries[-1].content_hash if tenant_entries else None

async def _auto_log_to_evidence(minutes: CABMinutes, meeting: CABMeeting) -> EvidenceLogEntry:
    """Automatically log CAB minutes to evidence store with hash chaining."""
    
    # Extract key decisions and approvals
    decisions_made = minutes.key_decisions.copy()
    approvals_granted = []
    approvals_rejected = []
    
    for discussion in minutes.item_discussions:
        if discussion.decision:
            decisions_made.append(f"Item {discussion.item_id}: {discussion.decision}")
        
        # Count votes for approvals/rejections
        approve_votes = len([v for v in discussion.votes if v.vote == VoteType.APPROVE])
        reject_votes = len([v for v in discussion.votes if v.vote == VoteType.REJECT])
        
        if approve_votes > reject_votes:
            approvals_granted.append(discussion.item_id)
        elif reject_votes > approve_votes:
            approvals_rejected.append(discussion.item_id)
    
    # Create evidence content
    evidence_content = {
        "meeting_id": minutes.meeting_id,
        "minutes_id": minutes.minutes_id,
        "tenant_id": minutes.tenant_id,
        "meeting_date": meeting.meeting_date.isoformat(),
        "chair_person": meeting.chair_person,
        "attendees": minutes.attendees_present,
        "decisions": decisions_made,
        "approvals_granted": approvals_granted,
        "approvals_rejected": approvals_rejected,
        "action_items": minutes.action_items,
        "created_at": minutes.created_at.isoformat()
    }
    
    content_hash = _generate_content_hash(evidence_content)
    previous_hash = _get_previous_hash(minutes.tenant_id)
    
    evidence_entry = EvidenceLogEntry(
        tenant_id=minutes.tenant_id,
        meeting_id=minutes.meeting_id,
        minutes_id=minutes.minutes_id,
        decisions_made=decisions_made,
        approvals_granted=approvals_granted,
        approvals_rejected=approvals_rejected,
        content_hash=content_hash,
        previous_hash=previous_hash,
        metadata={"evidence_content": evidence_content}
    )
    
    evidence_log.append(evidence_entry)
    
    # Update minutes to indicate evidence logging
    minutes.evidence_hash = content_hash
    minutes.auto_logged_to_evidence = True
    minutes.immutable_copy_stored = True
    
    logger.info(f"CAB minutes {minutes.minutes_id} auto-logged to evidence with hash {content_hash}")
    return evidence_entry

@app.post("/cab/meetings", response_model=CABMeeting)
async def create_cab_meeting(meeting: CABMeeting):
    """Create a new CAB meeting with agenda."""
    
    # Validate required fields
    if not meeting.members:
        raise HTTPException(status_code=400, detail="CAB meeting must have at least one member")
    
    if not meeting.agenda_items:
        raise HTTPException(status_code=400, detail="CAB meeting must have at least one agenda item")
    
    # Store meeting
    cab_meetings[meeting.meeting_id] = meeting
    
    logger.info(f"Created CAB meeting {meeting.meeting_id} for tenant {meeting.tenant_id} on {meeting.meeting_date}")
    return meeting

@app.post("/cab/meetings/{meeting_id}/start")
async def start_cab_meeting(meeting_id: str):
    """Start a CAB meeting (mark as in progress)."""
    
    if meeting_id not in cab_meetings:
        raise HTTPException(status_code=404, detail="CAB meeting not found")
    
    meeting = cab_meetings[meeting_id]
    meeting.status = MeetingStatus.IN_PROGRESS
    meeting.started_at = datetime.utcnow()
    
    logger.info(f"Started CAB meeting {meeting_id}")
    return {"status": "started", "started_at": meeting.started_at}

@app.post("/cab/meetings/{meeting_id}/minutes", response_model=CABMinutes)
async def create_cab_minutes(
    meeting_id: str, 
    minutes: CABMinutes,
    background_tasks: BackgroundTasks,
    auto_log_evidence: bool = True
):
    """Create and store CAB meeting minutes with auto-logging to evidence."""
    
    if meeting_id not in cab_meetings:
        raise HTTPException(status_code=404, detail="CAB meeting not found")
    
    meeting = cab_meetings[meeting_id]
    minutes.meeting_id = meeting_id
    minutes.tenant_id = meeting.tenant_id
    
    # Validate minutes content
    if not minutes.meeting_summary:
        raise HTTPException(status_code=400, detail="Meeting summary is required")
    
    if not minutes.minutes_prepared_by:
        raise HTTPException(status_code=400, detail="Minutes prepared by is required")
    
    # Store minutes
    cab_minutes[minutes.minutes_id] = minutes
    
    # Mark meeting as completed
    meeting.status = MeetingStatus.COMPLETED
    meeting.ended_at = datetime.utcnow()
    
    # Auto-log to evidence if requested
    if auto_log_evidence:
        background_tasks.add_task(_auto_log_to_evidence, minutes, meeting)
    
    logger.info(f"Created CAB minutes {minutes.minutes_id} for meeting {meeting_id}")
    return minutes

@app.post("/cab/meetings/{meeting_id}/vote")
async def record_vote(
    meeting_id: str,
    item_id: str,
    member_id: str,
    vote: VoteType,
    comments: Optional[str] = None
):
    """Record a vote from a CAB member on an agenda item."""
    
    if meeting_id not in cab_meetings:
        raise HTTPException(status_code=404, detail="CAB meeting not found")
    
    meeting = cab_meetings[meeting_id]
    
    # Validate member is part of the meeting
    member_ids = [m.member_id for m in meeting.members]
    if member_id not in member_ids:
        raise HTTPException(status_code=403, detail="Member not authorized for this meeting")
    
    # Validate agenda item exists
    agenda_item_ids = [item.item_id for item in meeting.agenda_items]
    if item_id not in agenda_item_ids:
        raise HTTPException(status_code=404, detail="Agenda item not found")
    
    vote_record = Vote(
        member_id=member_id,
        vote=vote,
        comments=comments
    )
    
    # Store vote (in production, this would be in the minutes when they're created)
    logger.info(f"Recorded vote: {member_id} voted {vote.value} on item {item_id} in meeting {meeting_id}")
    
    return {
        "status": "vote_recorded",
        "meeting_id": meeting_id,
        "item_id": item_id,
        "member_id": member_id,
        "vote": vote.value,
        "timestamp": vote_record.timestamp
    }

@app.get("/cab/meetings/{meeting_id}/minutes", response_model=CABMinutes)
async def get_cab_minutes(meeting_id: str):
    """Retrieve CAB meeting minutes."""
    
    minutes = next((m for m in cab_minutes.values() if m.meeting_id == meeting_id), None)
    if not minutes:
        raise HTTPException(status_code=404, detail="CAB minutes not found for this meeting")
    
    return minutes

@app.get("/cab/evidence-log", response_model=List[EvidenceLogEntry])
async def get_evidence_log(
    tenant_id: Optional[str] = None,
    meeting_id: Optional[str] = None,
    limit: int = 100
):
    """Retrieve evidence log entries for CAB minutes."""
    
    filtered_entries = evidence_log
    
    if tenant_id:
        filtered_entries = [e for e in filtered_entries if e.tenant_id == tenant_id]
    
    if meeting_id:
        filtered_entries = [e for e in filtered_entries if e.meeting_id == meeting_id]
    
    return filtered_entries[-limit:]

@app.post("/cab/minutes/{minutes_id}/approve")
async def approve_minutes(
    minutes_id: str,
    approved_by: str,
    background_tasks: BackgroundTasks
):
    """Approve CAB minutes (typically by meeting chair)."""
    
    if minutes_id not in cab_minutes:
        raise HTTPException(status_code=404, detail="CAB minutes not found")
    
    minutes = cab_minutes[minutes_id]
    minutes.minutes_approved_by = approved_by
    minutes.minutes_approved_at = datetime.utcnow()
    
    # Re-log to evidence with approval
    if minutes_id in cab_meetings:
        meeting = cab_meetings[minutes.meeting_id]
        background_tasks.add_task(_auto_log_to_evidence, minutes, meeting)
    
    logger.info(f"CAB minutes {minutes_id} approved by {approved_by}")
    return {
        "status": "approved",
        "approved_by": approved_by,
        "approved_at": minutes.minutes_approved_at
    }

@app.get("/cab/meetings", response_model=List[CABMeeting])
async def list_cab_meetings(
    tenant_id: Optional[str] = None,
    status: Optional[MeetingStatus] = None,
    limit: int = 50
):
    """List CAB meetings with optional filters."""
    
    meetings = list(cab_meetings.values())
    
    if tenant_id:
        meetings = [m for m in meetings if m.tenant_id == tenant_id]
    
    if status:
        meetings = [m for m in meetings if m.status == status]
    
    # Sort by meeting date (most recent first)
    meetings.sort(key=lambda x: x.meeting_date, reverse=True)
    
    return meetings[:limit]

@app.get("/cab/evidence-integrity/{tenant_id}")
async def verify_evidence_integrity(tenant_id: str):
    """Verify the hash chain integrity of CAB evidence for a tenant."""
    
    tenant_entries = [e for e in evidence_log if e.tenant_id == tenant_id]
    tenant_entries.sort(key=lambda x: x.timestamp)
    
    integrity_report = {
        "tenant_id": tenant_id,
        "total_entries": len(tenant_entries),
        "chain_intact": True,
        "broken_links": [],
        "verification_timestamp": datetime.utcnow()
    }
    
    # Verify hash chain
    for i, entry in enumerate(tenant_entries):
        if i == 0:
            # First entry should have no previous hash
            if entry.previous_hash is not None:
                integrity_report["chain_intact"] = False
                integrity_report["broken_links"].append(f"Entry {entry.log_id}: unexpected previous hash")
        else:
            # Subsequent entries should reference previous entry's hash
            expected_previous = tenant_entries[i-1].content_hash
            if entry.previous_hash != expected_previous:
                integrity_report["chain_intact"] = False
                integrity_report["broken_links"].append(f"Entry {entry.log_id}: hash chain broken")
    
    return integrity_report

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA CAB Minutes & Evidence Logging Service", "task": "3.2.27"}
