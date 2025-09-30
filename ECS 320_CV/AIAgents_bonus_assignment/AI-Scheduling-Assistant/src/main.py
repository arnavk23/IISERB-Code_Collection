from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import uvicorn
from datetime import datetime, timedelta
import copy

# Import the intelligent meeting scheduler
from propose_time import intelligent_meeting_scheduler

app = FastAPI(title="Intelligent Meeting Scheduler API", version="1.0.0")


# Pydantic models for request/response validation
class Attendee(BaseModel):
    email: str


class MeetingRequest(BaseModel):
    Request_id: str
    Datetime: str
    Location: str
    From: str
    Attendees: List[Attendee]
    Subject: str
    EmailContent: str


class Event(BaseModel):
    StartTime: str
    EndTime: str
    NumAttendees: int
    Attendees: List[str]
    Summary: str


class AttendeeEvents(BaseModel):
    email: str
    events: List[Event]


class MeetingResponse(BaseModel):
    Request_id: str
    Datetime: str
    Location: str
    From: str
    Attendees: List[AttendeeEvents]
    Subject: str
    EmailContent: str
    EventStart: str
    EventEnd: str
    Duration_mins: str
    MetaData: Dict[str, Any]


def filter_off_hours_events(events: List[Dict]) -> List[Dict]:
    """Filter out events with 'Off Hours' in the summary"""
    return events


def apply_rescheduling_to_attendee_events(
    attendee_events: Dict[str, List[Dict]],
    decision: Dict[str, Any],
    input_request: dict,
) -> Dict[str, List[Dict]]:
    """Apply rescheduling decisions to attendee events"""
    # Start with a deep copy of original events
    updated_events = copy.deepcopy(attendee_events)

    if not decision:
        return updated_events

    # Get the rescheduling details
    proposed_start = decision.get("proposed_final_start")
    proposed_end = decision.get("proposed_final_end")
    rescheduled_start = decision.get("conflicting_final_start")
    rescheduled_end = decision.get("conflicting_final_end")

    print(f"🔧 Processing rescheduling:")
    print(f"   Proposed meeting time: {proposed_start} to {proposed_end}")
    print(f"   Rescheduled conflict time: {rescheduled_start} to {rescheduled_end}")

    # Only proceed if we have a valid proposed meeting time
    if not (proposed_start and proposed_end):
        print("⚠️ No valid proposed meeting time found")
        return updated_events

    # Helper function to check if two time slots overlap
    def times_overlap(start1, end1, start2, end2):
        """Check if two time ranges overlap"""
        from datetime import datetime

        try:
            s1 = datetime.fromisoformat(start1.replace("+05:30", ""))
            e1 = datetime.fromisoformat(end1.replace("+05:30", ""))
            s2 = datetime.fromisoformat(start2.replace("+05:30", ""))
            e2 = datetime.fromisoformat(end2.replace("+05:30", ""))

            # Two ranges overlap if start1 < end2 and start2 < end1
            return s1 < e2 and s2 < e1
        except:
            return False

    # Step 1: Handle rescheduling of conflicting events (if any)
    if rescheduled_start and rescheduled_end:
        print("🔄 Rescheduling conflicting events...")

        # Find and reschedule conflicting events
        for email, events in updated_events.items():
            events_to_remove = []
            events_to_add = []

            for i, event in enumerate(events):
                # Check if this event overlaps with the proposed time
                if times_overlap(
                    event["StartTime"], event["EndTime"], proposed_start, proposed_end
                ):
                    # Skip "Off Hours" and "SELF" events - they can't be rescheduled
                    if event.get("Summary", "") == "Off Hours":
                        print(
                            f"   ⚠️ Cannot reschedule {event['Summary']} for {email} - protected event"
                        )
                        continue

                    print(
                        f"   📅 Found overlapping event for {email}: {event['Summary']} ({event['StartTime']} to {event['EndTime']})"
                    )

                    # Mark for removal
                    events_to_remove.append(i)

                    # Create rescheduled version
                    rescheduled_event = {
                        "StartTime": rescheduled_start,
                        "EndTime": rescheduled_end,
                        "NumAttendees": event["NumAttendees"],
                        "Attendees": event["Attendees"].copy(),
                        "Summary": event["Summary"],
                    }
                    events_to_add.append(rescheduled_event)

            # Remove conflicting events (in reverse order to maintain indices)
            for i in reversed(events_to_remove):
                print(f"   ❌ Removing overlapping event from {email}")
                events.pop(i)

            # Add rescheduled events
            for event in events_to_add:
                print(
                    f"   ➕ Adding rescheduled event for {email} at {event['StartTime']}"
                )
                events.append(event)

    # Step 2: Add the new meeting to ALL attendees
    print("📝 Adding new meeting to all attendees...")

    # Get all attendee emails from the request
    all_attendee_emails = [att["email"] for att in input_request["Attendees"]]
    if input_request["From"] not in all_attendee_emails:
        all_attendee_emails.insert(0, input_request["From"])

    meeting_subject = input_request.get("Subject", "Meeting")

    # Create the new meeting event
    new_meeting_event = {
        "StartTime": proposed_start,
        "EndTime": proposed_end,
        "NumAttendees": len(all_attendee_emails),
        "Attendees": all_attendee_emails.copy(),
        "Summary": meeting_subject,
    }

    # Add to each attendee's calendar
    for email in all_attendee_emails:
        # Ensure attendee exists in updated_events
        if email not in updated_events:
            updated_events[email] = []

        # Check if this exact meeting already exists (prevent duplicates)
        meeting_exists = any(
            (
                event["StartTime"] == proposed_start
                and event["EndTime"] == proposed_end
                and event["Summary"] == meeting_subject
            )
            for event in updated_events[email]
        )

        if not meeting_exists:
            print(f"   ✅ Adding new meeting '{meeting_subject}' to {email}")
            updated_events[email].append(new_meeting_event.copy())
        else:
            print(f"   ⏭️ Meeting already exists for {email}, skipping")

    return updated_events


def process_scheduler_results(input_request: dict, results: dict) -> dict:
    """Process the scheduler results and format according to output specification"""
    print("🔄 Processing scheduler results...")

    # Get the decision information
    decision = results.get("decision", {})
    attendee_events = results.get("attendee_events", {})

    print(f"📊 Original attendee events count:")
    for email, events in attendee_events.items():
        print(f"   {email}: {len(events)} events")

    # Apply rescheduling to attendee events
    updated_attendee_events = apply_rescheduling_to_attendee_events(
        attendee_events, decision, input_request
    )

    print(f"📊 After rescheduling:")
    for email, events in updated_attendee_events.items():
        print(f"   {email}: {len(events)} events")

    # Filter out off-hours events for all attendees
    print("🔍 Filtering out 'Off Hours' events...")
    for email in updated_attendee_events:
        original_count = len(updated_attendee_events[email])
        updated_attendee_events[email] = filter_off_hours_events(
            updated_attendee_events[email]
        )
        filtered_count = len(updated_attendee_events[email])
        if original_count != filtered_count:
            print(
                f"   {email}: Filtered {original_count - filtered_count} off-hours events"
            )

    # Determine final event start and end times
    event_start = decision.get("proposed_final_start", "")
    event_end = decision.get("proposed_final_end", "")

    print(f"🎯 Final scheduled meeting: {event_start} to {event_end}")

    # Calculate duration
    duration_mins = (
        calculate_duration_minutes(event_start, event_end)
        if event_start and event_end
        else "30"
    )

    # Build the attendees list with their events
    attendees_list = []

    # Get all attendee emails from the original request
    all_attendee_emails = [att["email"] for att in input_request["Attendees"]]
    if input_request["From"] not in all_attendee_emails:
        all_attendee_emails.insert(0, input_request["From"])

    print("📋 Building final attendee list:")

    # Include ALL attendees, even if they have no events
    for email in all_attendee_emails:
        attendee_events_list = updated_attendee_events.get(email, [])

        # Sort events by start time
        attendee_events_list.sort(key=lambda x: x["StartTime"])

        print(f"   {email}: {len(attendee_events_list)} final events")

        attendees_list.append({"email": email, "events": attendee_events_list})

    # Build the final response
    response = {
        "Request_id": input_request["Request_id"],
        "Datetime": input_request["Datetime"],
        "Location": input_request["Location"],
        "From": input_request["From"],
        "Attendees": attendees_list,
        "Subject": input_request["Subject"],
        "EmailContent": input_request["EmailContent"],
        "EventStart": event_start,
        "EventEnd": event_end,
        "Duration_mins": duration_mins,
        "MetaData": {},
    }

    print("✅ Scheduler processing complete!")
    return response


def calculate_duration_minutes(start_time: str, end_time: str) -> str:
    """Calculate duration in minutes between start and end time"""
    try:
        start_dt = datetime.fromisoformat(start_time.replace("+05:30", ""))
        end_dt = datetime.fromisoformat(end_time.replace("+05:30", ""))
        duration = (end_dt - start_dt).total_seconds() / 60
        return str(int(duration))
    except:
        return "30"  # Default to 30 minutes


# def process_scheduler_results(input_request: dict, results: dict) -> dict:
#     """Process the scheduler results and format according to output specification"""

#     # Get the decision information
#     decision = results.get("decision", {})
#     attendee_events = results.get("attendee_events", {})

#     # Apply rescheduling to attendee events
#     updated_attendee_events = apply_rescheduling_to_attendee_events(
#         attendee_events, decision
#     )

#     # Filter out off-hours events for all attendees
#     for email in updated_attendee_events:
#         updated_attendee_events[email] = filter_off_hours_events(
#             updated_attendee_events[email]
#         )

#     # Determine final event start and end times
#     event_start = decision.get("proposed_final_start", "")
#     event_end = decision.get("proposed_final_end", "")

#     # Calculate duration
#     duration_mins = (
#         calculate_duration_minutes(event_start, event_end)
#         if event_start and event_end
#         else "30"
#     )

#     # Build the attendees list with their events
#     attendees_list = []

#     # Add sender to attendees if not already present
#     all_attendee_emails = [att["email"] for att in input_request["Attendees"]]
#     if input_request["From"] not in all_attendee_emails:
#         all_attendee_emails.insert(0, input_request["From"])

#     for email in all_attendee_emails:
#         attendee_events_list = updated_attendee_events.get(email, [])

#         # Sort events by start time
#         attendee_events_list.sort(key=lambda x: x["StartTime"])

#         attendees_list.append({"email": email, "events": attendee_events_list})

#     # Build the final response
#     response = {
#         "Request_id": input_request["Request_id"],
#         "Datetime": input_request["Datetime"],
#         "Location": input_request["Location"],
#         "From": input_request["From"],
#         "Attendees": attendees_list,
#         "Subject": input_request["Subject"],
#         "EmailContent": input_request["EmailContent"],
#         "EventStart": event_start,
#         "EventEnd": event_end,
#         "Duration_mins": duration_mins,
#         "MetaData": {},
#     }

#     return response


@app.post("/receive", response_model=MeetingResponse)
async def schedule_meeting(request: MeetingRequest):
    """
    Process a meeting scheduling request and return the scheduled meeting details
    with attendee availability and any necessary rescheduling.
    """
    max_retries = 2
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # Convert Pydantic model to dict for the scheduler function
            input_dict = {
                "Request_id": request.Request_id,
                "Datetime": request.Datetime,
                "Location": request.Location,
                "From": request.From,
                "Attendees": [{"email": att.email} for att in request.Attendees],
                "Subject": request.Subject,
                "EmailContent": request.EmailContent,
            }

            # Call the intelligent meeting scheduler
            results = intelligent_meeting_scheduler(input_dict)

            # Process the results into the required output format
            response_data = process_scheduler_results(input_dict, results)

            return MeetingResponse(**response_data)

        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to schedule meeting after {max_retries} retries. Error: {str(e)}",
                )
            # Log the retry attempt (in production, use proper logging)
            print(f"Scheduling attempt {retry_count} failed: {str(e)}. Retrying...")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "intelligent-meeting-scheduler"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Intelligent Meeting Scheduler API",
        "version": "1.0.0",
        "endpoints": {
            "POST /schedule-meeting": "Schedule a meeting with intelligent conflict resolution",
            "GET /health": "Health check endpoint",
            "GET /docs": "API documentation",
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Assuming this file is named main.py
        host="localhost",
        port=5000,
        reload=True,
        log_level="info",
    )
