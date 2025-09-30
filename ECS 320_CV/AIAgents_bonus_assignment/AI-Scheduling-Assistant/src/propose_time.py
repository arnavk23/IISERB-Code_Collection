from datetime import datetime, timedelta
import json
from get_attendee_events import (
    get_all_attendee_events_2_days_parallel,
)
from request_to_time import extract_time_window


def parse_time(time_str):
    """Convert time string to datetime object"""
    return datetime.fromisoformat(time_str.replace("+05:30", ""))


def format_time(dt):
    """Convert datetime object to time string"""
    return dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")


def has_conflict(proposed_start, proposed_end, attendee_events):
    """
    Check if proposed time conflicts with any attendee's existing events
    Returns: (has_conflict: bool, conflicting_meetings: list)
    """
    proposed_start_dt = parse_time(proposed_start)
    proposed_end_dt = parse_time(proposed_end)

    conflicting_meetings = []

    for email, events in attendee_events.items():
        for event in events:
            # Skip "Off Hours" events
            if event["Summary"] == "Off Hours":
                continue

            event_start = parse_time(event["StartTime"])
            event_end = parse_time(event["EndTime"])

            # Check for overlap
            if proposed_start_dt < event_end and proposed_end_dt > event_start:
                conflicting_meetings.append(
                    {
                        "attendee": email,
                        "meeting": event,
                        "start": event["StartTime"],
                        "end": event["EndTime"],
                        "summary": event["Summary"],
                    }
                )

    return len(conflicting_meetings) > 0, conflicting_meetings


def find_first_free_slot_in_window(
    start_window_str, end_window_str, duration_mins, attendee_events
):
    """
    Finds the first available time slot of a given duration within a specified window
    by checking the gaps between existing events.
    """
    start_window = parse_time(start_window_str)
    end_window = parse_time(end_window_str)
    duration = timedelta(minutes=duration_mins)

    # 1. Merge all non-"Off Hours" events from all attendees into a single list of busy times
    busy_times = []
    for email, events in attendee_events.items():
        for event in events:
            if event["Summary"] == "Off Hours":
                continue
            busy_times.append(
                {
                    "start": parse_time(event["StartTime"]),
                    "end": parse_time(event["EndTime"]),
                }
            )

    # If there are no busy intervals, the entire window is available
    if not busy_times:
        if start_window + duration <= end_window:
            return format_time(start_window), format_time(start_window + duration)
        else:
            return None, None

    # 2. Sort and merge overlapping busy intervals to get a clean timeline
    busy_times.sort(key=lambda x: x["start"])
    merged_busy_times = [busy_times[0]]
    for current in busy_times[1:]:
        last = merged_busy_times[-1]
        if current["start"] < last["end"]:
            last["end"] = max(last["end"], current["end"])
        else:
            merged_busy_times.append(current)

    # 3. Check for a free slot in the gaps between busy intervals
    # Check the gap from the window start to the first busy interval
    last_busy_end = start_window
    for busy_slot in merged_busy_times:
        free_start = max(last_busy_end, start_window)
        free_end = min(busy_slot["start"], end_window)

        if free_end - free_start >= duration:
            return format_time(free_start), format_time(free_start + duration)

        last_busy_end = busy_slot["end"]

    # Check the gap after the last busy interval to the window end
    if end_window - max(last_busy_end, start_window) >= duration:
        free_start = max(last_busy_end, start_window)
        return format_time(free_start), format_time(free_start + duration)

    # 4. If no suitable gap is found
    return None, None


def find_free_slots(duration_minutes, attendee_events):
    """
    Find all free slots of given duration where all attendees are available
    Returns: list of (start_time, end_time) tuples
    """
    # Get all non-off-hours events and sort by start time
    all_events = []
    for email, events in attendee_events.items():
        for event in events:
            if event["Summary"] != "Off Hours":
                all_events.append(
                    {
                        "start": parse_time(event["StartTime"]),
                        "end": parse_time(event["EndTime"]),
                        "attendee": email,
                    }
                )

    all_events.sort(key=lambda x: x["start"])

    # Find gaps between meetings
    free_slots = []
    duration = timedelta(minutes=duration_minutes)

    # Start from 9 AM of earliest event
    current_time = all_events[0]["start"].replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    end_time = current_time + timedelta(days=2)

    i = 0
    while current_time < end_time:
        # Check if current_time conflicts with any event
        conflicts = False
        next_event_start = end_time

        for event in all_events:
            if current_time < event["end"] and current_time + duration > event["start"]:
                conflicts = True
                if event["end"] > current_time:
                    next_event_start = min(next_event_start, event["end"])

        if not conflicts:
            # Check if we have enough time for the meeting
            next_conflict = end_time
            for event in all_events:
                if event["start"] >= current_time:
                    next_conflict = min(next_conflict, event["start"])

            if next_conflict >= current_time + duration:
                free_slots.append(
                    (format_time(current_time), format_time(current_time + duration))
                )

        # Move to next time slot
        if conflicts:
            current_time = next_event_start
        else:
            current_time += timedelta(minutes=30)  # Check every 30 minutes

    print("free slots: ", free_slots[:5])

    return free_slots[:5]  # Return first 5 free slots


def schedule_with_llm(input_request, proposed_time, conflicting_meetings, free_slots):
    """
    Use LLM to decide final scheduling
    """
    import openai

    BASE_URL = "http://localhost:4000/v1"
    MODEL_PATH = "Models/meta-llama/Llama-3.3-70B-Instruct"

    client = openai.OpenAI(
        base_url=BASE_URL,
        api_key="NULL",
    )

    prompt = f"""
You are a meeting scheduler. Analyze the situation and decide the final meeting times.

PROPOSED MEETING:
- Subject: {input_request.get("Subject")}
- Requested Time: {proposed_time["start_time"]} to {proposed_time["end_time"]}
- Duration: {proposed_time["duration"]} minutes

CONFLICTING MEETINGS:
{json.dumps(conflicting_meetings, indent=2)}

AVAILABLE FREE SLOTS:
{json.dumps(free_slots, indent=2)}

RULES:
1. Evaluate meeting importance based on subject/content
2. If proposed meeting is more important, move conflicting meeting to free slot
3. If conflicting meeting is more important, move proposed meeting to free slot
4. Priority: urgent/client > project updates > regular meetings
5. In case of equal importance, don't reschedule conflicting meeting

Return JSON with:
- "proposed_final_start": final start time for proposed meeting
- "proposed_final_end": final end time for proposed meeting  
- "conflicting_final_start": final start time for conflicting meeting 
- "conflicting_final_end": final end time for conflicting meeting 
- "decision_reason": brief explanation

Example:
{{
    "proposed_final_start": "2025-07-21T07:30:00+05:30",
    "proposed_final_end": "2025-07-21T08:00:00+05:30",
    "conflicting_final_start": "2025-07-21T09:00:00+05:30", 
    "conflicting_final_end": "2025-07-21T10:00:00+05:30",
    "decision_reason": "Moved conflicting meeting as proposed is urgent client matter"
}}
"""

    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {
                "role": "system",
                "content": "You are a meeting scheduler. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.1,
    )

    result = response.choices[0].message.content.strip("```").strip("json")
    return json.loads(result)


def intelligent_meeting_scheduler(input_request):
    """
    Main scheduler function implementing the 4-step algorithm with optimization.
    """
    # Step 1: Get proposed time window
    proposed_time = extract_time_window(input_request)
    print(f"Step 1 - Proposed time: {proposed_time}")

    # Step 2: Fetch calendars and check for conflicts
    attendee_events = get_all_attendee_events_2_days_parallel(
        proposed_time, input_request
    )
    has_conflicts, conflicting_meetings = has_conflict(
        proposed_time["start_time"], proposed_time["end_time"], attendee_events
    )

    # Case 1: No events in the window at all. Schedule immediately.
    if not has_conflicts:
        print("Step 2 - No conflicts found, scheduling directly.")
        return {
            "decision": {
                "proposed_final_start": proposed_time["start_time"],
                "proposed_final_end": proposed_time["end_time"],
                "conflicting_final_start": None,
                "conflicting_final_end": None,
                "decision_reason": "Scheduled at the start of the requested window as it was free.",
            },
            "attendee_events": attendee_events,
        }

    print(
        f"Step 2 - Found {len(conflicting_meetings)} potential conflicts. Checking for free slots to avoid LLM call."
    )

    # OPTIMIZATION: Before calling the LLM, check if a free slot exists in the requested window.
    first_available_start, first_available_end = find_first_free_slot_in_window(
        proposed_time["start_time"],
        proposed_time["end_time"],
        proposed_time["duration"],
        attendee_events,
    )

    # Case 2: Optimization successful. A free slot was found. Schedule and skip LLM.
    if first_available_start:
        print(
            f"Step 2.5 - Optimization success! Found free slot: {first_available_start}. Skipping LLM."
        )
        return {
            "decision": {
                "proposed_final_start": first_available_start,
                "proposed_final_end": first_available_end,
                "conflicting_final_start": None,
                "conflicting_final_end": None,
                "decision_reason": "Scheduled in the first available slot within the requested window, avoiding a conflict.",
            },
            "attendee_events": attendee_events,
        }

    # Case 3: True conflict. No free slots in the window. Proceed with LLM.
    print(
        "Step 2.5 - Optimization failed. No direct free slot found. Proceeding to LLM for rescheduling."
    )

    # Step 3: Find free slots in a wider range for the LLM to use
    free_slots = find_free_slots(proposed_time["duration"], attendee_events)
    print(f"Step 3 - Found {len(free_slots)} free slots for LLM to consider.")

    if not free_slots:
        print("No free slots available for rescheduling! Returning original proposal.")
        return {
            "decision": {
                "proposed_final_start": proposed_time["start_time"],
                "proposed_final_end": proposed_time["end_time"],
                "conflicting_final_start": None,
                "conflicting_final_end": None,
                "decision_reason": "Conflict exists but no alternative slots were found.",
            },
            "attendee_events": attendee_events,
        }

    # Step 4: Use LLM to decide on the best resolution
    print("Step 4 - Calling LLM for intelligent scheduling.")
    llm_decision = schedule_with_llm(
        input_request, proposed_time, conflicting_meetings, free_slots
    )
    print(f"Step 4 - LLM decision: {llm_decision.get('decision_reason', 'N/A')}")

    return {
        "conflicts": conflicting_meetings,
        "decision": llm_decision,
        "attendee_events": attendee_events,
    }


# Usage
if __name__ == "__main__":
    # Test with your existing input_request
    input_request = json.loads("""
    {
        "Request_id": "6118b54f-907b-4451-8d48-dd13d76033b5",
        "Datetime": "19-07-2025T12:34:55",
        "Location": "IISc Bangalore", 
        "From": "userone.amd@gmail.com",
        "Attendees": [
            {"email": "usertwo.amd@gmail.com"},
            {"email": "userthree.amd@gmail.com"}
        ],
        "Subject": "Team - Client Meeting",
        "EmailContent": "Hi Team. The client wants to meet us at Monday at 10 AM to resolve issues."
    }
    """)

    result = intelligent_meeting_scheduler(input_request)
    print(f"\nFinal Result: {json.dumps(result, indent=2)}")
