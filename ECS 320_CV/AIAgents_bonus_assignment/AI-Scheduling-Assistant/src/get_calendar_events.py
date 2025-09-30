import json
from datetime import datetime, timezone, timedelta
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def retrive_calendar_events(user, start, end):
    events_list = []
    token_path = "Keys/" + user.split("@")[0] + ".token"
    user_creds = Credentials.from_authorized_user_file(token_path)
    calendar_service = build("calendar", "v3", credentials=user_creds)
    events_result = (
        calendar_service.events()
        .list(
            calendarId="primary",
            timeMin=start,
            timeMax=end,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items")

    for event in events:
        attendee_list = []
        try:
            for attendee in event["attendees"]:
                attendee_list.append(attendee["email"])
        except:
            attendee_list.append("SELF")
        start_time = event["start"].get("dateTime", event["start"].get("date"))
        end_time = event["end"].get("dateTime", event["end"].get("date"))
        events_list.append(
            {
                "StartTime": start_time,
                "EndTime": end_time,
                "NumAttendees": len(set(attendee_list)),
                "Attendees": list(set(attendee_list)),
                "Summary": event["summary"],
            }
        )
    return events_list
