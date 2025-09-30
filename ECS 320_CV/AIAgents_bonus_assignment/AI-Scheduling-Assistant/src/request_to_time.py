import openai
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from time_profiler import timeit

# Your existing client configuration
BASE_URL = "http://localhost:4000/v1"
MODEL_PATH = "Models/meta-llama/Llama-3.3-70B-Instruct"

client = openai.OpenAI(
    base_url=BASE_URL,
    api_key="NULL",  # vLLM doesn't require an API key
)


@timeit
def extract_time_window(meeting_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract time window from meeting request using LLM.

    Args:
        meeting_request: Dictionary containing meeting details with EmailContent

    Returns:
        Dictionary with duration (float in hours), start_time and end_time (ISO timestamps)
    """

    # Get current datetime for reference
    current_time = datetime.now()
    current_iso = current_time.isoformat()

    # Extract relevant information for the LLM
    email_content = meeting_request.get("EmailContent", "")
    subject = meeting_request.get("Subject", "")
    request_datetime = meeting_request.get("Datetime", current_iso)

    # Create the prompt with examples and instructions
    prompt = f"""You are a meeting scheduler AI. Extract time windows from natural language meeting requests.

Request made on: {request_datetime}

EXAMPLES:
Input: "Let's meet on Thursday at 10 am for 20 minutes"
Output: {{"duration": 20, "start_time": "2025-07-24T10:00:00+05:30", "end_time": "2025-07-24T10:20:00+05:30"}}

Input: "Can we meet Friday for 1 hour"
Output: {{"duration": 60, "start_time": "2025-07-25T00:00:00+05:30", "end_time": "2025-07-25T23:59:00+05:30"}}

Input: "Let's schedule 30 minutes sometime next week"
Output: {{"duration": 30, "start_time": "2025-07-21T00:00:00+05:30", "end_time": "2025-07-25T23:30:00+05:30"}}

Input: "Meeting tomorrow at 2 PM for 45 minutes"
Output: {{"duration": 45, "start_time": "2025-07-21T14:00:00+05:30", "end_time": "2025-07-21T14:45:00+05:30"}}

RULES:
1. Duration is in hours (decimal format)
2. If specific time given, start_time = exact time, end_time = start_time
3. If only day given, start_time = day 00:00, end_time = day 23:59
4. If vague time like "next week", give the full range where meeting can fit
5. Calculate all times relative to request date: {request_datetime}
6. Use ISO 8601 format for timestamps
7. If duration is not specified, assume 30 minutes
8. Return ONLY valid JSON, no explanations

MEETING REQUEST:
Subject: {subject}
Content: {email_content}

Extract the time window:"""

    try:
        # Call the LLM
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise meeting scheduler. Return only valid JSON with duration, start_time, and end_time fields.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        # Extract and parse the response
        llm_output = response.choices[0].message.content.strip()

        # Try to extract JSON from the response
        try:
            # Handle cases where LLM might add extra text
            json_start = llm_output.find("{")
            json_end = llm_output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_output[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = json.loads(llm_output)
        except json.JSONDecodeError:
            # Fallback: try to parse the entire response
            result = json.loads(llm_output)

        # Validate the required fields
        required_fields = ["duration", "start_time", "end_time"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        return result

    except Exception as e:
        # Return a default response if LLM fails
        print(f"Error extracting time window: {e}")


def preprocess_meeting_request(raw_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the input to extract only relevant parts for time extraction.

    Args:
        raw_request: Full meeting request dictionary

    Returns:
        Simplified dictionary with relevant fields
    """
    return {
        "Subject": raw_request.get("Subject", ""),
        "EmailContent": raw_request.get("EmailContent", ""),
        "Datetime": raw_request.get("Datetime", datetime.now().isoformat()),
        "Location": raw_request.get("Location", ""),
    }


# Example usage
if __name__ == "__main__":
    # Test with the provided example
    test_cases = [
        {
            "name": "Test Case 1 - Thursday 30 minutes (Broad Day)",
            "data": {
                "Request_id": "6118b54f-907b-4451-8d48-dd13d76033a5",
                "Datetime": "19-07-2025T12:34:55",
                "Location": "IISc Bangalore",
                "From": "userone.amd@gmail.com",
                "Attendees": [
                    {"email": "usertwo.amd@gmail.com"},
                    {"email": "userthree.amd@gmail.com"},
                ],
                "Subject": "Agentic AI Project Status Update",
                "EmailContent": "Hi team, let's meet on Thursday for 30 minutes to discuss the status of Agentic AI Project.",
            },
        },
        {
            "name": "Test Case 2 - Monday 9:00 AM (Specific Time)",
            "data": {
                "Request_id": "6118b54f-907b-4451-8d48-dd13d76033b5",
                "Datetime": "19-07-2025T12:34:55",
                "Location": "IISc Bangalore",
                "From": "userone.amd@gmail.com",
                "Attendees": [
                    {"email": "usertwo.amd@gmail.com"},
                    {"email": "userthree.amd@gmail.com"},
                ],
                "Subject": "Client Validation - Urgent",
                "EmailContent": "Hi Team. We've just received quick feedback from the client indicating that the instructions we provided aren't working on their end. Let's prioritize resolving this promptly. Let's meet Monday at 9:00 AM to discuss and resolve this issue.",
            },
        },
        {
            "name": "Test Case 3 - Tuesday 11:00 AM (Specific Time)",
            "data": {
                "Request_id": "6118b54f-907b-4451-8d48-dd13d76033c5",
                "Datetime": "19-07-2025T12:34:55",
                "Location": "IISc Bangalore",
                "From": "userone.amd@gmail.com",
                "Attendees": [
                    {"email": "usertwo.amd@gmail.com"},
                    {"email": "userthree.amd@gmail.com"},
                ],
                "Subject": "Project Status",
                "EmailContent": "Hi Team. Let's meet on Tuesday at 11:00 A.M and discuss about our on-going Projects.",
            },
        },
        {
            "name": "Test Case 4 - Next Week (Broad Week)",
            "data": {
                "Request_id": "6118b54f-907b-4451-8d48-dd13d76033d5",
                "Datetime": "19-07-2025T12:34:55",
                "Location": "IISc Bangalore",
                "From": "userone.amd@gmail.com",
                "Attendees": [
                    {"email": "usertwo.amd@gmail.com"},
                    {"email": "userthree.amd@gmail.com"},
                ],
                "Subject": "Client Feedback",
                "EmailContent": "Hi Team. let's meet next week to discuss",
            },
        },
        {
            "name": "Test Case 5 ",
            "data": {
                "Request_id": "6118b54f-907b-4451-8d48-dd13d76033d6",
                "Datetime": "19-07-2025T12:34:55",
                "Location": "IISc Bangalore",
                "From": "userone.amd@gmail.com",
                "Attendees": [
                    {"email": "usertwo.amd@gmail.com"},
                    {"email": "userthree.amd@gmail.com"},
                ],
                "Subject": "Weekly Review",
                "EmailContent": "Hi team, let's meet on Thursday at 11:00 AM to discuss the status of Agentic AI Project.",
            },
        },
    ]

    for sample_request in test_cases:
        print(f"Running {sample_request['data']['EmailContent']}...")

        # Preprocess the request
        processed_request = preprocess_meeting_request(sample_request["data"])

        # Extract time window using LLM
        time_window = extract_time_window(processed_request)

        print("Extracted Time Window:")
        print(json.dumps(time_window, indent=2))
        print("-" * 50)
