#!/usr/bin/env python3
"""
Test script for the Meeting Time Extraction functionality.
This script can be run independently to test the time extraction function.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_offline_tests():
    """Run tests without LLM to verify basic logic."""

    print("Running Offline Tests (without LLM)...")
    print("=" * 50)

    # Mock the LLM response for offline testing
    class MockExtractor:
        def __init__(self):
            self.timezone_offset = "+05:30"

        def extract_meeting_info(self, request_data):
            from request_to_time import MeetingTimeExtractor

            # Create a real extractor but override the LLM call
            extractor = MeetingTimeExtractor.__new__(MeetingTimeExtractor)
            extractor.timezone_offset = "+05:30"

            email_content = request_data.get("EmailContent", "").lower()

            # Simple parsing for testing
            if "30 minutes" in email_content:
                duration = 30
            elif "hour" in email_content or "long" in email_content:
                duration = 60
            else:
                duration = 30

            # Extract relative time
            relative_time = ""
            if "thursday" in email_content:
                relative_time = "Thursday"
            elif "monday" in email_content and "9:00 am" in email_content:
                relative_time = "Monday at 9:00 AM"
            elif "tuesday" in email_content and "11:00" in email_content:
                relative_time = "Tuesday at 11:00 AM"
            elif "wednesday" in email_content and "10:00" in email_content:
                relative_time = "Wednesday at 10:00 AM"

            # Use the real calculation methods
            base_datetime = extractor._parse_datetime(request_data.get("Datetime", ""))
            start_time, end_time = extractor._calculate_meeting_times(
                base_datetime, duration, relative_time
            )

            return {
                "duration_minutes": duration,
                "start_time": start_time,
                "end_time": end_time,
                "relative_time": relative_time,
            }

    # Test cases
    test_cases = [
        {
            "name": "Test Case 1 - Thursday 30 minutes",
            "data": {
                "Request_id": "6118b54f-907b-4451-8d48-dd13d76033a5",
                "Datetime": "19-07-2025T12:34:55",
                "EmailContent": "Hi team, let's meet on Thursday for 30 minutes to discuss the status of Agentic AI Project.",
            },
            "expected": {"duration_minutes": 30, "day": "Thursday"},
        },
        {
            "name": "Test Case 2 - Monday 9:00 AM",
            "data": {
                "Request_id": "6118b54f-907b-4451-8d48-dd13d76033b5",
                "Datetime": "19-07-2025T12:34:55",
                "EmailContent": "Let's meet Monday at 9:00 AM to discuss and resolve this issue.",
            },
            "expected": {"duration_minutes": 30, "day": "Monday", "time": "09:00"},
        },
        {
            "name": "Test Case 3 - Tuesday 11:00 AM",
            "data": {
                "Request_id": "6118b54f-907b-4451-8d48-dd13d76033c5",
                "Datetime": "19-07-2025T12:34:55",
                "EmailContent": "Let's meet on Tuesday at 11:00 A.M and discuss about our on-going Projects.",
            },
            "expected": {"duration_minutes": 30, "day": "Tuesday", "time": "11:00"},
        },
    ]

    mock_extractor = MockExtractor()

    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 30)

        try:
            result = mock_extractor.extract_meeting_info(test_case["data"])

            print(f"Email: {test_case['data']['EmailContent']}")
            print(f"Duration: {result['duration_minutes']} minutes ✓")
            print(f"Start Time: {result['start_time']}")
            print(f"End Time: {result['end_time']}")
            print(f"Relative Time: {result['relative_time']}")

            # Basic validation
            assert (
                result["duration_minutes"] == test_case["expected"]["duration_minutes"]
            )
            if "day" in test_case["expected"]:
                assert (
                    test_case["expected"]["day"].lower()
                    in result["relative_time"].lower()
                )

            print("✓ Test PASSED")

        except Exception as e:
            print(f"✗ Test FAILED: {e}")

    print("\n" + "=" * 50)
    print("Offline testing completed!")


def show_usage_example():
    """Show how to use the function."""

    print("\nUsage Example:")
    print("=" * 50)

    example_request = {
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
    }

    print("Input JSON:")
    import json

    print(json.dumps(example_request, indent=2))

    print("\nCode to extract time info:")
    print("""
from request_to_time import extract_meeting_time_info

# Extract meeting time information
result = extract_meeting_time_info(request_data)

print(f"Duration: {result['duration_minutes']} minutes")
print(f"Start Time: {result['start_time']}")
print(f"End Time: {result['end_time']}")
""")

    print("\nExpected Output:")
    print("""
Duration: 30 minutes
Start Time: 2025-07-24T10:00:00+05:30
End Time: 2025-07-24T10:30:00+05:30
""")


if __name__ == "__main__":
    print("Meeting Time Extraction - Test Suite")
    print("====================================")

    # Run offline tests
    run_offline_tests()

    # Show usage example
    show_usage_example()

    print("\nNote: For full functionality with LLM, ensure:")
    print("1. vLLM server is running on http://localhost:3000/v1")
    print("2. OpenAI library is installed: pip install openai")
    print(
        "3. Model is available at: /home/user/Models/deepseek-ai/deepseek-llm-7b-chat"
    )
