import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from request_to_time import extract_time_window
from get_calendar_events import retrive_calendar_events
from time_profiler import timeit


@timeit
def get_attendee_events(proposed_time, user_email):
    """get the attendee's calendar events for:
    1. the day of the propose start_time
    2. the next day of the proposed end_time"""
    user = user_email
    start_time = proposed_time["start_time"]
    end_time = proposed_time["end_time"]

    events = retrive_calendar_events(user, start_time, end_time)
    return events


@timeit
def get_attendee_events_2_days(proposed_time, user_email):
    """get the attendee's calendar events for:
    1. the day of the propose start_time
    2. the next day of the proposed end_time"""
    user = user_email
    start_day = proposed_time["start_time"].split("T")[0]
    end_day = proposed_time["end_time"].split("T")[0]
    end_day_plus_one = proposed_time["end_time"].split("T")[0]
    end_day_plus_one = end_day_plus_one.split("-")
    end_day_plus_one[2] = str(
        int(end_day_plus_one[2]) + 0
    )  # Note: This adds 0, should it be +1?
    end_day_plus_one = "-".join(end_day_plus_one)

    start_day_time_stamp = f"{start_day}T00:00:00+05:30"
    end_day_time_stamp = f"{end_day}T23:59:59+05:30"
    end_day_plus_one_time_stamp = f"{end_day_plus_one}T23:59:59+05:30"

    events = retrive_calendar_events(
        user, start_day_time_stamp, end_day_plus_one_time_stamp
    )
    return events


# OPTION 1: Using ThreadPoolExecutor (synchronous parallel execution)
@timeit
def get_all_attendee_events_parallel(proposed_time, input_request, max_workers=None):
    """Get all attendees' calendar events in parallel using ThreadPoolExecutor."""
    events = {}

    # Collect all attendees (sender + other attendees)
    all_attendees = [input_request["From"]]
    all_attendees.extend([attendee["email"] for attendee in input_request["Attendees"]])

    # Use ThreadPoolExecutor to fetch events in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_email = {
            executor.submit(get_attendee_events, proposed_time, email): email
            for email in all_attendees
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_email):
            email = future_to_email[future]
            try:
                events[email] = future.result()
            except Exception as exc:
                print(f"Attendee {email} generated an exception: {exc}")
                events[email] = None

    return events


@timeit
def get_all_attendee_events_2_days_parallel(
    proposed_time, input_request, max_workers=None
):
    """Get all attendees' calendar events for 2 days in parallel using ThreadPoolExecutor."""
    events = {}

    # Collect all attendees (sender + other attendees)
    all_attendees = [input_request["From"]]
    all_attendees.extend([attendee["email"] for attendee in input_request["Attendees"]])

    # Use ThreadPoolExecutor to fetch events in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_email = {
            executor.submit(get_attendee_events_2_days, proposed_time, email): email
            for email in all_attendees
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_email):
            email = future_to_email[future]
            try:
                events[email] = future.result()
            except Exception as exc:
                print(f"Attendee {email} generated an exception: {exc}")
                events[email] = None

    return events


# OPTION 2: Async version (if you want to make the underlying API calls async)
async def get_attendee_events_async(proposed_time, user_email):
    """Async version of get_attendee_events."""
    # Run the synchronous function in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, get_attendee_events, proposed_time, user_email
    )


async def get_attendee_events_2_days_async(proposed_time, user_email):
    """Async version of get_attendee_events_2_days."""
    # Run the synchronous function in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, get_attendee_events_2_days, proposed_time, user_email
    )


@timeit
async def get_all_attendee_events_async(proposed_time, input_request):
    """Get all attendees' calendar events asynchronously."""
    # Collect all attendees (sender + other attendees)
    all_attendees = [input_request["From"]]
    all_attendees.extend([attendee["email"] for attendee in input_request["Attendees"]])

    # Create tasks for all attendees
    tasks = [get_attendee_events_async(proposed_time, email) for email in all_attendees]

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Build the events dictionary
    events = {}
    for email, result in zip(all_attendees, results):
        if isinstance(result, Exception):
            print(f"Attendee {email} generated an exception: {result}")
            events[email] = None
        else:
            events[email] = result

    return events


@timeit
async def get_all_attendee_events_2_days_async(proposed_time, input_request):
    """Get all attendees' calendar events for 2 days asynchronously."""
    # Collect all attendees (sender + other attendees)
    all_attendees = [input_request["From"]]
    all_attendees.extend([attendee["email"] for attendee in input_request["Attendees"]])

    # Create tasks for all attendees
    tasks = [
        get_attendee_events_2_days_async(proposed_time, email)
        for email in all_attendees
    ]

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Build the events dictionary
    events = {}
    for email, result in zip(all_attendees, results):
        if isinstance(result, Exception):
            print(f"Attendee {email} generated an exception: {result}")
            events[email] = None
        else:
            events[email] = result

    return events


# Helper function to run async version from sync code
def run_async_get_all_attendee_events(proposed_time, input_request):
    """Wrapper to run the async version from synchronous code."""
    return asyncio.run(get_all_attendee_events_async(proposed_time, input_request))


def run_async_get_all_attendee_events_2_days(proposed_time, input_request):
    """Wrapper to run the async version from synchronous code."""
    return asyncio.run(
        get_all_attendee_events_2_days_async(proposed_time, input_request)
    )


# OPTION 3: If you want to keep your original functions and just parallelize the calls
@timeit
def get_all_attendee_events_original_parallel(proposed_time, input_request):
    """Original logic but with parallel execution."""
    events = {}

    # Get all attendee emails
    from_email = input_request["From"]
    attendee_emails = [attendee["email"] for attendee in input_request["Attendees"]]
    all_emails = [from_email] + attendee_emails

    # Function to get events for a single attendee
    def get_events_for_attendee(email):
        return email, get_attendee_events(proposed_time, email)

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=len(all_emails)) as executor:
        results = list(executor.map(get_events_for_attendee, all_emails))

    # Build events dictionary
    for email, event_data in results:
        events[email] = event_data

    return events


@timeit
def get_all_attendee_events_2_days_original_parallel(proposed_time, input_request):
    """Original 2-day logic but with parallel execution."""
    events = {}

    # Get all attendee emails
    from_email = input_request["From"]
    attendee_emails = [attendee["email"] for attendee in input_request["Attendees"]]
    all_emails = [from_email] + attendee_emails

    # Function to get events for a single attendee
    def get_events_for_attendee(email):
        return email, get_attendee_events_2_days(proposed_time, email)

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=len(all_emails)) as executor:
        results = list(executor.map(get_events_for_attendee, all_emails))

    # Build events dictionary
    for email, event_data in results:
        events[email] = event_data

    return events
