import time

def check_tool_choice(actual_tools, expected_tools):
    return set(actual_tools) == set(expected_tools)

def check_answer_contains(answer, expected_values):
    answer = answer.casefold()

    return all(
        str(value).casefold() in answer
        for value in expected_values
    )

def check_no_forbidden_values(answer, forbidden_values):
    answer = answer.casefold()

    return all(
        str(value).casefold() not in answer
        for value in forbidden_values
    )

def measure_response_time(function, *args, **kwargs):
    start = time.perf_counter()

    result = function(*args, **kwargs)

    elapsed = time.perf_counter() - start

    return result, elapsed

def calculate_accuracy(results):
    if not results:
        return None

    return sum(results) / len(results) * 100