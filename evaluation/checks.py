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

def calculate_accuracy(results):
    if not results:
        return None

    return round(sum(results) / len(results) * 100, 2)

def classify_error(exc):
    error_message = str(exc).upper()

    if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
        return "RATE_LIMIT"

    return "EXECUTION_ERROR"