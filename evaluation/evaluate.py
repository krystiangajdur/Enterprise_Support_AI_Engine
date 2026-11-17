from time import perf_counter
from datetime import datetime, timezone
import uuid
import json
import traceback

from app.agent import ask
from app.config import EVALUATION_JSON, RESULTS_DIR
from evaluation.checks import check_tool_choice, calculate_accuracy, classify_error


def evaluate_turn(turn, scenario, run_id, turn_number):
    scenario_id = scenario["id"]

    customer_id = turn.get("customer_id", scenario["customer_id"])
    session_key = turn.get(
        "session_key",
        scenario.get("session_key", "default")
    )

    session_id = f"eval-{run_id}-{scenario_id}-{session_key}"
    question = turn["question"]

    expected_tools = turn.get("expected_tools")

    started = perf_counter()

    record = {
        "scenario_id": scenario_id,
        "category": scenario["category"],
        "turn_number": turn_number,
        "customer_id": customer_id,
        "session_id": session_id,
        "question": question,
        "answer": None,
        "actual_tools": None,
        "expected_tools": expected_tools,
        "latency_seconds": None,
        "checks": {
            name: "NOT_EVALUATED"
            for name in turn.get("checks", [])
        },
        "error": None,
        "error_type": None,
    }

    try:
        response = ask(
            question=question,
            customer_id=customer_id,
            session_id=session_id,
        )

        record["answer"] = response["answer"]
        record["actual_tools"] = response["tools"]

        if not isinstance(record["actual_tools"], list):
            raise TypeError("tools_used must be a list")

        if expected_tools is not None:
            passed = check_tool_choice(
                record["actual_tools"],
                expected_tools
            )

            record["checks"]["tool_choice"] = (
                "PASS" if passed else "FAIL"
            )
        else:
            record["checks"]["tool_choice"] = "NOT_EVALUATED"

    except Exception as exc:
        error_type = classify_error(exc)

        record["error"] = str(exc)
        record["error_type"] = error_type

        print(
            f"[{error_type}] {scenario_id} "
            f"turn {turn_number}: {exc}"
        )

        if expected_tools is not None:
            record["checks"]["tool_choice"] = "ERROR"

    finally:
        record["latency_seconds"] = round(
            perf_counter() - started, 3
        )

    return record


def evaluate_scenario(scenario, run_id):
    results = []

    print(f"\n[{scenario['category']}] {scenario['id']}")

    for turn_number, turn in enumerate(
        scenario["turns"], start=1
    ):
        record = evaluate_turn(
            turn=turn,
            scenario=scenario,
            run_id=run_id,
            turn_number=turn_number,
        )

        results.append(record)

        print(
            f"  Turn {turn_number}: "
            f"{record['latency_seconds']:.2f}s | "
            f"{record['checks'].get('tool_choice', 'NOT_EVALUATED')}"
        )

    return results



def save_results(run_id, results):

    tool_results = [
        record["checks"]["tool_choice"]
        for record in results
        if record["checks"].get("tool_choice") in ("PASS", "FAIL")
    ]

    tool_accuracy = calculate_accuracy([
        status == "PASS"
        for status in tool_results
    ])

    execution_errors = sum(
        record["error"] is not None
        for record in results
    )

    rate_limit_errors = sum(
        record.get("error_type") == "RATE_LIMIT"
        for record in results
    )

    other_errors = execution_errors - rate_limit_errors

    execution_success_rate = calculate_accuracy([
        record["error"] is None
        for record in results
    ])

    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),

        "evaluation_summary": {
            "total_turns": len(results),
            "tool_choice_accuracy": tool_accuracy,
            "tool_choice_evaluated": len(tool_results),
            "execution_success_rate": execution_success_rate,
            "execution_errors": execution_errors,
            "rate_limit_errors": rate_limit_errors,
            "other_errors": other_errors
        },

        "results": results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / f"results-{run_id}.json","w",encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report



def run_evaluation():
    with open(EVALUATION_JSON, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    run_id = uuid.uuid4().hex[:12]
    results = []

    for scenario in dataset["scenarios"][29:40]:
        scenario_results = evaluate_scenario(
            scenario=scenario,
            run_id=run_id
        )

        results.extend(scenario_results)

    report = save_results(run_id, results)

    summary = report["evaluation_summary"]

    print("\nEvaluation completed!")
    print(f"Total turns: {summary['total_turns']}")
    print(f"Tool choice accuracy: {summary['tool_choice_accuracy']}%")
    print(f"Tool choice evaluated: {summary['tool_choice_evaluated']}")
    print(f"Execution success rate: {summary['execution_success_rate']}%")
    print(f"Execution errors: {summary['execution_errors']}")
    print(f"Rate limit errors: {summary['rate_limit_errors']}")
    print(f"Other errors: {summary['other_errors']}")
    print(f"Results saved to: {RESULTS_DIR / f'results-{run_id}.json'}")


if __name__ == "__main__":
    run_evaluation()