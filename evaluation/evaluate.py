from time import perf_counter
from datetime import datetime, timezone
import uuid
import json

from app.agent import ask
from app.config import EVALUATION_JSON, RESULTS_DIR
from evaluation.checks import check_tool_choice

def evaluate_turn(turn, scenario, run_id, turn_number):
    scenario_id = scenario["id"]

    customer_id = turn.get("customer_id", scenario["customer_id"])
    session_key = turn.get(
        "session_key",
        scenario.get("session_key", "default")
    )

    session_id = f"eval-{run_id}-{scenario_id}-{session_key}"
    question = turn["question"]

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
        "expected_tools": turn.get("expected_tools"),
        "latency_seconds": None,
        "checks": {
            name: "NOT_EVALUATED"
            for name in turn.get("checks", [])
        },
        "error": None,
    }

    print(f"Expected tools: {turn.get('expected_tools', 'MISSING')}")

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

        if "expected_tools" in turn:
            passed = check_tool_choice(
                record["actual_tools"],
                turn["expected_tools"]
            )

            record["checks"]["tool_choice"] = (
                "PASS" if passed else "FAIL"
            )
        else:
            record["checks"]["tool_choice"] = "NOT_EVALUATED"

    except Exception as exc:
        record["error"] = str(exc)

        print(f"[ERROR] {type(exc).__name__}: {exc}")

        if "expected_tools" in turn:
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
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_turns": len(results),
        "results": results,
    }

    results_path = RESULTS_DIR

    results_path = results_path.with_name(
        f"results-{run_id}.json"
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report

def run_evaluation():
    with open(EVALUATION_JSON, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    run_id = uuid.uuid4().hex[:12]
    results = []

    for scenario in dataset["scenarios"][1:10]:
        scenario_results = evaluate_scenario(
            scenario=scenario,
            run_id=run_id
        )

        results.extend(scenario_results)

    report = save_results(run_id, results)

    print(f"\nEvaluation completed!")
    print(f"Total turns: {report['total_turns']}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    run_evaluation()
