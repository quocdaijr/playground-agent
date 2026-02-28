"""
PlaygroundAgent Evaluation Runner
======================
Runs a suite of test cases against the live agent graph and reports pass/fail.
Reports are saved as JSON to evals/reports/.

Usage:
    python -m evals.evaluator
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.core.langgraph.graph import agent_graph
from app.utils.graph import build_initial_state, extract_reply

EVAL_CASES = [
    {"input": "Calculate sqrt(144)", "expected_contains": "12"},
    {"input": "What is 2 ** 10?", "expected_contains": "1024"},
    {"input": "What time is it?", "expected_contains": "UTC"},
    {"input": "Hello!", "expected_contains": None},  # Just verify the agent responds
    {"input": "Sort this list: 9,3,7,1,5", "expected_contains": "1"},
]


async def run_evals() -> dict:
    results = []
    for case in EVAL_CASES:
        state = build_initial_state(case["input"])
        output = await agent_graph.ainvoke(state)
        reply = extract_reply(output)

        passed = True
        if case["expected_contains"]:
            passed = case["expected_contains"].lower() in reply.lower()

        results.append({
            "input": case["input"],
            "reply": reply,
            "passed": passed,
            "expected_contains": case["expected_contains"],
        })
        print(f"{'✅' if passed else '❌'} {case['input']!r} → {reply[:80]!r}")

    success_rate = sum(r["passed"] for r in results) / len(results)
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "success_rate": round(success_rate, 3),
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "results": results,
    }

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\nReport saved: {report_path}")
    print(f"Success rate: {success_rate:.0%}")
    return report


if __name__ == "__main__":
    asyncio.run(run_evals())
