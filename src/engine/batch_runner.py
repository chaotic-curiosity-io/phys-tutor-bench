"""Batch runner for executing the full scenario bank against one or more models."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.engine.conversation_loop import ConversationLoop, save_conversation
from src.engine.student_simulator import StudentSimulator
from src.engine.tutor_runner import create_tutor_backend, DEFAULT_TUTOR_SYSTEM_PROMPT
from src.scenarios.schema import Scenario, ConversationRecord

console = Console()


def load_scenarios(scenarios_dir: str | Path) -> list[Scenario]:
    """Load all scenario JSON files from a directory tree."""
    scenarios_dir = Path(scenarios_dir)
    scenarios = []
    for path in sorted(scenarios_dir.rglob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            scenarios.append(Scenario(**data))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {path}: {e}[/yellow]")
    return scenarios


def estimate_cost(
    n_scenarios: int,
    avg_turns: int = 10,
    avg_tokens_per_turn: int = 800,
    input_cost_per_1k: float = 0.003,
    output_cost_per_1k: float = 0.015,
) -> float:
    """Estimate the API cost for a benchmark run."""
    # Each scenario: tutor + student responses for avg_turns
    total_tokens = n_scenarios * avg_turns * avg_tokens_per_turn * 2
    input_tokens = total_tokens * 0.6  # rough split
    output_tokens = total_tokens * 0.4
    cost = (input_tokens / 1000 * input_cost_per_1k) + (output_tokens / 1000 * output_cost_per_1k)
    return cost


def run_single_scenario(
    scenario: Scenario,
    model: str,
    student_model: str = "claude-sonnet-4-20250514",
    student_temperature: float = 0.7,
    max_turns: int = 15,
    transfer_injection_offset: int = 2,
    tutor_system_prompt: str = DEFAULT_TUTOR_SYSTEM_PROMPT,
    api_base: str | None = None,
    api_key: str | None = None,
) -> ConversationRecord:
    """Run a single scenario and return the conversation record."""
    tutor = create_tutor_backend(model, api_base=api_base, api_key=api_key)
    student = StudentSimulator(
        scenario=scenario,
        model=student_model,
        temperature=student_temperature,
    )
    loop = ConversationLoop(
        scenario=scenario,
        tutor=tutor,
        student=student,
        max_turns=max_turns,
        transfer_injection_offset=transfer_injection_offset,
        tutor_system_prompt=tutor_system_prompt,
    )
    return loop.run()


def run_benchmark(
    scenarios_dir: str | Path,
    model: str,
    output_dir: str | Path = "results",
    concurrency: int = 5,
    max_turns: int = 15,
    student_model: str = "claude-sonnet-4-20250514",
    student_temperature: float = 0.7,
    cost_limit: float = 50.0,
    api_base: str | None = None,
    api_key: str | None = None,
    scenario_ids: list[str] | None = None,
) -> Path:
    """Run the full benchmark suite against a model.

    Args:
        scenarios_dir: Directory containing scenario JSON files
        model: Model identifier for the tutor under test
        output_dir: Base output directory
        concurrency: Number of parallel conversations
        max_turns: Maximum turns per conversation
        student_model: Model for the student simulator
        student_temperature: Temperature for student simulator
        cost_limit: Abort if estimated cost exceeds this (USD)
        api_base: Optional API base URL for generic backends
        api_key: Optional API key for generic backends
        scenario_ids: Optional list of specific scenario IDs to run

    Returns:
        Path to the results directory for this run.
    """
    scenarios = load_scenarios(scenarios_dir)
    if scenario_ids:
        scenarios = [s for s in scenarios if s.id in scenario_ids]

    if not scenarios:
        console.print("[red]No scenarios found![/red]")
        raise SystemExit(1)

    # Cost estimation
    est_cost = estimate_cost(len(scenarios))
    console.print(f"\n[bold]Benchmark Run Configuration[/bold]")
    console.print(f"  Model under test: {model}")
    console.print(f"  Student simulator: {student_model}")
    console.print(f"  Scenarios: {len(scenarios)}")
    console.print(f"  Max turns: {max_turns}")
    console.print(f"  Concurrency: {concurrency}")
    console.print(f"  Estimated cost: ${est_cost:.2f}")

    if est_cost > cost_limit:
        console.print(f"[red]Estimated cost ${est_cost:.2f} exceeds limit ${cost_limit:.2f}. Aborting.[/red]")
        raise SystemExit(1)

    # Set up output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    model_slug = model.replace("/", "_").replace(":", "_")
    run_dir = Path(output_dir) / model_slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    run_meta = {
        "model_under_test": model,
        "student_model": student_model,
        "student_temperature": student_temperature,
        "max_turns": max_turns,
        "n_scenarios": len(scenarios),
        "timestamp": timestamp,
        "concurrency": concurrency,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    # Execute scenarios
    records: list[ConversationRecord] = []
    errors: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running scenarios...", total=len(scenarios))

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_scenario = {}
            for scenario in scenarios:
                future = executor.submit(
                    run_single_scenario,
                    scenario=scenario,
                    model=model,
                    student_model=student_model,
                    student_temperature=student_temperature,
                    max_turns=max_turns,
                    api_base=api_base,
                    api_key=api_key,
                )
                future_to_scenario[future] = scenario

            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    record = future.result()
                    records.append(record)
                    save_conversation(record, run_dir / "conversations")
                except Exception as e:
                    errors.append({
                        "scenario_id": scenario.id,
                        "error": str(e),
                    })
                    console.print(f"[red]Error on {scenario.id}: {e}[/red]")
                finally:
                    progress.advance(task)

    # Save summary
    summary = {
        "total_scenarios": len(scenarios),
        "completed": len(records),
        "errors": len(errors),
        "error_details": errors,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    console.print(f"\n[green]Benchmark complete:[/green] {len(records)}/{len(scenarios)} scenarios")
    console.print(f"Results saved to: {run_dir}")

    if errors:
        console.print(f"[yellow]{len(errors)} errors occurred. See run_summary.json.[/yellow]")

    return run_dir
