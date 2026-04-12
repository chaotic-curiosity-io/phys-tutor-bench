"""PhysTutorBench CLI — main entry point for the benchmark system."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console

console = Console()

# Default paths
DEFAULT_CONFIG = "config/default.yaml"
DEFAULT_TAXONOMY = "data/taxonomies/misconception_taxonomy.json"
DEFAULT_SCENARIOS = "data/scenarios"
DEFAULT_RESULTS = "results"


def load_config(config_path: str = DEFAULT_CONFIG) -> dict:
    """Load YAML configuration."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


@click.group()
@click.option("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """PhysTutorBench — Benchmark for evaluating AI physics tutoring quality."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.option("--topic", type=click.Choice(["mechanics", "em", "thermal_waves", "modern_quantum"]),
              help="Generate scenarios for a specific topic only")
@click.option("--count", type=int, default=None,
              help="Override per-misconception scenario count")
@click.option("--model", default="claude-sonnet-4-20250514",
              help="Model to use for generation")
@click.option("--taxonomy", default=DEFAULT_TAXONOMY, help="Path to taxonomy JSON")
@click.option("--output", default=DEFAULT_SCENARIOS, help="Output directory")
def generate_scenarios(topic: str | None, count: int | None, model: str,
                       taxonomy: str, output: str) -> None:
    """Generate tutoring scenarios from the misconception taxonomy."""
    from src.scenarios.generate_scenarios import generate_all_scenarios

    console.print(f"[bold]Generating scenarios[/bold]")
    if topic:
        console.print(f"  Topic filter: {topic}")
    if count:
        console.print(f"  Per-misconception count: {count}")

    scenarios = generate_all_scenarios(
        taxonomy_path=taxonomy,
        output_dir=output,
        topic_filter=topic,
        count_override=count,
        model=model,
    )
    console.print(f"[green]Generated {len(scenarios)} scenarios.[/green]")


@cli.command()
@click.option("--scenario", required=True, help="Scenario ID or path to scenario JSON")
@click.option("--model", required=True, help="Model name for the tutor under test")
@click.option("--max-turns", default=15, help="Maximum conversation turns")
@click.option("--output", default=DEFAULT_RESULTS, help="Output directory")
@click.option("--api-base", default=None, help="Custom API base URL")
@click.option("--api-key", default=None, help="Custom API key")
@click.pass_context
def run_single(ctx: click.Context, scenario: str, model: str,
               max_turns: int, output: str, api_base: str | None,
               api_key: str | None) -> None:
    """Run a single scenario against a model (for development/debugging)."""
    from src.engine.batch_runner import run_single_scenario, load_scenarios
    from src.engine.conversation_loop import save_conversation

    config = ctx.obj["config"]
    student_cfg = config.get("student_simulator", {})

    # Load the scenario
    scenario_path = Path(scenario)
    if scenario_path.exists():
        with open(scenario_path) as f:
            from src.scenarios.schema import Scenario
            sc = Scenario(**json.load(f))
    else:
        # Search by ID
        all_scenarios = load_scenarios(DEFAULT_SCENARIOS)
        matches = [s for s in all_scenarios if s.id == scenario]
        if not matches:
            console.print(f"[red]Scenario '{scenario}' not found.[/red]")
            raise SystemExit(1)
        sc = matches[0]

    console.print(f"[bold]Running scenario:[/bold] {sc.id}")
    console.print(f"  Model: {model}")
    console.print(f"  Topic: {sc.topic_area.value}/{sc.subtopic}")
    console.print(f"  Misconception: {sc.misconception_tags}")

    record = run_single_scenario(
        scenario=sc,
        model=model,
        student_model=student_cfg.get("model", "claude-sonnet-4-20250514"),
        student_temperature=student_cfg.get("temperature", 0.7),
        max_turns=max_turns,
        api_base=api_base,
        api_key=api_key,
    )

    out_path = save_conversation(record, output)
    console.print(f"\n[green]Conversation saved:[/green] {out_path}")
    console.print(f"  Turns: {record.total_turns}")
    console.print(f"  Termination: {record.termination_reason}")

    # Print transcript
    console.print(f"\n[bold]Transcript:[/bold]")
    for msg in record.messages:
        role_label = "[blue]TUTOR[/blue]" if msg.role == "tutor" else "[yellow]STUDENT[/yellow]"
        console.print(f"\n[Turn {msg.turn_number}] {role_label}")
        console.print(msg.content)


@cli.command()
@click.option("--model", required=True, help="Model name for the tutor under test")
@click.option("--concurrency", default=5, help="Number of parallel conversations")
@click.option("--max-turns", default=15, help="Maximum conversation turns")
@click.option("--scenarios-dir", default=DEFAULT_SCENARIOS, help="Path to scenarios directory")
@click.option("--output", default=DEFAULT_RESULTS, help="Output directory")
@click.option("--cost-limit", default=50.0, help="Maximum estimated cost in USD")
@click.option("--scenario-ids", default=None, help="Comma-separated list of specific scenario IDs")
@click.option("--api-base", default=None, help="Custom API base URL")
@click.option("--api-key", default=None, help="Custom API key")
@click.pass_context
def run_benchmark(ctx: click.Context, model: str, concurrency: int,
                  max_turns: int, scenarios_dir: str, output: str,
                  cost_limit: float, scenario_ids: str | None,
                  api_base: str | None, api_key: str | None) -> None:
    """Run the full benchmark suite against a model."""
    from src.engine.batch_runner import run_benchmark as _run_benchmark

    config = ctx.obj["config"]
    student_cfg = config.get("student_simulator", {})

    ids = scenario_ids.split(",") if scenario_ids else None

    _run_benchmark(
        scenarios_dir=scenarios_dir,
        model=model,
        output_dir=output,
        concurrency=concurrency,
        max_turns=max_turns,
        student_model=student_cfg.get("model", "claude-sonnet-4-20250514"),
        student_temperature=student_cfg.get("temperature", 0.7),
        cost_limit=cost_limit,
        api_base=api_base,
        api_key=api_key,
        scenario_ids=ids,
    )


@cli.command()
@click.option("--results-dir", required=True, help="Directory containing conversation records")
@click.option("--judge-model", default="claude-opus-4-20250514", help="Model for the LLM judge")
@click.option("--scenarios-dir", default=DEFAULT_SCENARIOS, help="Path to scenarios directory")
@click.option("--concurrency", default=3, help="Number of parallel scoring requests")
def score(results_dir: str, judge_model: str, scenarios_dir: str, concurrency: int) -> None:
    """Score a set of conversation records."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.engine.batch_runner import load_scenarios
    from src.engine.conversation_loop import load_conversation
    from src.scoring.llm_judge import LLMJudge
    from src.scoring.scorecard import save_score

    judge = LLMJudge(model=judge_model)

    # Load scenarios for reference
    all_scenarios = load_scenarios(scenarios_dir)
    scenario_map = {s.id: s for s in all_scenarios}

    # Load conversations
    results_path = Path(results_dir)
    conv_files = sorted(results_path.rglob("*.json"))
    conv_files = [f for f in conv_files if not f.name.endswith("_score.json")
                  and f.name not in ("run_metadata.json", "run_summary.json")]

    console.print(f"[bold]Scoring {len(conv_files)} conversations[/bold]")
    console.print(f"  Judge model: {judge_model}")

    scores_dir = results_path / "scores"

    def score_one(conv_path: Path):
        conv = load_conversation(conv_path)
        scenario = scenario_map.get(conv.scenario_id)
        if not scenario:
            return None, f"Scenario {conv.scenario_id} not found"
        result = judge.score(conv, scenario)
        save_score(result, scores_dir)
        return result, None

    scored = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(score_one, f): f for f in conv_files}
        for future in as_completed(futures):
            result, error = future.result()
            if error:
                console.print(f"[yellow]{error}[/yellow]")
                errors += 1
            else:
                scored += 1
                console.print(f"  Scored: {result.conversation_id} (composite: {result.composite_score:.2f})")

    console.print(f"\n[green]Scored {scored} conversations.[/green]")
    if errors:
        console.print(f"[yellow]{errors} errors.[/yellow]")


@cli.command()
@click.option("--results-dir", required=True, help="Base results directory")
@click.option("--compare", default=None, help="Comma-separated model names to compare")
@click.option("--output", default=None, help="Output directory for plots")
def scorecard(results_dir: str, compare: str | None, output: str | None) -> None:
    """Generate scorecard report with comparisons and visualizations."""
    from src.scoring.scorecard import load_scores, Scorecard

    results_path = Path(results_dir)
    all_scores = load_scores(results_path)

    if compare:
        model_names = [m.strip() for m in compare.split(",")]
        all_scores = [s for s in all_scores if s.model_under_test in model_names]

    if not all_scores:
        console.print("[red]No scores found.[/red]")
        raise SystemExit(1)

    sc = Scorecard(all_scores)
    sc.print_summary()

    # Generate plots
    out_dir = Path(output) if output else results_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = sc.generate_comparison_plot(out_dir / "comparison.png")
    console.print(f"  Comparison plot: {comparison_path}")

    heatmap_path = sc.generate_heatmap(out_dir / "heatmap.png")
    console.print(f"  Heatmap: {heatmap_path}")

    dist_paths = sc.generate_distribution_plots(out_dir / "distributions")
    console.print(f"  Distribution plots: {len(dist_paths)} generated")

    # Save JSON summary
    summary_path = out_dir / "scorecard.json"
    summary_path.write_text(json.dumps(sc.to_json(), indent=2))
    console.print(f"  JSON summary: {summary_path}")


@cli.command()
@click.option("--conversations", required=True, help="Directory containing conversations to annotate")
@click.option("--db", default="data/validation/human_annotations.db", help="SQLite database path")
def annotate(conversations: str, db: str) -> None:
    """Launch the Streamlit annotation interface for human validation."""
    console.print("[bold]Launching annotation interface...[/bold]")
    console.print(f"  Conversations: {conversations}")
    console.print(f"  Database: {db}")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/validation/human_annotation_interface.py",
        "--",
        conversations,
    ])


@cli.command()
@click.option("--annotations", default="data/validation/human_annotations.db",
              help="Path to annotations SQLite database")
@click.option("--results-dir", default=None, help="Results directory for construct validity (needs scores)")
@click.option("--scenarios-dir", default=DEFAULT_SCENARIOS, help="Scenarios directory for discriminant validity")
@click.option("--run-discriminant", is_flag=True, help="Run discriminant validity testing (requires API calls)")
@click.option("--n-scenarios", default=5, help="Number of scenarios for discriminant testing")
def validate(annotations: str, results_dir: str | None, scenarios_dir: str,
             run_discriminant: bool, n_scenarios: int) -> None:
    """Run validation analyses (agreement, construct validity, discriminant validity)."""
    # Agreement analysis
    if Path(annotations).exists():
        console.print("\n[bold]Running inter-rater agreement analysis...[/bold]")
        from src.validation.agreement_analysis import run_agreement_analysis
        run_agreement_analysis(annotations)
    else:
        console.print(f"[yellow]No annotations found at {annotations}. Skipping agreement analysis.[/yellow]")

    # Construct validity
    if results_dir:
        console.print("\n[bold]Running construct validity analysis...[/bold]")
        from src.scoring.scorecard import load_scores
        from src.validation.construct_validity import run_construct_validity
        scores = load_scores(results_dir)
        if scores:
            run_construct_validity(scores)
        else:
            console.print("[yellow]No scores found. Skipping construct validity.[/yellow]")

    # Discriminant validity
    if run_discriminant:
        console.print("\n[bold]Running discriminant validity testing...[/bold]")
        from src.engine.batch_runner import load_scenarios
        from src.scoring.llm_judge import LLMJudge
        from src.validation.discriminant_validity import run_discriminant_validity

        scenarios = load_scenarios(scenarios_dir)
        if scenarios:
            judge = LLMJudge()
            run_discriminant_validity(scenarios, judge, n_scenarios=n_scenarios)
        else:
            console.print("[yellow]No scenarios found. Skipping discriminant validity.[/yellow]")


if __name__ == "__main__":
    cli()
