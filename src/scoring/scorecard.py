"""Scorecard generation — aggregation, comparison, and visualization of benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.table import Table

from src.scoring.rubric import ALL_DIMENSIONS, DEFAULT_WEIGHTS, compute_composite_score
from src.scenarios.schema import ConversationScore, DimensionScore

console = Console()


def load_scores(results_dir: str | Path) -> list[ConversationScore]:
    """Load all score JSON files from a results directory."""
    results_dir = Path(results_dir)
    scores = []
    for path in sorted(results_dir.rglob("*_score.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            scores.append(ConversationScore(**data))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {path}: {e}[/yellow]")
    return scores


def save_score(score: ConversationScore, output_dir: str | Path) -> Path:
    """Save a conversation score to a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{score.conversation_id}_score.json"
    path.write_text(json.dumps(score.model_dump(), indent=2))
    return path


class Scorecard:
    """Aggregates and analyzes benchmark scores for one or more models."""

    def __init__(self, scores: list[ConversationScore]):
        self.scores = scores
        self._by_model: dict[str, list[ConversationScore]] = {}
        for s in scores:
            self._by_model.setdefault(s.model_under_test, []).append(s)

    @property
    def models(self) -> list[str]:
        return sorted(self._by_model.keys())

    def model_scores(self, model: str) -> list[ConversationScore]:
        return self._by_model.get(model, [])

    def dimension_average(self, model: str, dimension_id: str) -> float:
        """Average score for a model on a specific dimension."""
        model_scores = self.model_scores(model)
        if not model_scores:
            return 0.0
        values = []
        for cs in model_scores:
            for ds in cs.scores:
                if ds.dimension == dimension_id:
                    values.append(ds.score)
        return np.mean(values) if values else 0.0

    def dimension_std(self, model: str, dimension_id: str) -> float:
        """Standard deviation for a model on a specific dimension."""
        model_scores = self.model_scores(model)
        if not model_scores:
            return 0.0
        values = []
        for cs in model_scores:
            for ds in cs.scores:
                if ds.dimension == dimension_id:
                    values.append(ds.score)
        return float(np.std(values)) if values else 0.0

    def composite_average(self, model: str, weights: dict[str, float] | None = None) -> float:
        """Average composite score for a model."""
        model_scores = self.model_scores(model)
        if not model_scores:
            return 0.0
        return np.mean([s.composite_score for s in model_scores])

    def per_topic_breakdown(self, model: str, scenario_map: dict[str, str] | None = None) -> dict[str, dict[str, float]]:
        """Break down scores by topic area.

        Args:
            scenario_map: Optional mapping of scenario_id -> topic_area.
                If not provided, attempts to infer from scenario_id prefix.
        """
        topic_scores: dict[str, list[float]] = {}
        for cs in self.model_scores(model):
            topic = self._infer_topic(cs.scenario_id, scenario_map)
            topic_scores.setdefault(topic, []).append(cs.composite_score)

        return {
            topic: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "count": len(vals),
            }
            for topic, vals in topic_scores.items()
        }

    def _infer_topic(self, scenario_id: str, scenario_map: dict[str, str] | None = None) -> str:
        """Infer topic area from scenario ID prefix."""
        if scenario_map and scenario_id in scenario_map:
            return scenario_map[scenario_id]
        prefix = scenario_id.split("-")[0]
        prefix_map = {
            "mech": "mechanics",
            "em": "em",
            "thwv": "thermal_waves",
            "modq": "modern_quantum",
        }
        return prefix_map.get(prefix, "unknown")

    def print_summary(self) -> None:
        """Print a rich table summary to the console."""
        table = Table(title="PhysTutorBench Scorecard")
        table.add_column("Model", style="bold")
        table.add_column("N", justify="right")
        for dim in ALL_DIMENSIONS:
            table.add_column(dim.abbreviation, justify="right")
        table.add_column("Composite", justify="right", style="bold green")

        for model in self.models:
            n = len(self.model_scores(model))
            row = [model, str(n)]
            for dim in ALL_DIMENSIONS:
                avg = self.dimension_average(model, dim.id)
                row.append(f"{avg:.2f}")
            comp = self.composite_average(model)
            row.append(f"{comp:.2f}")
            table.add_row(*row)

        console.print(table)

    def generate_comparison_plot(self, output_path: str | Path) -> Path:
        """Generate a bar chart comparing models across dimensions."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        models = self.models
        n_models = len(models)
        n_dims = len(ALL_DIMENSIONS)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_dims)
        width = 0.8 / max(n_models, 1)

        for i, model in enumerate(models):
            means = [self.dimension_average(model, dim.id) for dim in ALL_DIMENSIONS]
            stds = [self.dimension_std(model, dim.id) for dim in ALL_DIMENSIONS]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=model, capsize=3)

        ax.set_ylabel("Score (0-4)")
        ax.set_title("PhysTutorBench — Model Comparison by Dimension")
        ax.set_xticks(x)
        ax.set_xticklabels([dim.abbreviation for dim in ALL_DIMENSIONS])
        ax.set_ylim(0, 4.5)
        ax.legend(loc="upper right", fontsize="small")
        ax.axhline(y=2, color="gray", linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def generate_distribution_plots(self, output_dir: str | Path) -> list[Path]:
        """Generate score distribution plots for each dimension."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for dim in ALL_DIMENSIONS:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model in self.models:
                values = []
                for cs in self.model_scores(model):
                    for ds in cs.scores:
                        if ds.dimension == dim.id:
                            values.append(ds.score)
                if values:
                    ax.hist(values, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                            alpha=0.5, label=model, edgecolor="black")

            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.set_title(f"{dim.name} ({dim.abbreviation}) — Score Distribution")
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.legend()
            plt.tight_layout()

            path = output_dir / f"dist_{dim.id}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)

        return paths

    def generate_heatmap(self, output_path: str | Path) -> Path:
        """Generate a heatmap of model x dimension scores."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        models = self.models
        dim_ids = [dim.id for dim in ALL_DIMENSIONS]
        dim_labels = [dim.abbreviation for dim in ALL_DIMENSIONS]

        data = np.zeros((len(models), len(dim_ids)))
        for i, model in enumerate(models):
            for j, dim_id in enumerate(dim_ids):
                data[i, j] = self.dimension_average(model, dim_id)

        fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.8)))
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            xticklabels=dim_labels,
            yticklabels=models,
            vmin=0,
            vmax=4,
            cmap="RdYlGn",
            ax=ax,
        )
        ax.set_title("PhysTutorBench — Scores by Model and Dimension")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def to_json(self) -> dict:
        """Export scorecard data as a JSON-serializable dict."""
        result = {}
        for model in self.models:
            model_data = {
                "n_conversations": len(self.model_scores(model)),
                "composite_mean": float(self.composite_average(model)),
                "dimensions": {},
            }
            for dim in ALL_DIMENSIONS:
                model_data["dimensions"][dim.id] = {
                    "mean": float(self.dimension_average(model, dim.id)),
                    "std": float(self.dimension_std(model, dim.id)),
                }
            result[model] = model_data
        return result
