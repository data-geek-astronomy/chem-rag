"""
evaluate.py - Evaluation metrics for chemical property extraction accuracy.

Metrics:
- Exact match: extracted value == ground truth (within tolerance)
- Unit-normalized match: values equal after converting to Celsius/SI
- Recall: what % of ground truth properties did we find
- Precision: of what we extracted, what % is correct
- F1: harmonic mean of precision and recall
"""

import json
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.table import Table

from extract import ChemicalProperties, TemperatureValue, normalize_to_celsius

console = Console()


@dataclass
class GroundTruth:
    """A verified chemical property entry for evaluation."""
    compound_name: str
    property_name: str        # e.g., "boiling_point_celsius", "flash_point_celsius"
    expected_value: float     # Always in SI/Celsius for temperatures
    tolerance: float = 2.0   # ± tolerance for numeric match (°C)
    source: str = ""          # Reference source


@dataclass
class EvalResult:
    compound_name: str
    property_name: str
    expected: float
    extracted: Optional[float]
    match: bool
    delta: Optional[float]    # Difference from expected


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def matched(self) -> int:
        return sum(1 for r in self.results if r.match)

    @property
    def extracted_count(self) -> int:
        return sum(1 for r in self.results if r.extracted is not None)

    @property
    def precision(self) -> float:
        if self.extracted_count == 0:
            return 0.0
        correct_of_extracted = sum(1 for r in self.results if r.extracted is not None and r.match)
        return correct_of_extracted / self.extracted_count

    @property
    def recall(self) -> float:
        if self.total == 0:
            return 0.0
        return self.matched / self.total

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return round(2 * p * r / (p + r), 4)


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _get_temperature_celsius(temp: Optional[TemperatureValue]) -> Optional[float]:
    """Extract celsius value from a TemperatureValue, normalizing if needed."""
    if temp is None or temp.value is None:
        return None
    if temp.value_celsius is not None:
        return temp.value_celsius
    return normalize_to_celsius(temp.value, temp.unit)


def evaluate_extraction(
    extracted: ChemicalProperties,
    ground_truths: list[GroundTruth],
) -> EvalReport:
    """
    Compare extracted properties against ground truth entries.
    Only evaluates properties present in the ground truth list.
    """
    report = EvalReport()

    # Map property names to extracted values (all in Celsius where applicable)
    extracted_values = {
        "boiling_point_celsius": _get_temperature_celsius(extracted.boiling_point),
        "melting_point_celsius": _get_temperature_celsius(extracted.melting_point),
        "flash_point_celsius": _get_temperature_celsius(extracted.flash_point),
        "molecular_weight": extracted.molecular_weight,
        "density": extracted.density,
    }

    for gt in ground_truths:
        if gt.compound_name.lower() != extracted.compound_name.lower():
            continue

        extracted_val = extracted_values.get(gt.property_name)
        delta = None
        match = False

        if extracted_val is not None:
            delta = round(abs(extracted_val - gt.expected_value), 3)
            match = delta <= gt.tolerance

        report.results.append(EvalResult(
            compound_name=gt.compound_name,
            property_name=gt.property_name,
            expected=gt.expected_value,
            extracted=extracted_val,
            match=match,
            delta=delta,
        ))

    return report


def print_eval_report(report: EvalReport, title: str = "Evaluation Results") -> None:
    """Print a formatted evaluation report table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Compound", style="cyan")
    table.add_column("Property", style="white")
    table.add_column("Expected", justify="right")
    table.add_column("Extracted", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Match", justify="center")

    for r in report.results:
        match_str = "[green]✓[/green]" if r.match else (
            "[yellow]~[/yellow]" if r.extracted is not None else "[red]✗[/red]"
        )
        delta_str = f"±{r.delta}" if r.delta is not None else "—"
        extracted_str = f"{r.extracted:.2f}" if r.extracted is not None else "NOT FOUND"

        table.add_row(
            r.compound_name,
            r.property_name,
            f"{r.expected:.2f}",
            extracted_str,
            delta_str,
            match_str,
        )

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"Precision={report.precision:.0%}  "
        f"Recall={report.recall:.0%}  "
        f"F1={report.f1:.3f}  "
        f"({report.matched}/{report.total} matched)"
    )


# ---------------------------------------------------------------------------
# Ground truth loading (CSV format for easy editing)
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: Path) -> list[GroundTruth]:
    """
    Load ground truth from CSV.
    Expected columns: compound_name, property_name, expected_value, tolerance, source
    """
    entries = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(GroundTruth(
                compound_name=row["compound_name"].strip(),
                property_name=row["property_name"].strip(),
                expected_value=float(row["expected_value"]),
                tolerance=float(row.get("tolerance", 2.0)),
                source=row.get("source", ""),
            ))
    console.print(f"[cyan]Loaded {len(entries)} ground truth entries from {csv_path.name}[/cyan]")
    return entries


def save_ground_truth_template(csv_path: Path) -> None:
    """Create a template ground truth CSV for users to fill in."""
    rows = [
        {"compound_name": "Ethanol", "property_name": "boiling_point_celsius", "expected_value": 78.37, "tolerance": 2.0, "source": "NIST"},
        {"compound_name": "Ethanol", "property_name": "flash_point_celsius", "expected_value": 13.0, "tolerance": 2.0, "source": "NIST"},
        {"compound_name": "Ethanol", "property_name": "melting_point_celsius", "expected_value": -114.1, "tolerance": 2.0, "source": "NIST"},
        {"compound_name": "Acetone", "property_name": "boiling_point_celsius", "expected_value": 56.05, "tolerance": 2.0, "source": "NIST"},
        {"compound_name": "Acetone", "property_name": "flash_point_celsius", "expected_value": -20.0, "tolerance": 2.0, "source": "NIST"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["compound_name", "property_name", "expected_value", "tolerance", "source"])
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]Ground truth template saved to {csv_path}[/green]")
