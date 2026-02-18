"""
extract.py - LLM-powered chemical property extraction using structured outputs.

Handles:
- Structured extraction of chemical properties (boiling point, flash point, etc.)
- Unit normalization (°C, °F, K → canonical form)
- Confidence scoring
- Pydantic models for type-safe outputs
"""

import os
import json
from typing import Optional
from pydantic import BaseModel, Field

from google import genai
from google.genai import types as genai_types
from rich.console import Console

console = Console()


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a module-level singleton Gemini client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


# ---------------------------------------------------------------------------
# Pydantic models for structured extraction
# ---------------------------------------------------------------------------

class TemperatureValue(BaseModel):
    value: Optional[float] = Field(None, description="Numeric temperature value")
    unit: str = Field("°C", description="Unit: °C, °F, or K")
    value_celsius: Optional[float] = Field(None, description="Value normalized to Celsius")
    condition: Optional[str] = Field(None, description="e.g., 'at 1 atm', 'at 760 mmHg'")


class FlammabilityInfo(BaseModel):
    is_flammable: Optional[bool] = Field(None, description="Whether the substance is flammable")
    flash_point: Optional[TemperatureValue] = Field(None, description="Flash point temperature")
    flammability_class: Optional[str] = Field(None, description="e.g., Class IA, Class IB, IIIA")
    lower_explosive_limit: Optional[float] = Field(None, description="LEL as percentage")
    upper_explosive_limit: Optional[float] = Field(None, description="UEL as percentage")


class ChemicalProperties(BaseModel):
    compound_name: str = Field(..., description="Name of the chemical compound")
    cas_number: Optional[str] = Field(None, description="CAS Registry Number")
    boiling_point: Optional[TemperatureValue] = Field(None, description="Boiling point")
    melting_point: Optional[TemperatureValue] = Field(None, description="Melting/freezing point")
    flash_point: Optional[TemperatureValue] = Field(None, description="Flash point")
    flammability: Optional[FlammabilityInfo] = Field(None, description="Flammability details")
    molecular_weight: Optional[float] = Field(None, description="Molecular weight in g/mol")
    density: Optional[float] = Field(None, description="Density in g/cm³ at standard conditions")
    vapor_pressure: Optional[str] = Field(None, description="Vapor pressure with units and conditions")
    solubility: Optional[str] = Field(None, description="Water solubility description")
    confidence: float = Field(0.0, description="Extraction confidence 0.0-1.0")
    source_snippets: list[str] = Field(default_factory=list, description="Relevant text snippets used")


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a chemical data extraction specialist. Extract physical and chemical properties from technical literature and return EXACTLY this JSON schema — no extra fields, no nested confidence objects:

{
  "compound_name": "string",
  "cas_number": "string or null",
  "boiling_point": {"value": float, "unit": "°C", "value_celsius": float, "condition": "string or null"} or null,
  "melting_point": {"value": float, "unit": "°C", "value_celsius": float, "condition": "string or null"} or null,
  "flash_point":   {"value": float, "unit": "°C", "value_celsius": float, "condition": "string or null"} or null,
  "flammability": {
    "is_flammable": bool or null,
    "flash_point": {"value": float, "unit": "°C", "value_celsius": float, "condition": "string or null"} or null,
    "flammability_class": "string or null",
    "lower_explosive_limit": float or null,
    "upper_explosive_limit": float or null
  } or null,
  "molecular_weight": float or null,
  "density": float or null,
  "vapor_pressure": "string or null",
  "solubility": "string or null",
  "confidence": float (0.0-1.0),
  "source_snippets": ["exact text snippets used"]
}

Rules:
1. Only extract values explicitly stated in the provided text. Never guess or infer.
2. If a value is not present, set it to null.
3. All temperatures must use value_celsius (convert F→C or K→C if needed).
4. molecular_weight and density are plain floats, not objects.
5. vapor_pressure is a plain string like "5.95 kPa at 20 °C".
6. confidence: 0.9+ = explicit value found, 0.5-0.8 = implied, <0.5 = uncertain.
"""


def normalize_to_celsius(value: float, unit: str) -> float:
    """Normalize temperature to Celsius."""
    unit = unit.strip().upper().replace("°", "").replace("DEG", "")
    if unit in ("C", "°C", "CELSIUS"):
        return value
    elif unit in ("F", "°F", "FAHRENHEIT"):
        return round((value - 32) * 5 / 9, 2)
    elif unit in ("K", "KELVIN"):
        return round(value - 273.15, 2)
    return value


def extract_properties(
    compound_name: str,
    context_chunks: list[dict],
    model: str = "gpt-4o-mini",
) -> ChemicalProperties:
    """
    Extract chemical properties for a compound from retrieved context chunks.

    Args:
        compound_name: Target chemical name
        context_chunks: List of {text, source_file, page_number} dicts from retrieval
        model: OpenAI model to use

    Returns:
        ChemicalProperties with extracted values
    """
    if not context_chunks:
        console.print(f"[yellow]No context found for '{compound_name}'[/yellow]")
        return ChemicalProperties(compound_name=compound_name, confidence=0.0)

    # Build context string with source attribution
    context_parts = []
    for i, chunk in enumerate(context_chunks[:8]):  # Cap at 8 chunks to stay within token budget
        source = f"[Source: {chunk['source_file']}, p.{chunk['page_number']}]"
        context_parts.append(f"--- Chunk {i+1} {source} ---\n{chunk['text']}")

    context_str = "\n\n".join(context_parts)

    user_message = f"""Extract all available chemical properties for: **{compound_name}**

Context from technical documents:
{context_str}

Return structured JSON matching the ChemicalProperties schema. Only include values explicitly found in the context above."""

    try:
        response = _get_client().models.generate_content(
            model=model,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0,  # Deterministic for data extraction
            ),
        )

        raw_json = response.text
        data = json.loads(raw_json)

        # Ensure compound name is set
        data["compound_name"] = compound_name

        # Gemini sometimes wraps scalar fields in {"value": x, ...} dicts.
        # Flatten them back to their scalar equivalents before Pydantic validation.
        for scalar_field in ("molecular_weight", "density"):
            if isinstance(data.get(scalar_field), dict):
                data[scalar_field] = data[scalar_field].get("value")
        if isinstance(data.get("vapor_pressure"), dict):
            vp = data["vapor_pressure"]
            val = vp.get("value") or vp.get("pressure")
            unit = vp.get("unit", "")
            cond = vp.get("condition", "")
            parts = [str(val)] if val is not None else []
            if unit:
                parts.append(unit)
            if cond:
                parts.append(f"({cond})")
            data["vapor_pressure"] = " ".join(parts) if parts else None

        # Normalize temperatures to Celsius if not already done
        props = ChemicalProperties(**data)
        props = _normalize_temperatures(props)

        console.print(f"[green]Extracted properties for '{compound_name}' (confidence: {props.confidence})[/green]")
        return props

    except Exception as e:
        console.print(f"[red]Extraction failed for '{compound_name}': {e}[/red]")
        return ChemicalProperties(compound_name=compound_name, confidence=0.0)


def _normalize_temperatures(props: ChemicalProperties) -> ChemicalProperties:
    """Post-process: ensure all TemperatureValue fields have value_celsius set."""
    for field_name in ("boiling_point", "melting_point", "flash_point"):
        temp = getattr(props, field_name)
        if temp and temp.value is not None and temp.value_celsius is None:
            temp.value_celsius = normalize_to_celsius(temp.value, temp.unit)

    if props.flammability and props.flammability.flash_point:
        fp = props.flammability.flash_point
        if fp.value is not None and fp.value_celsius is None:
            fp.value_celsius = normalize_to_celsius(fp.value, fp.unit)

    return props


def format_extraction_report(props: ChemicalProperties) -> str:
    """Pretty-print extracted properties for CLI display."""
    lines = [
        f"\n[bold cyan]Chemical: {props.compound_name}[/bold cyan]",
        f"CAS: {props.cas_number or 'N/A'}",
        f"Confidence: {props.confidence:.0%}",
        "",
    ]

    def fmt_temp(t: Optional[TemperatureValue], label: str) -> str:
        if not t or t.value is None:
            return f"  {label}: Not found"
        c = f" ({t.value_celsius}°C)" if t.value_celsius is not None and t.unit != "°C" else ""
        cond = f" — {t.condition}" if t.condition else ""
        return f"  {label}: {t.value} {t.unit}{c}{cond}"

    lines.append("[bold]Physical Properties:[/bold]")
    lines.append(fmt_temp(props.boiling_point, "Boiling Point"))
    lines.append(fmt_temp(props.melting_point, "Melting Point"))
    lines.append(fmt_temp(props.flash_point, "Flash Point"))

    if props.molecular_weight:
        lines.append(f"  Molecular Weight: {props.molecular_weight} g/mol")
    if props.density:
        lines.append(f"  Density: {props.density} g/cm³")
    if props.vapor_pressure:
        lines.append(f"  Vapor Pressure: {props.vapor_pressure}")
    if props.solubility:
        lines.append(f"  Solubility: {props.solubility}")

    if props.flammability:
        lines.append("\n[bold]Flammability:[/bold]")
        f = props.flammability
        lines.append(f"  Flammable: {f.is_flammable}")
        lines.append(fmt_temp(f.flash_point, "Flash Point"))
        if f.flammability_class:
            lines.append(f"  Class: {f.flammability_class}")
        if f.lower_explosive_limit is not None:
            lines.append(f"  LEL: {f.lower_explosive_limit}%  UEL: {f.upper_explosive_limit}%")

    if props.source_snippets:
        lines.append(f"\n[dim]Source snippets: {len(props.source_snippets)} found[/dim]")

    return "\n".join(lines)
