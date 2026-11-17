"""Self-contained copies of the two research case-study inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files

import numpy as np

from ._validation import FloatArray
from .models import DecisionMaker


@dataclass(frozen=True)
class CaseStudy:
    name: str
    alternatives: tuple[str, ...]
    criteria: tuple[str, ...]
    types: tuple[str, ...]
    acm: FloatArray
    decision_makers: tuple[DecisionMaker, ...]
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    integer_criteria: tuple[int, ...]
    provenance: str


def load_case_study(name: str) -> CaseStudy:
    """Load ``'production'`` or ``'storage'`` without running experiments.

    Returns fresh arrays each time. Bounds describe nonnegativity and the
    manuscript's percentage/rating scales, not calibrated uncertainty ranges.
    Decision-maker inputs are the illustrative profiles supplied in the project.
    """
    if name not in ("production", "storage"):
        raise ValueError("case study must be 'production' or 'storage'")
    resource = files("pysomit").joinpath("data", f"{name}.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    dms = []
    for dm in data["decision_makers"]:
        dm["comparisons"] = {int(k): v for k, v in dm["comparisons"].items()}
        dms.append(DecisionMaker(**dm))
    return CaseStudy(
        data["name"],
        tuple(data["alternatives"]),
        tuple(data["criteria"]),
        tuple(data["types"]),
        np.array(data["acm"], dtype=float),
        tuple(dms),
        np.array(data["lower_bounds"], dtype=float),
        np.array([np.inf if b is None else b for b in data["upper_bounds"]]),
        tuple(data["integer_criteria"]),
        data["provenance"],
    )
