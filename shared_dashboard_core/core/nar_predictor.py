from __future__ import annotations

from .models import PredictionResult
from .condition_fit import extract_condition_fit_sources
from .course_materials import attach_course_materials_to_result
from .nar_notebook_logic import apply_nar_newspaper_html_features, predict_nar_from_html
from .ver4_engine import apply_prediction_logic, prediction_logic_version as normalize_prediction_logic_version


def predict_nar(
    html_files: dict[str, str],
    file_names: dict[str, str] | None = None,
    *,
    prediction_logic_version: str = "v3",
) -> PredictionResult:
    result = predict_nar_from_html(html_files, file_names or {})
    if normalize_prediction_logic_version(prediction_logic_version) == "market":
        attach_course_materials_to_result(result, html_files)
        # JSON-shortcut input keeps the raw newspaper under a market-only key
        # so the legacy NAR prediction path remains unchanged.  Attach its
        # factual interval/class/body-weight evidence only to Market Compare.
        newspaper_context = html_files.get("newspaper_context", "")
        if newspaper_context and not html_files.get("newspaper"):
            enriched, race_info, _ = apply_nar_newspaper_html_features(
                result.overall_table,
                dict(result.race_info or {}),
                newspaper_context,
            )
            result.overall_table = enriched
            result.race_info = race_info
            debug = dict(result.debug_info or {})
            sources = dict(debug.get("condition_fit_sources") or {})
            for number, values in extract_condition_fit_sources(enriched).items():
                merged = dict(sources.get(number) or {})
                merged.update(values)
                sources[number] = merged
            debug["condition_fit_sources"] = sources
            result.debug_info = debug
    return apply_prediction_logic(result, prediction_logic_version)
