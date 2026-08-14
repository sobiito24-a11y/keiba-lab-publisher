from __future__ import annotations

from .models import PredictionResult
from .course_materials import attach_course_materials_to_result
from .jra_notebook_logic import predict_jra_from_html
from .ver4_engine import apply_prediction_logic, prediction_logic_version as normalize_prediction_logic_version


def predict_jra(
    html_files: dict[str, str],
    file_names: dict[str, str] | None = None,
    *,
    prediction_logic_version: str = "v3",
) -> PredictionResult:
    result = predict_jra_from_html(html_files, file_names or {})
    if normalize_prediction_logic_version(prediction_logic_version) == "market":
        attach_course_materials_to_result(result, html_files)
    return apply_prediction_logic(result, prediction_logic_version)
