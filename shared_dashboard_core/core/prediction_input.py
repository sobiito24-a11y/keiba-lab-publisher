from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .jra_predictor import predict_jra
from .models import PredictionResult, RaceMode
from .nar_predictor import predict_nar


@dataclass(frozen=True)
class PredictorHtmlInput:
    """Canonical boundary object shared by direct upload and batch prediction."""

    race_mode: RaceMode
    html_files: dict[str, str]
    file_names: dict[str, str]


def normalize_predictor_html_input(
    mode: RaceMode,
    html_files: Mapping[str, str],
    file_names: Mapping[str, str] | None = None,
) -> PredictorHtmlInput:
    """Copy and validate HTML maps without changing optional-kind semantics."""

    if mode not in {"jra", "nar"}:
        raise ValueError(f"未対応の競馬モードです: {mode}")
    normalized_html: dict[str, str] = {}
    normalized_names: dict[str, str] = {}
    names = file_names or {}
    for raw_kind, html_text in html_files.items():
        kind = str(raw_kind)
        if not isinstance(html_text, str):
            raise TypeError(f"{kind} HTMLは文字列で渡してください")
        normalized_html[kind] = html_text
        file_name = names.get(raw_kind, names.get(kind, ""))
        if file_name is not None and not isinstance(file_name, str):
            raise TypeError(f"{kind} のファイル名は文字列で渡してください")
        normalized_names[kind] = file_name or ""
    return PredictorHtmlInput(
        race_mode=mode,
        html_files=normalized_html,
        file_names=normalized_names,
    )


def predict_from_html_inputs(
    mode: RaceMode,
    html_files: Mapping[str, str],
    file_names: Mapping[str, str] | None = None,
    *,
    prediction_logic_version: str,
) -> PredictionResult:
    """Route both UI and Dashboard inputs through the canonical predictors."""

    predictor_input = normalize_predictor_html_input(mode, html_files, file_names)
    if predictor_input.race_mode == "nar":
        return predict_nar(
            predictor_input.html_files,
            predictor_input.file_names,
            prediction_logic_version=prediction_logic_version,
        )
    return predict_jra(
        predictor_input.html_files,
        predictor_input.file_names,
        prediction_logic_version=prediction_logic_version,
    )
