"""Extract compact, sanitized evidence from the executed Colab notebooks.

The original timestamped small-data and active-learning result trees are not
available in this checkout. The notebooks do retain executed summaries and
Pandas HTML tables. This script extracts only allowlisted numeric fields from
those outputs; it never copies raw stdout, prompts, tracebacks, or cell source
into the generated artifacts.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from html.parser import HTMLParser
from itertools import product
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMALL_NOTEBOOK = (
    PROJECT_ROOT / "colab notebooks" / "Learning curve_data efficiency.ipynb"
)
DEFAULT_ACTIVE_NOTEBOOK = PROJECT_ROOT / "colab notebooks" / "Active learning.ipynb"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "diagnostics"

SMALL_RUN_CELL_ID = "247mi-kb7PcA"
SMALL_LATEST_CELL_ID = "sRfWUGK3-6Fh"
SMALL_BASE_SUMMARY_CELL_ID = "HkUw4vKz_D5_"
SMALL_GIT_PULL_CELL_ID = "FpdNv2RiBmjW"
SMALL_POSTPROCESS_CELL_ID = "L-zOELyjBseC"
SMALL_EXTENDED_SUMMARY_CELL_ID = "WE6QKEmmBypF"
ACTIVE_STANDARD_TABLE_CELL_ID = "9kMHymJxff3-"
ACTIVE_STRESS_TABLE_CELL_ID = "g3xYGPVyyAPN"
ACTIVE_COMBINED_HEAD_CELL_ID = "spgzrhvcfc-U"
ACTIVE_LATEST_FULL_TABLE_CELL_ID = "CfEAZSvEyF_M"

ACTIVE_STANDARD_RUN_CELLS = (
    (
        "XYAOY18uqyqd",
        "matbench_jdft2d",
        "magpie",
        "gpu_20260606_160326_utc",
    ),
    (
        "7besjW9uq9Ce",
        "matbench_jdft2d",
        "magpie_structure_all",
        "gpu_20260606_164604_utc",
    ),
    (
        "ErX9iMUAq9s9",
        "matbench_phonons",
        "magpie",
        "gpu_20260606_173042_utc",
    ),
    (
        "bYshnfgzrAc9",
        "matbench_phonons",
        "magpie_structure_all",
        "gpu_20260606_183904_utc",
    ),
)
ACTIVE_STRESS_RUN_CELL = "ObY4k77Hqchv"

SMALL_RUN_ID = "gpu_20260605_223535_utc"
SMALL_RUNNER_COMMIT = "cfd0314fffa891360c2da77c9dbb4dd6a80f4e2d"
SMALL_RUNNER_BLOB = "8e32410d2959f8012bc3f45aaa243f15591afdc4"
SMALL_POSTPROCESS_COMMIT = "443f218ceb79b1e35f7c9bac37f2baedfaef2ad4"
SMALL_POSTPROCESS_BLOB = "e8bd2409f0a7b547e82e5894dcf2cb543035c53e"
ACTIVE_RUNNER_COMMIT = "7cf3fc1c61487b241e16dfa70666786e70cbd1af"
ACTIVE_RUNNER_BLOB = "0731324ffebcdae89124b62881400b9bd790cec8"
ACTIVE_HELPER_BLOBS = {
    "src/matbench_tabpfn/features.py": "7e096aa7359abcb0bb0a99af24cb7c73cba982e7",
    "src/matbench_tabpfn/models.py": "c7e0cf4a6892b9116c0a12561020b80c4642c86c",
    "src/matbench_tabpfn/paths.py": "0b89c78797405d7c9675b5837961175fb40ff6a9",
}

TASK_UNITS = {
    "matbench_jdft2d": "meV/atom",
    "matbench_phonons": "cm^-1",
}
ACTIVE_STANDARD_STRATEGIES = (
    "random",
    "extra_trees_greedy",
    "extra_trees_ucb",
    "tabpfn_greedy",
    "tabpfn_disagreement_ucb",
)
ACTIVE_STRESS_STRATEGIES = (
    "random",
    "extra_trees_greedy",
    "tabpfn_greedy",
    "tabpfn_disagreement_ucb",
)
ACTIVE_SHARED_DEFAULT_OPTIONS = (
    "--random-seed",
    "--objective",
    "--initial-selection",
    "--max-acquisition-fraction",
    "--ucb-beta",
    "--n-estimators",
    "--tabpfn-n-estimators",
    "--tabpfn-predict-batch-size",
    "--feature-n-jobs",
    "--no-feature-cache",
    "--show-feature-progress",
    "--include-oracle",
)

FEATURE_DISPLAY_TO_KEY = {
    "Magpie composition": "magpie",
    "Magpie + all structure": "magpie_structure_all",
}
MODEL_DISPLAY_TO_KEY = {
    "Extra trees": "extra_trees",
    "TabPFN": "tabpfn",
}
ACTIVE_STRATEGY_ORDER = {
    "random": 0,
    "extra_trees_greedy": 1,
    "extra_trees_ucb": 2,
    "tabpfn_greedy": 3,
    "tabpfn_disagreement_ucb": 4,
}
TASK_ORDER = {"matbench_jdft2d": 0, "matbench_phonons": 1}
FEATURE_ORDER = {"magpie": 0, "magpie_structure_all": 1}

SMALL_FIELDS = (
    "claim_id",
    "claim_scope",
    "task",
    "model",
    "feature_set",
    "metric",
    "condition",
    "value",
    "unit",
    "run_id",
    "notebook_cell_id",
    "value_origin",
)

ACTIVE_FIELDS = (
    "variant",
    "source_run_id",
    "task",
    "unit",
    "feature_set",
    "feature_set_display",
    "strategy",
    "initial_fraction",
    "top_fraction",
    "n_runs",
    "found_global_best_rate",
    "acquisitions_to_global_best_mean",
    "acquisitions_to_global_best_median",
    "first_top_fraction_acquisitions_mean",
    "final_objective_gap_mean",
    "final_objective_regret_fraction_mean",
    "final_best_objective_percentile_mean",
    "final_top_fraction_hit_rate_mean",
    "final_top_fraction_hit_count_mean",
    "fit_seconds_sum",
    "predict_seconds_sum",
    "notebook_cell_id",
    "value_origin",
)

SECRET_PATTERN = re.compile(r"tabpfn_sk__[A-Za-z0-9_-]+")


class _HtmlTableParser(HTMLParser):
    """Collect simple HTML table rows while ignoring surrounding Colab markup."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag == "table" and self._table is None:
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell_parts is not None:
            assert self._row is not None
            self._row.append("".join(self._cell_parts).strip())
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            assert self._table is not None
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sanitized compact evidence from executed Colab notebooks."
    )
    parser.add_argument(
        "--small-notebook",
        type=Path,
        default=DEFAULT_SMALL_NOTEBOOK,
    )
    parser.add_argument(
        "--active-notebook",
        type=Path,
        default=DEFAULT_ACTIVE_NOTEBOOK,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in memory and fail if committed artifacts differ.",
    )
    return parser.parse_args()


def _load_notebook(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    if not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{path} does not contain a notebook cell list.")
    return notebook


def _cell_by_id(notebook: dict[str, object], cell_id: str) -> dict[str, object]:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    matches = [
        cell
        for cell in cells
        if isinstance(cell, dict)
        and isinstance(cell.get("metadata"), dict)
        and cell["metadata"].get("id") == cell_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one notebook cell {cell_id!r}, found {len(matches)}.")
    cell = matches[0]
    metadata = cell.get("metadata", {})
    execution_info = metadata.get("executionInfo", {}) if isinstance(metadata, dict) else {}
    status = (
        execution_info.get("status") if isinstance(execution_info, dict) else None
    )
    if status != "ok":
        raise ValueError(f"Notebook cell {cell_id!r} has status {status!r}, not 'ok'.")
    outputs = cell.get("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError(f"Notebook cell {cell_id!r} has malformed outputs.")
    if any(
        isinstance(output, dict) and output.get("output_type") == "error"
        for output in outputs
    ):
        raise ValueError(f"Notebook cell {cell_id!r} contains an error output.")
    return cell


def _execution_timestamp(cell: dict[str, object], *, cell_id: str) -> int:
    metadata = cell.get("metadata", {})
    execution_info = metadata.get("executionInfo", {}) if isinstance(metadata, dict) else {}
    timestamp = (
        execution_info.get("timestamp")
        if isinstance(execution_info, dict)
        else None
    )
    if not isinstance(timestamp, int):
        raise ValueError(f"Notebook cell {cell_id!r} has no integer execution timestamp.")
    return timestamp


def _join_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value or "")


def _source(cell: dict[str, object]) -> str:
    return _join_text(cell.get("source"))


def _stream_text(cell: dict[str, object]) -> str:
    parts: list[str] = []
    outputs = cell.get("outputs", [])
    assert isinstance(outputs, list)
    for output in outputs:
        if isinstance(output, dict) and output.get("output_type") == "stream":
            parts.append(_join_text(output.get("text")))
    return "\n".join(parts)


def _plain_text(cell: dict[str, object]) -> str:
    parts: list[str] = []
    outputs = cell.get("outputs", [])
    assert isinstance(outputs, list)
    for output in outputs:
        if not isinstance(output, dict):
            continue
        data = output.get("data", {})
        if isinstance(data, dict) and "text/plain" in data:
            parts.append(_join_text(data["text/plain"]))
        elif output.get("output_type") == "stream":
            parts.append(_join_text(output.get("text")))
    return "\n".join(parts)


def _html_tables(cell: dict[str, object]) -> list[list[dict[str, str]]]:
    parsed_tables: list[list[dict[str, str]]] = []
    outputs = cell.get("outputs", [])
    assert isinstance(outputs, list)
    for output in outputs:
        if not isinstance(output, dict):
            continue
        data = output.get("data", {})
        if not isinstance(data, dict) or "text/html" not in data:
            continue
        parser = _HtmlTableParser()
        parser.feed(_join_text(data["text/html"]))
        for table in parser.tables:
            if len(table) < 2:
                continue
            headers = table[0]
            if headers and headers[0] == "":
                headers = headers[1:]
            records: list[dict[str, str]] = []
            for raw_row in table[1:]:
                row = raw_row
                if len(row) == len(headers) + 1:
                    row = row[1:]
                if len(row) != len(headers):
                    raise ValueError(
                        "HTML table row length does not match its headers: "
                        f"{len(row)} != {len(headers)}."
                    )
                records.append(dict(zip(headers, row, strict=True)))
            parsed_tables.append(records)
    return parsed_tables


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _execution_users(notebook: dict[str, object]) -> list[str]:
    users: set[str] = set()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        metadata = cell.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        execution_info = metadata.get("executionInfo", {})
        if not isinstance(execution_info, dict):
            continue
        user = execution_info.get("user", {})
        if isinstance(user, dict) and user.get("displayName"):
            users.add(str(user["displayName"]))
    return sorted(users)


def _one_match(pattern: str, text: str, *, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"Expected one {label}, found {len(matches)}.")
    return matches[0]


def _assert_exact_grid(
    actual: Iterable[tuple[object, ...]],
    expected: Iterable[tuple[object, ...]],
    *,
    label: str,
) -> int:
    actual_counter = Counter(actual)
    expected_counter = Counter(expected)
    if actual_counter != expected_counter:
        missing = list((expected_counter - actual_counter).elements())[:3]
        unexpected = list((actual_counter - expected_counter).elements())[:3]
        raise ValueError(
            f"{label} does not match its exact Cartesian grid; "
            f"missing={missing}, unexpected_or_duplicate={unexpected}."
        )
    return sum(actual_counter.values())


def _code_revision_record(
    relative_path: str,
    *,
    executed_commit: str,
    executed_blob: str,
    evidence: str,
    certainty: str = "direct_notebook_lineage",
) -> dict[str, object]:
    current_blob = _git_blob_sha1(PROJECT_ROOT / relative_path)
    return {
        "path": relative_path,
        "executed_commit": executed_commit,
        "executed_blob": executed_blob,
        "current_checkout_blob": current_blob,
        "current_checkout_matches_executed_blob": current_blob == executed_blob,
        "evidence": evidence,
        "certainty": certainty,
    }


def _add_small_row(
    rows: list[dict[str, str]],
    *,
    claim_scope: str,
    task: str,
    model: str,
    feature_set: str,
    metric: str,
    condition: str,
    value: str,
    unit: str,
    run_id: str,
    cell_id: str,
) -> None:
    if unit == "mae":
        unit = TASK_UNITS[task]
    elif unit == "percent":
        unit = "%"
    elif unit == "correlation":
        unit = "dimensionless"
    claim_id = "_".join(
        part
        for part in (
            claim_scope,
            task,
            model,
            feature_set,
            metric,
            condition,
        )
        if part
    )
    rows.append(
        {
            "claim_id": claim_id,
            "claim_scope": claim_scope,
            "task": task,
            "model": model,
            "feature_set": feature_set,
            "metric": metric,
            "condition": condition,
            "value": value,
            "unit": unit,
            "run_id": run_id,
            "notebook_cell_id": cell_id,
            "value_origin": "notebook_display_text",
        }
    )


def _extract_small_data(
    notebook: dict[str, object],
    notebook_path: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    run_cell = _cell_by_id(notebook, SMALL_RUN_CELL_ID)
    run_source = _source(run_cell)
    required_source_fragments = (
        "run_small_data_diagnostics.py",
        "--models extra_trees tabpfn",
        "--fractions 0.1 0.2 0.4 0.6 0.8 1.0",
        "--repeats 1",
        "--device cuda",
    )
    for fragment in required_source_fragments:
        if fragment not in run_source:
            raise ValueError(f"Small-data run cell is missing {fragment!r}.")

    run_output = _stream_text(run_cell)
    run_match = _one_match(
        r"Run root: .*/(gpu_\d{8}_\d{6}_utc)",
        run_output,
        label="canonical small-data run ID",
    )
    run_id = run_match.group(1)
    if run_id != SMALL_RUN_ID:
        raise ValueError(f"Unexpected canonical small-data run ID: {run_id}.")

    condition_pattern = re.compile(
        r"^(?P<task>matbench_(?:jdft2d|phonons)) \| "
        r"(?P<feature>magpie|magpie_structure_all) \| "
        r"(?P<model>extra_trees|tabpfn) \| fold (?P<fold>\d) \| fraction "
        r"(?P<fraction>0\.10|0\.20|0\.40|0\.60|0\.80|1\.00) \| "
        r"repeat (?P<repeat>0)$",
        flags=re.MULTILINE,
    )
    actual_conditions = [
        (
            match.group("task"),
            match.group("feature"),
            match.group("model"),
            int(match.group("fold")),
            match.group("fraction"),
            int(match.group("repeat")),
        )
        for match in condition_pattern.finditer(run_output)
    ]
    expected_conditions = product(
        ("matbench_jdft2d", "matbench_phonons"),
        ("magpie", "magpie_structure_all"),
        ("extra_trees", "tabpfn"),
        range(5),
        ("0.10", "0.20", "0.40", "0.60", "0.80", "1.00"),
        (0,),
    )
    condition_count = _assert_exact_grid(
        actual_conditions,
        expected_conditions,
        label="Small-data condition log",
    )

    latest_cell = _cell_by_id(notebook, SMALL_LATEST_CELL_ID)
    if "glob.glob(\"results/small_data_diagnostics/*\")" not in _source(latest_cell):
        raise ValueError("Small-data latest-run selector source anchor is missing.")
    expected_latest = f"results/small_data_diagnostics/{run_id}"
    if _stream_text(latest_cell).strip() != expected_latest:
        raise ValueError("Small-data latest-run selector does not identify the canonical run.")

    base_cell = _cell_by_id(notebook, SMALL_BASE_SUMMARY_CELL_ID)
    if _source(base_cell).strip() != "!cat {latest}/tables/small_data_diagnostic_summary.md":
        raise ValueError("Base small-data summary is not read from the selected run.")
    base_text = _plain_text(base_cell)
    if "# Small-Data Diagnostic Summary" not in base_text:
        raise ValueError("Base small-data summary marker is missing.")
    if base_text.strip() not in run_output:
        raise ValueError("Base small-data summary is not verbatim in the canonical run output.")

    pull_cell = _cell_by_id(notebook, SMALL_GIT_PULL_CELL_ID)
    if _source(pull_cell).strip() != "!git pull":
        raise ValueError("Small-data revision-update source anchor is missing.")
    pull_output = _stream_text(pull_cell)
    if "Updating cfd0314..443f218" not in pull_output:
        raise ValueError("Small-data runner/postprocessor revision transition is missing.")

    postprocess_cell = _cell_by_id(notebook, SMALL_POSTPROCESS_CELL_ID)
    if _source(postprocess_cell).strip() != (
        "!python scripts/summarize_small_data_diagnostics.py"
    ):
        raise ValueError("Small-data postprocessor source anchor is missing.")
    postprocess_output = _stream_text(postprocess_cell)
    expected_updated = (
        "Updated run folder: "
        f"/content/matbench-tabpfn/results/small_data_diagnostics/{run_id}"
    )
    if expected_updated not in postprocess_output:
        raise ValueError("Small-data postprocessor does not identify the canonical run.")

    extended_cell = _cell_by_id(notebook, SMALL_EXTENDED_SUMMARY_CELL_ID)
    if _source(extended_cell).strip() != (
        "!cat {latest}/tables/extended_small_data_diagnostic_summary.md"
    ):
        raise ValueError("Extended small-data summary is not read from the selected run.")
    extended_text = _plain_text(extended_cell)
    if "# Extended Small-Data Diagnostic Summary" not in extended_text:
        raise ValueError("Extended small-data summary marker is missing.")
    if extended_text.strip() not in postprocess_output:
        raise ValueError(
            "Extended small-data summary is not verbatim in the canonical-run "
            "postprocessor output."
        )

    lineage_cells = (
        (SMALL_RUN_CELL_ID, run_cell),
        (SMALL_LATEST_CELL_ID, latest_cell),
        (SMALL_BASE_SUMMARY_CELL_ID, base_cell),
        (SMALL_GIT_PULL_CELL_ID, pull_cell),
        (SMALL_POSTPROCESS_CELL_ID, postprocess_cell),
        (SMALL_EXTENDED_SUMMARY_CELL_ID, extended_cell),
    )
    execution_timestamps = {
        cell_id: _execution_timestamp(cell, cell_id=cell_id)
        for cell_id, cell in lineage_cells
    }
    if list(execution_timestamps.values()) != sorted(execution_timestamps.values()):
        raise ValueError("Small-data evidence cells are not in execution-time order.")

    rows: list[dict[str, str]] = []

    best_pattern = re.compile(
        r"- `(?P<task>matbench_\w+)`: best full-train diagnostic model is "
        r"(?P<model>Extra trees|TabPFN) with "
        r"(?P<feature>Magpie composition|Magpie \+ all structure), "
        r"MAE=(?P<value>-?\d+(?:\.\d+)?)\."
    )
    best_matches = list(best_pattern.finditer(base_text))
    if len(best_matches) != 2:
        raise ValueError(f"Expected 2 best full-train rows, found {len(best_matches)}.")
    for match in best_matches:
        _add_small_row(
            rows,
            claim_scope="full_train",
            task=match.group("task"),
            model=MODEL_DISPLAY_TO_KEY[match.group("model")],
            feature_set=FEATURE_DISPLAY_TO_KEY[match.group("feature")],
            metric="best_model_mae",
            condition="official_folds",
            value=match.group("value"),
            unit="mae",
            run_id=run_id,
            cell_id=SMALL_BASE_SUMMARY_CELL_ID,
        )

    representation_pattern = re.compile(
        r"- `(?P<task>matbench_\w+)` / `(?P<model>extra_trees|tabpfn)`: "
        r"all-structure improves over composition proxy by "
        r"(?P<value>-?\d+(?:\.\d+)?)% MAE\."
    )
    representation_matches = list(representation_pattern.finditer(base_text))
    if len(representation_matches) != 4:
        raise ValueError(
            "Expected 4 full-train representation rows, found "
            f"{len(representation_matches)}."
        )
    for match in representation_matches:
        _add_small_row(
            rows,
            claim_scope="full_train",
            task=match.group("task"),
            model=match.group("model"),
            feature_set="structure_minus_composition",
            metric="mae_pct_change",
            condition="official_folds",
            value=match.group("value"),
            unit="percent",
            run_id=run_id,
            cell_id=SMALL_BASE_SUMMARY_CELL_ID,
        )

    hardest_pattern = re.compile(
        r"- `(?P<task>matbench_\w+)`: hardest full-train target regime "
        r"in this diagnostic is `high` for Extra trees with Magpie composition, "
        r"MAE=(?P<value>-?\d+(?:\.\d+)?)\."
    )
    hardest_matches = list(hardest_pattern.finditer(base_text))
    if len(hardest_matches) != 2:
        raise ValueError(f"Expected 2 hardest-regime rows, found {len(hardest_matches)}.")
    for match in hardest_matches:
        _add_small_row(
            rows,
            claim_scope="target_regime",
            task=match.group("task"),
            model="extra_trees",
            feature_set="magpie",
            metric="mae",
            condition="high_target_full_train",
            value=match.group("value"),
            unit="mae",
            run_id=run_id,
            cell_id=SMALL_BASE_SUMMARY_CELL_ID,
        )

    aulc_pattern = re.compile(
        r"- `(?P<task>matbench_\w+)`: lowest AULC is "
        r"(?P<model>Extra trees|TabPFN) with "
        r"(?P<feature>Magpie composition|Magpie \+ all structure), "
        r"AULC=(?P<value>-?\d+(?:\.\d+)?)\."
    )
    aulc_matches = list(aulc_pattern.finditer(extended_text))
    if len(aulc_matches) != 2:
        raise ValueError(f"Expected 2 AULC winner rows, found {len(aulc_matches)}.")
    for match in aulc_matches:
        _add_small_row(
            rows,
            claim_scope="data_efficiency",
            task=match.group("task"),
            model=MODEL_DISPLAY_TO_KEY[match.group("model")],
            feature_set=FEATURE_DISPLAY_TO_KEY[match.group("feature")],
            metric="aulc_mean_mae",
            condition="fractions_0.1_to_1.0",
            value=match.group("value"),
            unit="mae",
            run_id=run_id,
            cell_id=SMALL_EXTENDED_SUMMARY_CELL_ID,
        )

    high_pattern = re.compile(
        r"- `(?P<task>matbench_\w+)` / "
        r"(?P<model>Extra trees|TabPFN): all-structure reduces "
        r"high-regime MAE by (?P<value>-?\d+(?:\.\d+)?)%\."
    )
    high_matches = list(high_pattern.finditer(extended_text))
    if len(high_matches) != 4:
        raise ValueError(
            f"Expected 4 high-target reduction rows, found {len(high_matches)}."
        )
    for match in high_matches:
        _add_small_row(
            rows,
            claim_scope="target_regime",
            task=match.group("task"),
            model=MODEL_DISPLAY_TO_KEY[match.group("model")],
            feature_set="structure_minus_composition",
            metric="mae_pct_change",
            condition="high_target",
            value=match.group("value"),
            unit="percent",
            run_id=run_id,
            cell_id=SMALL_EXTENDED_SUMMARY_CELL_ID,
        )

    disagreement_pattern = re.compile(
        r"- `(?P<task>matbench_\w+)` / "
        r"(?P<feature>Magpie composition|Magpie \+ all structure): "
        r"Spearman\(disagreement, TabPFN error\)="
        r"(?P<spearman>-?\d+(?:\.\d+)?); "
        r"high-disagreement TabPFN MAE is "
        r"(?P<high>-?\d+(?:\.\d+)?) vs "
        r"(?P<lower>-?\d+(?:\.\d+)?) for lower-disagreement samples\."
    )
    disagreement_matches = list(disagreement_pattern.finditer(extended_text))
    if len(disagreement_matches) != 4:
        raise ValueError(
            "Expected 4 disagreement/error-proxy rows, found "
            f"{len(disagreement_matches)}."
        )
    for match in disagreement_matches:
        common = {
            "claim_scope": "disagreement_error_proxy",
            "task": match.group("task"),
            "model": "tabpfn",
            "feature_set": FEATURE_DISPLAY_TO_KEY[match.group("feature")],
            "run_id": run_id,
            "cell_id": SMALL_EXTENDED_SUMMARY_CELL_ID,
        }
        _add_small_row(
            rows,
            metric="spearman_disagreement_error",
            condition="all_test_samples",
            value=match.group("spearman"),
            unit="correlation",
            **common,
        )
        _add_small_row(
            rows,
            metric="mae",
            condition="high_disagreement",
            value=match.group("high"),
            unit="mae",
            **common,
        )
        _add_small_row(
            rows,
            metric="mae",
            condition="lower_disagreement",
            value=match.group("lower"),
            unit="mae",
            **common,
        )

    if len(rows) != 26:
        raise ValueError(f"Expected 26 small-data evidence rows, found {len(rows)}.")
    if len({row["claim_id"] for row in rows}) != len(rows):
        raise ValueError("Small-data claim IDs are not unique.")

    provenance: dict[str, object] = {
        "schema_version": 1,
        "source_notebook": _display_path(notebook_path),
        "source_notebook_sha256": _sha256(notebook_path),
        "source_notebook_git_blob": _git_blob_sha1(notebook_path),
        "execution_users": _execution_users(notebook),
        "canonical_run_id": run_id,
        "successful_cell_condition_count": condition_count,
        "condition_grid_exact": True,
        "evidence_cells": {
            "run": SMALL_RUN_CELL_ID,
            "latest_selector": SMALL_LATEST_CELL_ID,
            "base_summary": SMALL_BASE_SUMMARY_CELL_ID,
            "revision_update": SMALL_GIT_PULL_CELL_ID,
            "postprocessor": SMALL_POSTPROCESS_CELL_ID,
            "extended_summary": SMALL_EXTENDED_SUMMARY_CELL_ID,
        },
        "execution_timestamps": execution_timestamps,
        "row_count": len(rows),
        "field_origins": {
            "run_id": (
                "Final run-root line, latest-selector output, and postprocessor "
                "updated-run line all identify the same run."
            ),
            "values": "Allowlisted base and extended summary display text.",
            "units": "Matbench task metadata used by the executed runner.",
        },
        "aggregation_semantics": {
            "learning_curve_mae": (
                "Unweighted arithmetic mean of the five official-fold MAEs."
            ),
            "aulc": (
                "Normalized trapezoidal integral of fold-mean MAE at fractions "
                "0.1, 0.2, 0.4, 0.6, 0.8, and 1.0, divided by 0.9."
            ),
            "target_regimes": (
                "Three pd.qcut target bins computed separately within each "
                "official test fold."
            ),
            "target_regime_mae": (
                "Unweighted arithmetic mean of the five fold-level regime MAEs."
            ),
            "high_target_change": (
                "Signed (structure - composition) / composition * 100 at full train."
            ),
            "disagreement": (
                "Full-train predictions paired by task, feature, fold, repeat, "
                "and material ID; high disagreement is >= the per-task/feature "
                "0.75 quantile."
            ),
        },
        "executed_code": {
            "runner": _code_revision_record(
                "scripts/run_small_data_diagnostics.py",
                executed_commit=SMALL_RUNNER_COMMIT,
                executed_blob=SMALL_RUNNER_BLOB,
                evidence=(
                    "The later git-pull output starts at cfd0314, after the model "
                    "run and before adding the postprocessor."
                ),
            ),
            "postprocessor": _code_revision_record(
                "scripts/summarize_small_data_diagnostics.py",
                executed_commit=SMALL_POSTPROCESS_COMMIT,
                executed_blob=SMALL_POSTPROCESS_BLOB,
                evidence="The executed git-pull output updates cfd0314..443f218.",
            ),
        },
        "extraction_precision": "Values are preserved exactly as displayed.",
        "generator": "scripts/extract_colab_notebook_evidence.py",
        "redaction_policy": (
            "Only allowlisted numeric summary fields are emitted; raw stdout "
            "and interactive prompts are never copied."
        ),
        "limitations": [
            "The original timestamped result tree is unavailable.",
            "Only two of eight AULC values are printed numerically in the notebook.",
            "The complete learning-curve and target-regime CSV tables cannot be reconstructed.",
        ],
    }
    return rows, provenance


def _parse_cli_values(source: str, option: str) -> list[str]:
    match = _one_match(
        rf"{re.escape(option)}\s+([^\\\n]+)",
        source,
        label=f"{option} option",
    )
    return match.group(1).strip().split()


def _require_cli_values(source: str, option: str, expected: list[str]) -> None:
    actual = _parse_cli_values(source, option)
    if actual != expected:
        raise ValueError(
            f"Unexpected {option} values: expected {expected}, found {actual}."
        )


def _require_cli_options_absent(
    source: str,
    options: Iterable[str],
    *,
    cell_id: str,
) -> None:
    present = [
        option
        for option in options
        if re.search(rf"(?<![\w-]){re.escape(option)}(?:\s|$)", source)
    ]
    if present:
        raise ValueError(
            f"Active-learning cell {cell_id} unexpectedly overrides defaults: "
            f"{present}."
        )


def _completion_run_id(
    output: str,
    *,
    expected_conditions: Iterable[tuple[object, ...]],
) -> tuple[str, int]:
    match = _one_match(
        r"Active-learning screening complete: .*/(gpu_\d{8}_\d{6}_utc)",
        output,
        label="active-learning completion run ID",
    )
    condition_pattern = re.compile(
        r"^(?P<task>matbench_(?:jdft2d|phonons)) \| "
        r"(?P<feature>magpie|magpie_structure_all) \| "
        r"(?P<strategy>random|extra_trees_greedy|extra_trees_ucb|"
        r"tabpfn_greedy|tabpfn_disagreement_ucb) \| "
        r"fold (?P<fold>\d) \| init (?P<initial>\d+)% \| "
        r"repeat (?P<repeat>\d+)$",
        flags=re.MULTILINE,
    )
    actual_conditions = [
        (
            condition.group("task"),
            condition.group("feature"),
            condition.group("strategy"),
            int(condition.group("fold")),
            int(condition.group("initial")),
            int(condition.group("repeat")),
        )
        for condition in condition_pattern.finditer(output)
    ]
    condition_count = _assert_exact_grid(
        actual_conditions,
        expected_conditions,
        label="Active-learning condition log",
    )
    return match.group(1), condition_count


def _single_table_with_columns(
    cell: dict[str, object],
    required_columns: set[str],
    expected_rows: int,
) -> list[dict[str, str]]:
    matches = [
        table
        for table in _html_tables(cell)
        if table and required_columns.issubset(table[0])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one HTML table with {sorted(required_columns)}, "
            f"found {len(matches)}."
        )
    table = matches[0]
    if len(table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} HTML table rows, found {len(table)}."
        )
    return table


def _extract_active_learning(
    notebook: dict[str, object],
    notebook_path: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    run_map: dict[tuple[str, str], str] = {}
    successful_cell_condition_counts: dict[str, int] = {}
    for (
        cell_id,
        expected_task,
        expected_feature,
        expected_run_id,
    ) in ACTIVE_STANDARD_RUN_CELLS:
        cell = _cell_by_id(notebook, cell_id)
        source = _source(cell)
        _require_cli_values(source, "--tasks", [expected_task])
        _require_cli_values(source, "--feature-sets", [expected_feature])
        _require_cli_values(
            source,
            "--strategies",
            list(ACTIVE_STANDARD_STRATEGIES),
        )
        _require_cli_values(source, "--initial-fractions", ["0.1", "0.2"])
        _require_cli_values(source, "--max-acquisitions", ["100"])
        _require_cli_values(source, "--acquisition-batch-size", ["10"])
        _require_cli_values(source, "--repeats", ["3"])
        _require_cli_values(source, "--device", ["cuda"])
        _require_cli_options_absent(
            source,
            (*ACTIVE_SHARED_DEFAULT_OPTIONS, "--folds", "--top-fraction"),
            cell_id=cell_id,
        )
        output = _stream_text(cell)
        run_id, condition_count = _completion_run_id(
            output,
            expected_conditions=product(
                (expected_task,),
                (expected_feature,),
                ACTIVE_STANDARD_STRATEGIES,
                range(5),
                (10, 20),
                range(3),
            ),
        )
        if run_id != expected_run_id:
            raise ValueError(
                f"Unexpected active-learning run ID in {cell_id}: {run_id}."
            )
        run_map[(expected_task, expected_feature)] = run_id
        successful_cell_condition_counts[run_id] = condition_count

    expected_run_map = {
        (task, feature): run_id
        for _, task, feature, run_id in ACTIVE_STANDARD_RUN_CELLS
    }
    if run_map != expected_run_map:
        raise ValueError(f"Unexpected standard active-learning run map: {run_map}.")

    stress_cell = _cell_by_id(notebook, ACTIVE_STRESS_RUN_CELL)
    stress_source = _source(stress_cell)
    _require_cli_values(
        stress_source,
        "--tasks",
        ["matbench_jdft2d", "matbench_phonons"],
    )
    _require_cli_values(
        stress_source,
        "--feature-sets",
        ["magpie", "magpie_structure_all"],
    )
    _require_cli_values(
        stress_source,
        "--strategies",
        list(ACTIVE_STRESS_STRATEGIES),
    )
    _require_cli_values(stress_source, "--folds", ["0", "1"])
    _require_cli_values(stress_source, "--initial-fractions", ["0.05"])
    _require_cli_values(stress_source, "--top-fraction", ["0.01"])
    _require_cli_values(stress_source, "--max-acquisitions", ["100"])
    _require_cli_values(stress_source, "--acquisition-batch-size", ["10"])
    _require_cli_values(stress_source, "--repeats", ["1"])
    _require_cli_values(stress_source, "--device", ["cuda"])
    _require_cli_options_absent(
        stress_source,
        ACTIVE_SHARED_DEFAULT_OPTIONS,
        cell_id=ACTIVE_STRESS_RUN_CELL,
    )
    stress_run_id, stress_condition_count = _completion_run_id(
        _stream_text(stress_cell),
        expected_conditions=product(
            ("matbench_jdft2d", "matbench_phonons"),
            ("magpie", "magpie_structure_all"),
            ACTIVE_STRESS_STRATEGIES,
            (0, 1),
            (5,),
            (0,),
        ),
    )
    if stress_run_id != "gpu_20260606_204419_utc":
        raise ValueError(f"Unexpected stress-test run ID: {stress_run_id}.")
    successful_cell_condition_counts[stress_run_id] = stress_condition_count

    standard_table_cell = _cell_by_id(notebook, ACTIVE_STANDARD_TABLE_CELL_ID)
    if "ranking = combined_agg.sort_values" not in _source(standard_table_cell):
        raise ValueError("Active-learning standard table source anchor is missing.")
    standard_rows = _single_table_with_columns(
        standard_table_cell,
        {
            "task",
            "feature_set_display",
            "strategy",
            "initial_fraction",
            "final_objective_regret_fraction_mean",
            "final_top_fraction_hit_rate_mean",
            "final_best_objective_percentile_mean",
        },
        expected_rows=40,
    )
    standard_key_count = _assert_exact_grid(
        (
            (
                row["task"],
                FEATURE_DISPLAY_TO_KEY[row["feature_set_display"]],
                row["strategy"],
                row["initial_fraction"],
            )
            for row in standard_rows
        ),
        product(
            ("matbench_jdft2d", "matbench_phonons"),
            ("magpie", "magpie_structure_all"),
            ACTIVE_STANDARD_STRATEGIES,
            ("0.1", "0.2"),
        ),
        label="Active-learning standard display table",
    )

    combined_head_cell = _cell_by_id(notebook, ACTIVE_COMBINED_HEAD_CELL_ID)
    if "display(combined_agg.head())" not in _source(combined_head_cell):
        raise ValueError("Active-learning combined-head source anchor is missing.")
    if "combined_agg: (40, 19)" not in _stream_text(combined_head_cell):
        raise ValueError("Active-learning combined aggregate shape is not 40 by 19.")

    full_metric_columns = {
        "task",
        "unit",
        "feature_set",
        "feature_set_display",
        "strategy",
        "initial_fraction",
        "n_runs",
        "found_global_best_rate",
        "acquisitions_to_global_best_mean",
        "acquisitions_to_global_best_median",
        "first_top_fraction_acquisitions_mean",
        "final_objective_gap_mean",
        "final_objective_regret_fraction_mean",
        "final_best_objective_percentile_mean",
        "final_top_fraction_hit_rate_mean",
        "final_top_fraction_hit_count_mean",
        "fit_seconds_sum",
        "predict_seconds_sum",
    }
    partial_standard_tables = [
        (
            ACTIVE_COMBINED_HEAD_CELL_ID,
            _single_table_with_columns(
                combined_head_cell,
                full_metric_columns,
                expected_rows=5,
            ),
        )
    ]
    _assert_exact_grid(
        (
            (
                row["task"],
                row["feature_set"],
                row["strategy"],
                row["initial_fraction"],
            )
            for row in partial_standard_tables[0][1]
        ),
        product(
            ("matbench_jdft2d",),
            ("magpie",),
            ACTIVE_STANDARD_STRATEGIES,
            ("0.1",),
        ),
        label="Active-learning combined-head full-metric table",
    )

    latest_full_table_cell = _cell_by_id(
        notebook,
        ACTIVE_LATEST_FULL_TABLE_CELL_ID,
    )
    if "active_learning_aggregate_summary.csv" not in _source(latest_full_table_cell):
        raise ValueError("Active-learning latest full-table source anchor is missing.")
    if (
        "Latest Drive run: "
        "/content/drive/MyDrive/matbench_active_learning_results/"
        "gpu_20260606_183904_utc"
    ) not in _stream_text(latest_full_table_cell):
        raise ValueError("Active-learning latest full table is not linked to its run.")
    partial_standard_tables.append(
        (
            ACTIVE_LATEST_FULL_TABLE_CELL_ID,
            _single_table_with_columns(
                latest_full_table_cell,
                full_metric_columns,
                expected_rows=10,
            ),
        )
    )
    _assert_exact_grid(
        (
            (
                row["task"],
                row["feature_set"],
                row["strategy"],
                row["initial_fraction"],
            )
            for row in partial_standard_tables[1][1]
        ),
        product(
            ("matbench_phonons",),
            ("magpie_structure_all",),
            ACTIVE_STANDARD_STRATEGIES,
            ("0.1", "0.2"),
        ),
        label="Active-learning latest-run full-metric table",
    )

    stress_table_cell = _cell_by_id(notebook, ACTIVE_STRESS_TABLE_CELL_ID)
    if "active_learning_aggregate_summary.csv" not in _source(stress_table_cell):
        raise ValueError("Active-learning stress table source anchor is missing.")
    stress_rows = _single_table_with_columns(
        stress_table_cell,
        {
            "task",
            "feature_set",
            "strategy",
            "initial_fraction",
            "n_runs",
            "found_global_best_rate",
            "final_objective_regret_fraction_mean",
            "final_top_fraction_hit_rate_mean",
            "final_best_objective_percentile_mean",
        },
        expected_rows=16,
    )
    stress_key_count = _assert_exact_grid(
        (
            (
                row["task"],
                row["feature_set"],
                row["strategy"],
                row["initial_fraction"],
            )
            for row in stress_rows
        ),
        product(
            ("matbench_jdft2d", "matbench_phonons"),
            ("magpie", "magpie_structure_all"),
            ACTIVE_STRESS_STRATEGIES,
            ("0.05",),
        ),
        label="Active-learning stress display table",
    )

    rows: list[dict[str, str]] = []
    for source_row in standard_rows:
        feature_display = source_row["feature_set_display"]
        if feature_display not in FEATURE_DISPLAY_TO_KEY:
            raise ValueError(f"Unexpected feature display name: {feature_display!r}.")
        feature_set = FEATURE_DISPLAY_TO_KEY[feature_display]
        task = source_row["task"]
        row = {field: "" for field in ACTIVE_FIELDS}
        row.update(
            {
                "variant": "standard_top5pct",
                "source_run_id": run_map[(task, feature_set)],
                "task": task,
                "unit": TASK_UNITS[task],
                "feature_set": feature_set,
                "feature_set_display": feature_display,
                "strategy": source_row["strategy"],
                "initial_fraction": source_row["initial_fraction"],
                "top_fraction": "0.05",
                "n_runs": "15",
                "final_objective_regret_fraction_mean": source_row[
                    "final_objective_regret_fraction_mean"
                ],
                "final_best_objective_percentile_mean": source_row[
                    "final_best_objective_percentile_mean"
                ],
                "final_top_fraction_hit_rate_mean": source_row[
                    "final_top_fraction_hit_rate_mean"
                ],
                "notebook_cell_id": ACTIVE_STANDARD_TABLE_CELL_ID,
                "value_origin": "mixed_notebook_evidence",
            }
        )
        rows.append(row)

    standard_row_index = {
        (
            row["task"],
            row["feature_set"],
            row["strategy"],
            row["initial_fraction"],
        ): row
        for row in rows
    }
    for cell_id, table in partial_standard_tables:
        for source_row in table:
            key = (
                source_row["task"],
                source_row["feature_set"],
                source_row["strategy"],
                source_row["initial_fraction"],
            )
            if key not in standard_row_index:
                raise ValueError(
                    f"Partial full-metric row {key} is absent from the 40-row table."
                )
            target_row = standard_row_index[key]
            if "source_run" in source_row and source_row["source_run"] != (
                target_row["source_run_id"]
            ):
                raise ValueError(f"Partial full-metric row {key} has the wrong run ID.")
            for field in full_metric_columns:
                if field not in ACTIVE_FIELDS:
                    continue
                if target_row[field] and target_row[field] != source_row[field]:
                    raise ValueError(
                        f"Conflicting displayed value for {key}, field {field}: "
                        f"{target_row[field]!r} != {source_row[field]!r}."
                    )
                target_row[field] = source_row[field]
            target_row["notebook_cell_id"] += f";{cell_id}"

    for source_row in stress_rows:
        row = {field: "" for field in ACTIVE_FIELDS}
        for field in ACTIVE_FIELDS:
            if field in source_row:
                row[field] = source_row[field]
        row.update(
            {
                "variant": "stress_top1pct",
                "source_run_id": stress_run_id,
                "top_fraction": "0.01",
                "notebook_cell_id": ACTIVE_STRESS_TABLE_CELL_ID,
                "value_origin": "mixed_notebook_evidence",
            }
        )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            0 if row["variant"] == "standard_top5pct" else 1,
            TASK_ORDER[row["task"]],
            FEATURE_ORDER[row["feature_set"]],
            float(row["initial_fraction"]),
            ACTIVE_STRATEGY_ORDER[row["strategy"]],
        )
    )
    if len(rows) != 56:
        raise ValueError(f"Expected 56 active-learning evidence rows, found {len(rows)}.")
    unique_keys = {
        (
            row["variant"],
            row["task"],
            row["feature_set"],
            row["strategy"],
            row["initial_fraction"],
        )
        for row in rows
    }
    if len(unique_keys) != len(rows):
        raise ValueError("Active-learning evidence condition keys are not unique.")

    provenance: dict[str, object] = {
        "schema_version": 1,
        "source_notebook": _display_path(notebook_path),
        "source_notebook_sha256": _sha256(notebook_path),
        "source_notebook_git_blob": _git_blob_sha1(notebook_path),
        "execution_users": _execution_users(notebook),
        "standard_run_map": {
            f"{task}/{feature_set}": run_id
            for (task, feature_set), run_id in sorted(run_map.items())
        },
        "stress_run_id": stress_run_id,
        "successful_cell_condition_counts": successful_cell_condition_counts,
        "condition_grids_exact": True,
        "evidence_cells": {
            "standard_table": ACTIVE_STANDARD_TABLE_CELL_ID,
            "standard_partial_full_metric_tables": [
                ACTIVE_COMBINED_HEAD_CELL_ID,
                ACTIVE_LATEST_FULL_TABLE_CELL_ID,
            ],
            "stress_table": ACTIVE_STRESS_TABLE_CELL_ID,
            "standard_runs": [
                cell_id for cell_id, _, _, _ in ACTIVE_STANDARD_RUN_CELLS
            ],
            "stress_run": ACTIVE_STRESS_RUN_CELL,
        },
        "row_count": len(rows),
        "standard_row_count": standard_key_count,
        "standard_rows_with_additional_full_metrics": sum(
            len(table) for _, table in partial_standard_tables
        ),
        "stress_row_count": stress_key_count,
        "field_origins": {
            "source_run_id": (
                "Final completion lines in successful allowlisted run cells; "
                "five rows also display source_run in the combined-head table."
            ),
            "standard_common_metrics": (
                "The 40-row combined ranking HTML table in the executed notebook."
            ),
            "standard_additional_metrics": (
                "Full aggregate HTML tables display the remaining fields for "
                "five JDFT2D/Magpie/10% rows and ten "
                "phonons/structure/10%-20% rows."
            ),
            "standard_top_fraction": (
                "Executed runner default 0.05; the standard commands do not "
                "override it, and a rendered run summary corroborates it."
            ),
            "standard_n_runs": (
                "Exact five-fold by three-repeat condition grids; the partial "
                "full-metric tables also display 15."
            ),
            "units": (
                "Displayed Matbench task metadata in the partial standard and "
                "complete stress tables."
            ),
            "stress_metrics": (
                "The complete 16-row aggregate HTML table in the executed notebook."
            ),
            "blank_standard_fields": (
                "Left blank where the 40-row table omits a field and neither "
                "partial full-metric table displays that condition."
            ),
        },
        "effective_configuration": {
            "standard_explicit": {
                "strategies": list(ACTIVE_STANDARD_STRATEGIES),
                "initial_fractions": [0.1, 0.2],
                "max_acquisitions": 100,
                "acquisition_batch_size": 10,
                "repeats": 3,
                "device": "cuda",
            },
            "stress_explicit": {
                "tasks": ["matbench_jdft2d", "matbench_phonons"],
                "feature_sets": ["magpie", "magpie_structure_all"],
                "strategies": list(ACTIVE_STRESS_STRATEGIES),
                "folds": [0, 1],
                "initial_fractions": [0.05],
                "top_fraction": 0.01,
                "max_acquisitions": 100,
                "acquisition_batch_size": 10,
                "repeats": 1,
                "device": "cuda",
            },
            "shared_runner_defaults": {
                "random_seed": 42,
                "objective": "maximize",
                "initial_selection": "random",
                "max_acquisition_fraction": None,
                "ucb_beta": 0.5,
                "n_estimators": 500,
                "tabpfn_n_estimators": 8,
                "tabpfn_predict_batch_size": 128,
                "feature_n_jobs": 1,
                "use_feature_cache": True,
                "show_feature_progress": False,
                "include_oracle": False,
            },
            "standard_runner_defaults": {
                "folds": "all official folds",
                "top_fraction": 0.05,
            },
        },
        "executed_code": {
            "runner": _code_revision_record(
                "scripts/run_active_learning_screening.py",
                executed_commit=ACTIVE_RUNNER_COMMIT,
                executed_blob=ACTIVE_RUNNER_BLOB,
                evidence=(
                    "The runtime clone did not print HEAD; repository history "
                    "shows this blob at the active-learning commit, notebook "
                    "commit, and current checkout."
                ),
                certainty=(
                    "history_compatible_inference; runtime HEAD was not printed"
                ),
            ),
            "helpers": {
                path: _code_revision_record(
                    path,
                    executed_commit=ACTIVE_RUNNER_COMMIT,
                    executed_blob=blob,
                    evidence=(
                        "Repository-history compatibility; runtime HEAD was not "
                        "printed by the notebook."
                    ),
                    certainty=(
                        "history_compatible_inference; runtime HEAD was not printed"
                    ),
                )
                for path, blob in ACTIVE_HELPER_BLOBS.items()
            },
            "runtime_commit_certainty": (
                "History-compatible inference, not direct proof: the clone cell "
                "did not print a checkout SHA."
            ),
        },
        "extraction_precision": (
            "Values are preserved at the precision displayed by the notebook HTML."
        ),
        "generator": "scripts/extract_colab_notebook_evidence.py",
        "redaction_policy": (
            "Only allowlisted HTML table cells and completion identifiers are "
            "emitted; raw stdout and interactive prompts are never copied."
        ),
        "limitations": [
            "The original timestamped result trees and trace CSV are unavailable.",
            "Only three aggregate metrics are displayed for all 40 standard conditions.",
            "Additional aggregate fields are displayed for only 15 standard conditions.",
            "Exact acquisition trajectories cannot be reconstructed from embedded PNGs.",
            "Stress-test conditions contain only two runs each.",
        ],
    }
    return rows, provenance


def _csv_text(rows: Iterable[dict[str, str]], fields: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def render_artifacts(
    small_notebook: Path = DEFAULT_SMALL_NOTEBOOK,
    active_notebook: Path = DEFAULT_ACTIVE_NOTEBOOK,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[Path, str]:
    small_notebook = small_notebook.resolve()
    active_notebook = active_notebook.resolve()
    output_root = output_root.resolve()
    small_rows, small_provenance = _extract_small_data(
        _load_notebook(small_notebook),
        small_notebook,
    )
    active_rows, active_provenance = _extract_active_learning(
        _load_notebook(active_notebook),
        active_notebook,
    )
    artifacts = {
        output_root / "small_data" / "tables" / "notebook_summary.csv": _csv_text(
            small_rows,
            SMALL_FIELDS,
        ),
        output_root / "small_data" / "provenance.json": _json_text(small_provenance),
        output_root
        / "active_learning"
        / "tables"
        / "notebook_summary.csv": _csv_text(active_rows, ACTIVE_FIELDS),
        output_root / "active_learning" / "provenance.json": _json_text(
            active_provenance
        ),
    }
    for path, content in artifacts.items():
        if SECRET_PATTERN.search(content):
            raise ValueError(f"Secret-like token detected in generated artifact {path}.")
    return artifacts


def verify_artifacts(
    small_notebook: Path = DEFAULT_SMALL_NOTEBOOK,
    active_notebook: Path = DEFAULT_ACTIVE_NOTEBOOK,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[str]:
    mismatches: list[str] = []
    for path, expected in render_artifacts(
        small_notebook=small_notebook,
        active_notebook=active_notebook,
        output_root=output_root,
    ).items():
        if not path.is_file():
            mismatches.append(f"missing: {path}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            mismatches.append(f"out of date: {path}")
    return mismatches


def main() -> None:
    args = _parse_args()
    artifacts = render_artifacts(
        small_notebook=args.small_notebook,
        active_notebook=args.active_notebook,
        output_root=args.output_root,
    )
    if args.check:
        mismatches = verify_artifacts(
            small_notebook=args.small_notebook,
            active_notebook=args.active_notebook,
            output_root=args.output_root,
        )
        if mismatches:
            for mismatch in mismatches:
                print(f"FAIL  {mismatch}")
            raise SystemExit(1)
        for path in artifacts:
            print(f"PASS  {_display_path(path)}")
        return

    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"WROTE {_display_path(path)}")


if __name__ == "__main__":
    main()
