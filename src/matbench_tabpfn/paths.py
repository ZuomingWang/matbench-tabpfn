"""Output path helpers for reproducible experiment runs."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .settings import PROJECT_ROOT


@dataclass(frozen=True)
class RunPaths:
    """Canonical paths for a single experiment run."""

    root: Path
    metrics: Path
    predictions: Path
    features: Path
    figures: Path
    tables: Path
    logs: Path

    def mkdirs(self) -> None:
        for path in asdict(self).values():
            Path(path).mkdir(parents=True, exist_ok=True)

    def as_posix_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def default_run_id(prefix: str = "gpu") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    return f"{prefix}_{stamp}"


def create_run_paths(
    project_root: Path | str = PROJECT_ROOT,
    *,
    run_id: str | None = None,
    base_dir: Path | str = "results/runs",
    latest_root: Path | str | None = None,
    clean: bool = False,
    update_latest: bool = True,
) -> RunPaths:
    """Create a unified output tree for one run."""

    project_root = Path(project_root).resolve()
    run_id = run_id or default_run_id()
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = project_root / base_path

    root = base_path / run_id
    if root.exists() and clean:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    paths = RunPaths(
        root=root,
        metrics=root / "metrics",
        predictions=root / "predictions",
        features=root / "features",
        figures=root / "figures",
        tables=root / "tables",
        logs=root / "logs",
    )
    paths.mkdirs()

    if update_latest:
        latest_parent = Path(latest_root).resolve() if latest_root else project_root / "results"
        latest_parent.mkdir(parents=True, exist_ok=True)
        latest = latest_parent / "latest"
        if latest.exists() or latest.is_symlink():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        try:
            latest.symlink_to(root, target_is_directory=True)
        except OSError:
            (latest_parent / "latest_run.txt").write_text(
                str(root) + "\n", encoding="utf-8"
            )

    return paths


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path | str, data: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collect_environment_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect enough environment details to make a run auditable."""

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    try:
        import sklearn

        manifest["packages"]["scikit_learn"] = sklearn.__version__
    except Exception as exc:  # pragma: no cover - manifest best effort
        manifest["packages"]["scikit_learn_error"] = repr(exc)

    try:
        import matbench

        manifest["packages"]["matbench"] = getattr(matbench, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - manifest best effort
        manifest["packages"]["matbench_error"] = repr(exc)

    try:
        import matminer

        manifest["packages"]["matminer"] = getattr(matminer, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - manifest best effort
        manifest["packages"]["matminer_error"] = repr(exc)

    try:
        import tabpfn

        manifest["packages"]["tabpfn"] = getattr(tabpfn, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - manifest best effort
        manifest["packages"]["tabpfn_error"] = repr(exc)

    try:
        import torch

        manifest["packages"]["torch"] = torch.__version__
        manifest["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(idx).total_memory / 1024**3, 3
                    ),
                }
                for idx in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - manifest best effort
        manifest["cuda"] = {"error": repr(exc)}

    if extra:
        manifest["config"] = extra

    return manifest
