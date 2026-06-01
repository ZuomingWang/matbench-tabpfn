"""Check that the core project dependencies can be imported."""

from __future__ import annotations

import importlib


REQUIRED_PACKAGES = [
    "matbench",
    "matminer",
    "numpy",
    "pandas",
    "pymatgen",
    "sklearn",
    "tabpfn",
]


def main() -> None:
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except Exception as exc:  # pragma: no cover - diagnostic script
            missing.append((package, exc))

    if missing:
        print("Missing or broken imports:")
        for package, exc in missing:
            print(f"- {package}: {exc}")
        raise SystemExit(1)

    print("Core project dependencies imported successfully.")


if __name__ == "__main__":
    main()
