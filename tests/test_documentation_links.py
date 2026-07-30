"""Check that repository-local Markdown links resolve in a fresh clone."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "data:", "#")


class DocumentationLinkTests(unittest.TestCase):
    def test_local_markdown_links_resolve(self) -> None:
        broken: list[str] = []
        for markdown_path in sorted(PROJECT_ROOT.rglob("*.md")):
            if ".git" in markdown_path.parts:
                continue
            text = markdown_path.read_text(encoding="utf-8")
            for raw_target in MARKDOWN_LINK.findall(text):
                target = raw_target.strip().split()[0].strip("<>")
                if not target or target.startswith(EXTERNAL_PREFIXES):
                    continue
                target = unquote(target.split("#", 1)[0])
                if not target:
                    continue
                resolved = (markdown_path.parent / target).resolve()
                if not resolved.exists():
                    relative_source = markdown_path.relative_to(PROJECT_ROOT)
                    broken.append(f"{relative_source} -> {target}")

        self.assertEqual(broken, [], "Broken local links:\n" + "\n".join(broken))


if __name__ == "__main__":
    unittest.main()
