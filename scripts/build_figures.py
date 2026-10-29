"""Regenerate every figure / 重新生成所有图片: runs docs/*/figures/*.py."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    scripts = sorted(ROOT.glob("docs/*/figures/*.py"))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts")}
    failed = []
    for script in scripts:
        print(f"→ {script.relative_to(ROOT)}")
        if subprocess.run([sys.executable, str(script)], env=env).returncode != 0:
            failed.append(script)
    if failed:
        sys.exit(f"failed: {[str(f.relative_to(ROOT)) for f in failed]}")


if __name__ == "__main__":
    main()
