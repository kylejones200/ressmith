#!/usr/bin/env python
"""
Smoke test for ressmith integration with timesmith.

This script:
1. Installs ressmith in development mode
2. Installs timesmith (from local path or PyPI)
3. Runs the integration example
4. Exits with code 0 on success
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main() -> int:
    """Run smoke test."""
    repo_root = Path(__file__).parent.parent

    print("ResSmith Smoke Test")
    print("=" * 50)

    # Step 1: Install timesmith (try local first, then PyPI)
    print("\n1. Installing timesmith...")
    timesmith_path = repo_root.parent / "timesmith"
    if timesmith_path.exists() and (timesmith_path / "pyproject.toml").exists():
        print(f"   Found local timesmith at {timesmith_path}")
        code = run_command(["pip", "install", "-e", str(timesmith_path)])
        if code != 0:
            print("   Local install failed, trying PyPI...")
            code = run_command(["pip", "install", "timesmith>=0.2.0"])
    else:
        print("   Installing timesmith from PyPI...")
        code = run_command(["pip", "install", "timesmith>=0.2.0"])

    if code != 0:
        print("Failed to install timesmith")
        return code

    # Step 2: Install ressmith in development mode
    print("\n2. Installing ressmith in development mode...")
    code = run_command(["pip", "install", "-e", "."], cwd=repo_root)
    if code != 0:
        print("Failed to install ressmith")
        return code

    # Step 3: Run integration example
    print("\n3. Running integration example...")
    example_path = repo_root / "examples" / "integration_timesmith.py"
    if not example_path.exists():
        print(f"Example not found: {example_path}")
        return 1

    code = run_command([sys.executable, str(example_path)], cwd=repo_root)
    if code != 0:
        print("Integration example failed")
        return code

    print("\nSmoke test passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

