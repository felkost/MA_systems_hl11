"""`config.py` must put the cache-location variables into `os.environ` itself.

Stage 0 measured that pydantic-settings parses `.env` into `Settings` and
exports nothing: Hugging Face and pip read `os.environ` and nothing else, so
without an explicit write the model cache lands wherever the ambient
environment says -- which on a fresh machine can be `C:\\Users\\...`, the one
place CLAUDE.md's Forbidden list bans it from.

This runs the check in a **child process** with a deliberately hostile
ambient environment, not in-process: stage 0's own finding was that the rule
looked satisfied only because this machine's ambient `HF_HOME` already
happened to point off the system drive. Asserting from the current process
would repeat exactly that mistake.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CHILD_SCRIPT = (
    "import config\n"
    "import os\n"
    "print(os.environ['HF_HOME'])\n"
    "print(os.environ['PIP_CACHE_DIR'])\n"
    "print(os.environ['DEEPEVAL_HOME'])\n"
)


def test_config_exports_cache_paths_to_environ(tmp_path: Path) -> None:
    # Start from the real environment (Windows subprocess startup needs
    # PATH/SYSTEMROOT/etc.) and override only the two variables under test,
    # plus PYTHONPATH so a bare `import config` resolves from the child's
    # unrelated `tmp_path` cwd.
    hostile_env = dict(os.environ)
    hostile_env["PYTHONPATH"] = str(PROJECT_ROOT)
    hostile_env["HF_HOME"] = "C:\\Users\\hostile\\hf-cache"
    hostile_env["PIP_CACHE_DIR"] = "C:\\Users\\hostile\\pip-cache"
    hostile_env["DEEPEVAL_HOME"] = "C:\\Users\\hostile\\.deepeval"

    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        cwd=tmp_path,  # no .env here -- nothing this test relies on can leak in
        env=hostile_env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    hf_home, pip_cache_dir, deepeval_home = result.stdout.strip().splitlines()
    assert hf_home == str(PROJECT_ROOT / ".cache" / "hf")
    assert pip_cache_dir == str(PROJECT_ROOT / ".cache" / "pip")
    assert deepeval_home == str(PROJECT_ROOT / ".cache" / "deepeval")
    assert "hostile" not in hf_home
    assert "hostile" not in pip_cache_dir
    assert "hostile" not in deepeval_home
