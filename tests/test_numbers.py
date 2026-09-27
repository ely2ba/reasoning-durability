"""Every number in the paper is recomputed from frozen outputs: `python src/paper_numbers.py --check` must
exit 0. It recomputes every macro and compares the result with the written file (paper/numbers.generated.tex
until the switch to paper/numbers.tex), and fails if the paper uses a key the registry does not define."""

import subprocess
import sys

from conftest import REPO, needs


def test_numbers_check():
    script = needs(REPO / "src" / "paper_numbers.py", "src/paper_numbers.py")
    result = subprocess.run([sys.executable, str(script), "--check"], cwd=REPO, capture_output=True, text=True,
                            timeout=1800)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
