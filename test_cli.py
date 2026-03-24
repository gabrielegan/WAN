import subprocess
import sys


def test_compareWANSnoprint_requires_wan_pairs_file():
    # Running with no args should fail with exit code 1 and print usage.
    proc = subprocess.run(
        [sys.executable, "compareWANSnoprint.py"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    assert (
        "Usage: python compareWANSnoprint.py [-i INDICATOR_FILE] <wan_pairs_file>"
        in proc.stderr
    )