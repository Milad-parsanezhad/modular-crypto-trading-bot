import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_bot.evidence_ledger import validate_ledger  # noqa: E402


if __name__ == "__main__":
    checked = validate_ledger()
    print(f"validated {len(checked)} evidence records")
