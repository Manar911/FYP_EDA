import csv
from collections import Counter, defaultdict
from pathlib import Path


DATASET_PATH = Path("generated_data/dataset_full.csv")


def normalize_reason(reason: str) -> str:
    r = reason.strip().lower()

    if r == "accepted":
        return "accepted"

    if "unreachable" in r:
        return "range"

    if "runway too short" in r:
        return "runway"

    if "no medical capability" in r:
        return "medical_capability"

    if "temporarily closed by scenario constraints" in r:
        return "dynamic_closed"

    if "marked unsafe by scenario constraints" in r:
        return "dynamic_unsafe"

    if "country restricted by scenario constraints" in r:
        return "dynamic_country_restricted"

    if "airport status is" in r:
        return "closure_status"

    if "airport marked unsafe" in r:
        return "unsafe_status"

    if "airport is restricted" in r or "airport is military_restricted" in r:
        return "restricted_status"

    if "military-only airport not permitted" in r:
        return "military_only"

    return "other"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    overall = Counter()
    by_emergency = defaultdict(Counter)

    total_rows = 0
    total_rejected = 0
    total_accepted = 0

    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1

            emergency = row["emergency_type"].strip().lower()
            reason = row["feasibility_reason"].strip()

            category = normalize_reason(reason)

            overall[category] += 1
            by_emergency[emergency][category] += 1

            if category == "accepted":
                total_accepted += 1
            else:
                total_rejected += 1

    print("=" * 70)
    print("OVERALL REJECTION DISTRIBUTION")
    print("=" * 70)
    print(f"Total rows      : {total_rows}")
    print(f"Accepted rows   : {total_accepted}")
    print(f"Rejected rows   : {total_rejected}")
    print()

    for reason, count in overall.most_common():
        pct = (count / total_rows) * 100 if total_rows else 0.0
        print(f"{reason:25s} {count:6d}  ({pct:6.2f}%)")

    print()
    print("=" * 70)
    print("BY EMERGENCY TYPE")
    print("=" * 70)

    for emergency, counts in sorted(by_emergency.items()):
        emergency_total = sum(counts.values())
        print(f"\n[{emergency}] total={emergency_total}")
        for reason, count in counts.most_common():
            pct = (count / emergency_total) * 100 if emergency_total else 0.0
            print(f"  {reason:23s} {count:6d}  ({pct:6.2f}%)")


if __name__ == "__main__":
    main()