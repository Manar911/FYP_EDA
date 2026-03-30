import csv
from collections import Counter, defaultdict
from pathlib import Path

DATASET_PATH = Path("generated_data/dataset_full.csv")


def normalize_reason(reason: str) -> str:
    r = reason.strip().lower()

    if "accepted" in r:
        return "accepted"

    if "beyond extended range" in r or "unreachable" in r:
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

    overall_reason = Counter()
    by_emergency = defaultdict(Counter)
    zone_counter = Counter()
    binding_counter = Counter()
    feasible_per_scenario = defaultdict(int)

    total_rows = 0
    total_rejected = 0
    total_accepted = 0

    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "scenario_id",
            "emergency_type",
            "feasibility_reason",
            "distance_zone",
            "binding_side",
            "feasible",
        }
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1

            scenario_id = row["scenario_id"].strip()
            emergency = row["emergency_type"].strip().lower()
            reason = row["feasibility_reason"].strip()
            zone = row["distance_zone"].strip().lower()
            binding = row["binding_side"].strip().lower()
            feasible = int(row["feasible"])

            category = normalize_reason(reason)

            overall_reason[category] += 1
            by_emergency[emergency][category] += 1
            zone_counter[zone] += 1
            binding_counter[binding] += 1

            if feasible:
                total_accepted += 1
                feasible_per_scenario[scenario_id] += 1
            else:
                total_rejected += 1

    print("=" * 72)
    print("OVERALL REJECTION DISTRIBUTION")
    print("=" * 72)
    print(f"Total rows      : {total_rows}")
    print(f"Accepted rows   : {total_accepted}")
    print(f"Rejected rows   : {total_rejected}")
    print(f"Feasibility rate: {(total_accepted / total_rows) * 100:.2f}%")
    print()

    for reason, count in overall_reason.most_common():
        pct = (count / total_rows) * 100 if total_rows else 0.0
        print(f"{reason:25s} {count:6d}  ({pct:6.2f}%)")

    print()
    print("=" * 72)
    print("DISTANCE ZONE DISTRIBUTION")
    print("=" * 72)
    for zone, count in zone_counter.most_common():
        pct = (count / total_rows) * 100 if total_rows else 0.0
        print(f"{zone:25s} {count:6d}  ({pct:6.2f}%)")

    print()
    print("=" * 72)
    print("BINDING SIDE DISTRIBUTION")
    print("=" * 72)
    for side, count in binding_counter.most_common():
        pct = (count / total_rows) * 100 if total_rows else 0.0
        print(f"{side:25s} {count:6d}  ({pct:6.2f}%)")

    print()
    print("=" * 72)
    print("BY EMERGENCY TYPE")
    print("=" * 72)
    for emergency, counts in sorted(by_emergency.items()):
        emergency_total = sum(counts.values())
        print(f"\n[{emergency}] total={emergency_total}")
        for reason, count in counts.most_common():
            pct = (count / emergency_total) * 100 if emergency_total else 0.0
            print(f"  {reason:23s} {count:6d}  ({pct:6.2f}%)")

    print()
    print("=" * 72)
    print("FEASIBLE AIRPORTS PER SCENARIO")
    print("=" * 72)

    counts = list(feasible_per_scenario.values())
    if counts:
        avg_count = sum(counts) / len(counts)
        print(f"Scenarios with >=1 feasible airport : {len(counts)}")
        print(f"Average feasible airports/scenario  : {avg_count:.2f}")
        print(f"Minimum feasible airports/scenario  : {min(counts)}")
        print(f"Maximum feasible airports/scenario  : {max(counts)}")

        low_count = sum(1 for c in counts if c <= 1)
        print(f"Scenarios with <=1 feasible airport : {low_count}")
    else:
        print("No feasible airports found in any scenario.")


if __name__ == "__main__":
    main()