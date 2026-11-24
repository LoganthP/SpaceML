import json

with open("scripts/e2e_results.json") as f:
    d = json.load(f)

print(f"Total Routes Checked: {len(d['routes_tested'])} (13 in dark, 13 in light)")
print("\nControls Tested:")
for c in d["controls_tested"]:
    status = "PASS" if c["success"] else "FAIL"
    print(f"  [{status}] {c['control']}: {c['action']}")

total_violations = sum(d["axe_violations"].values())
print(f"\nTotal Axe Violations: {total_violations}")

overflow_count = sum(1 for v in d["overflow_checks"].values() if v["overflow"])
print(f"Total Horizontal Overflows (375px, 768px, 1280px): {overflow_count}")

print(f"\nUnexpected Console Errors: {len(d['console_errors'])}")
for e in d["console_errors"]:
    print(f"  {e}")

print(f"\nUnexpected Failed Requests: {len(d['failed_requests'])}")
for r in d["failed_requests"]:
    print(f"  {r}")

print(f"\nJobs Lifecycle: {d['jobs_lifecycle']}")
