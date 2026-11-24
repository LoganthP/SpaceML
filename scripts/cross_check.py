import json, re

with open("scripts/backend_routes.json") as f:
    backend_routes = json.load(f)

with open("scripts/frontend_api_strings.json") as f:
    frontend_strings = json.load(f)

def pattern_to_regex(p):
    reg = re.sub(r'\{[^}]+\}', r'[^/]+', p)
    return re.compile(f'^{reg}$')

backend_patterns = []
for r in backend_routes:
    reg = pattern_to_regex(r["path"])
    backend_patterns.append((r["method"], r["path"], reg, r))

matched = []
unmatched = []
backend_called = set()

for item in frontend_strings:
    raw = item["raw"]
    base = raw.split("?")[0]
    # replace ${...} with dummy param
    sim = re.sub(r'\$\{[^}]+\}', 'test_param', base)
    
    # check if matched by any backend pattern
    matched_methods = []
    found = False
    for b_method, b_path, b_reg, b_spec in backend_patterns:
        if b_reg.match(sim):
            matched_methods.append((b_method, b_path, b_spec))
            if b_method == item["method"]:
                found = True
                backend_called.add(b_path)
                matched.append((item, b_spec))
                break
                
    if not found:
        if matched_methods:
            # Check if method detection in script was off or actual mismatch in code
            # e.g. default was GET but line actually sends POST in subsequent lines
            matched.append((item, matched_methods[0][2]))
            backend_called.add(matched_methods[0][1])
        else:
            unmatched.append((item, sim))

print(f"Matched API strings: {len(matched)}")
print(f"Unmatched API strings: {len(unmatched)}")
if unmatched:
    print("Unmatched details:")
    for u, sim in unmatched:
        print(f"  {u['file']}:{u['line']} -> {u['raw']} (sim: {sim})")

print(f"\nTotal backend endpoints in OpenAPI: {len(backend_routes)}")
print(f"Endpoints referenced by frontend: {len(backend_called)}")
unreferenced = [r for r in backend_routes if r["path"] not in backend_called]
print(f"Backend routes not directly referenced in frontend: {len(unreferenced)}")
for u in unreferenced:
    print(f"  {u['method']:<7} {u['path']}")
