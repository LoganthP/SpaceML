import os, re, json
from pathlib import Path

src_dir = Path("frontend/src").resolve()

# Find any string or template literal starting with /api/ or containing /api/
api_str_regex = re.compile(r'([`\'"])(/api/[^`\'"]+)\1')

calls = []

for root, _, files in os.walk(src_dir):
    for f in files:
        if f.endswith((".ts", ".tsx")):
            fpath = Path(root) / f
            rel = fpath.relative_to(src_dir.parent.parent)
            text = fpath.read_text(encoding="utf-8")
            lines = text.split("\n")
            
            for idx, line in enumerate(lines, 1):
                for m in api_str_regex.finditer(line):
                    raw = m.group(2)
                    quote = m.group(1)
                    
                    # infer method
                    method = "GET"
                    if "method:" in line:
                        mm = re.search(r'method:\s*["\']([A-Z]+)["\']', line)
                        if mm: method = mm.group(1)
                    elif "createRun" in line:
                        method = "POST"
                    elif "uploadWithProgress" in line:
                        method = "POST"
                    elif "api.delete" in line or 'method: "DELETE"' in line or "DELETE" in line:
                        method = "DELETE"
                    elif "method: \"PUT\"" in line or "PUT" in line:
                        method = "PUT"
                    elif "method: \"POST\"" in line or "POST" in line:
                        method = "POST"
                    elif "useApi" in line or "usePoll" in line:
                        method = "GET"
                        
                    calls.append({
                        "file": str(rel),
                        "line": idx,
                        "raw": raw,
                        "method": method,
                        "quote": quote,
                        "line_text": line.strip()
                    })

with open("scripts/frontend_api_strings.json", "w") as out:
    json.dump(calls, out, indent=2)

print(f"Total /api/ string occurrences found: {len(calls)}")
