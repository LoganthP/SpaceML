"""Generate an explicit Vercel frontend-only config for an HTTPS Docker backend."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def configuration(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("Backend must be an HTTPS origin without a path, credentials, query or fragment")
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    config["services"] = {"frontend": config["services"]["frontend"]}
    config["rewrites"] = [
        {"source": "/api", "destination": url.rstrip("/") + "/api"},
        {"source": "/api/:path*", "destination": url.rstrip("/") + "/api/:path*"},
        {"source": "/(.*)", "destination": {"service": "frontend"}},
    ]
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", required=True)
    args = parser.parse_args()
    config = configuration(args.backend_url)
    output = ROOT / "vercel.docker.json"
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}; deploy with vercel --local-config vercel.docker.json --prod")
