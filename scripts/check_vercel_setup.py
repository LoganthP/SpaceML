"""Check the deployed Space ML API, protection, mount and operator capabilities.

Uses only the standard library. Never prints or stores authentication secrets.
"""
import argparse
import base64
import getpass
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class HealthRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        old, new = urlsplit(request.full_url), urlsplit(newurl)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            raise ValueError("Health redirected outside the app, often to Vercel sign-in. Use the public Production domain or configure an automation bypass; credentials were not forwarded.")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def fetch_health(origin, headers):
    request = urllib.request.Request(origin.rstrip("/") + "/api/health", headers=headers)
    opener = urllib.request.build_opener(HealthRedirect())
    with opener.open(request, timeout=45) as response:
        if "application/json" not in response.headers.get("Content-Type", ""):
            raise ValueError("Health returned HTML, often Vercel's sign-in page. Use the public Production domain or configure a deployment-protection automation bypass.")
        body = json.load(response)
        if not isinstance(body, dict) or "capabilities" not in body:
            raise ValueError("This URL did not return the Space ML health API. Check the /api service rewrite.")
        return body


def check(origin, headers, wait=0):
    deadline = time.monotonic() + wait
    while True:
        body = fetch_health(origin, headers)
        state = body.get("state", "unknown")
        checks = body.get("checks") or {}
        sandbox, drive = checks.get("sandbox") or {}, checks.get("drive") or {}
        print(f"Mode: {body.get('mode')}; execution: {body.get('execution')}; state: {state}", flush=True)
        print(f"Sandbox available: {sandbox.get('available')}; Drive: {drive.get('name')}; mounted: {drive.get('mounted')}; writable: {drive.get('writable')}", flush=True)
        missing = body.get("setup_required") or []
        if missing:
            print("Set these project environment variables, then REDEPLOY: " + ", ".join(missing))
            return 2
        if state == "ready":
            if not (sandbox.get("available") is True and drive.get("mounted") is True and drive.get("writable") is True and drive.get("paths_valid") is True):
                print("Readiness is inconsistent: inspect /api/health/drive before training.")
                return 3
            capabilities = body.get("capabilities") or {}
            if not (capabilities.get("write") and capabilities.get("train")):
                print("Compute is ready; operator sign-in is still required. Open /api/session in your browser, or rerun with --username operator.")
                return 4
            print("Compute, persistent paths and operator access are ready. Verify OPS-SAT Quick Train and a Sandbox restart next; health alone does not prove persistence.")
            return 0
        failure = body.get("startup_error") or {}
        if failure:
            print(f"Startup stage: {failure.get('stage')}; error: {failure.get('type')}")
        if body.get("detail"):
            print(body["detail"])
        if time.monotonic() >= deadline:
            print("Not ready. Follow VERCEL-SANDBOX-SETUP.md and inspect the Function/Sandbox startup diagnostics.")
            return 3
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", help="Public origin, e.g. https://spaceml.vercel.app")
    parser.add_argument("--username", help="Prompt privately for the operator password")
    parser.add_argument("--wait", type=int, default=0, metavar="SECONDS")
    args = parser.parse_args()
    parsed = urlsplit(args.origin)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        parser.error("Provide an origin only, without credentials, a page path or query string")
    if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        parser.error("Use HTTPS for deployed requests")
    headers = {"Accept": "application/json"}
    bypass = os.environ.get("SMC_DEPLOYMENT_BYPASS") or os.environ.get("VERCEL_AUTOMATION_BYPASS_SECRET")
    if bypass:
        headers["x-vercel-protection-bypass"] = bypass
    if args.username:
        password = getpass.getpass("Space ML operator password: ")
        headers["Authorization"] = "Basic " + base64.b64encode(f"{args.username}:{password}".encode()).decode()
    try:
        return check(args.origin, headers, max(0, args.wait))
    except urllib.error.HTTPError as error:
        print(f"Health request failed: HTTP {error.code}. Check operator credentials, deployment protection and the API rewrite.")
    except (OSError, ValueError) as error:
        print(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
