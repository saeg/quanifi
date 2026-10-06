#!/usr/bin/env python3
"""Run the five-processor Qiskit Grover demo after NiFi reports healthy.

Checks that the canvas is stopped, runs it, validates the newly written HTML
report, and stops the group again. The default target is 10 (two qubits).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import nifi_ready  # noqa: E402
import build_grover_examples as grover  # noqa: E402

ROOT = _TOOLS_DIR.parent


def put_json(path, token, body):
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Host": nifi_ready.HOST_HEADER,
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token,
    }
    req = urllib.request.Request(
        nifi_ready.BASE + path, data=data, headers=headers, method="PUT"
    )
    with urllib.request.urlopen(
        req, context=nifi_ready.CTX, timeout=nifi_ready.CALL_TIMEOUT
    ) as response:
        out = response.read().decode()
    return json.loads(out) if out else {}


def find_group(token):
    flow = nifi_ready.call("/flow/process-groups/root", token=token)
    groups = flow["processGroupFlow"]["flow"]["processGroups"]
    for group in groups:
        if group["component"]["name"] == grover.SIMPLE_GROUP_NAME:
            return group["component"]["id"]
    raise SystemExit(
        "quickstart group {!r} not found on the canvas".format(grover.SIMPLE_GROUP_NAME)
    )


def processor_states(token, gid):
    flow = nifi_ready.call("/flow/process-groups/{}".format(gid), token=token)
    processors = flow["processGroupFlow"]["flow"]["processors"]
    return {p["component"]["name"]: p["component"]["state"] for p in processors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_port = os.environ.get("QUANIFI_NIFI_PORT", "8443")
    parser.add_argument(
        "--base-url",
        default="https://127.0.0.1:{}/nifi-api".format(default_port),
    )
    parser.add_argument(
        "--user", default=os.environ.get("QUANIFI_NIFI_USERNAME", "admin")
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("QUANIFI_NIFI_PASSWORD", "quanifipassword"),
    )
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--leave-running", action="store_true")
    args = parser.parse_args(argv)

    nifi_ready._set_target(args.base_url)
    try:
        token = nifi_ready.login(args.user, args.password)
    except (urllib.error.URLError, OSError) as exc:
        print("quickstart_smoke: could not log in: {}".format(exc))
        return 1

    half_loaded = unconfigured = valid_count = total = None
    for attempt in range(3):
        half_loaded, unconfigured, valid_count, total = nifi_ready.audit_processors(
            token
        )
        if not half_loaded:
            break
        if attempt < 2:
            time.sleep(10)
    if half_loaded:
        print(
            "quickstart_smoke: {} processor(s) still half-loaded after 3 checks".format(
                len(half_loaded)
            )
        )
        return 2

    gid = find_group(token)
    states = processor_states(token, gid)
    not_stopped = {name: state for name, state in states.items() if state != "STOPPED"}
    if not_stopped:
        print(
            "quickstart_smoke: not every component is STOPPED: {}".format(not_stopped)
        )
        return 3

    if args.check_only:
        print("canvas present, {} processors, all STOPPED".format(len(states)))
        return 0

    report = Path(args.reports_dir) / "quickstart" / "qiskit-grover.html"
    before = report.read_text() if report.exists() else ""
    put_json("/flow/process-groups/" + gid, token, {"id": gid, "state": "RUNNING"})
    try:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            current = report.read_text() if report.exists() else ""
            if current != before and 'class="run-card"' in current:
                # Check only the latest card; earlier runs may have other targets.
                card = current.split('<section class="run-card">', 1)[1].split('</section>', 1)[0]
                import re
                top = re.search(r"<td>sim.top_result</td>\s*<td>([01]+)</td>", card)
                probability = re.search(r"<td>sim.top_probability</td>\s*<td>([0-9.]+)</td>", card)
                if top and probability:
                    if top.group(1) != grover.SIMPLE_TARGET or float(probability.group(1)) < 0.99:
                        print("FAIL: unexpected Grover result", top.group(1), probability.group(1))
                        return 1
                    print("PASS: Qiskit Grover target=10, top=" + top.group(1)
                          + ", probability=" + probability.group(1) + "; report=" + str(report))
                    return 0
            time.sleep(2)
        print("Timed out waiting for the Grover report; inspect the failure queue.")
        return 1
    finally:
        if not args.leave_running:
            put_json("/flow/process-groups/" + gid, token, {"id": gid, "state": "STOPPED"})


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
