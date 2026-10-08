# Quanifi — quantum-computing components for Apache NiFi
# Copyright (C) 2026 Neilson Ramalho
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation. This program is distributed WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
#
# You should have received a copy of the license along with this program; if not,
# see <https://www.gnu.org/licenses/>. Commercial licensing is also available:
# see COMMERCIAL.md at the repository root.

import datetime
import html as html_lib
import json
import math
import os
from pathlib import Path

import numpy as np
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import PropertyDescriptor, StandardValidators
from nifiapi.__jvm__ import JvmHolder

SENTINEL = "<!-- RUNS_START -->"

TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quanifi &mdash; <!-- FLOW_TITLE --></title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #ffffff;
           color: #24292f; margin: 0; padding: 24px; }
    h1 { color: #0969da; font-size: 1.5rem; border-bottom: 1px solid #d0d7de;
         padding-bottom: 12px; margin-bottom: 24px; }
    h3 { color: #0969da; font-size: 0.85rem; margin: 0 0 12px;
         text-transform: uppercase; letter-spacing: 0.07em; }
    .run-card { background: #f6f8fa; border: 1px solid #d0d7de;
                border-radius: 10px; margin-bottom: 32px; overflow: hidden; }
    .run-header { background: #eaeef2; padding: 12px 20px; display: flex;
                  justify-content: space-between; align-items: center;
                  border-bottom: 1px solid #d0d7de; }
    .run-title { color: #0969da; font-weight: 600; font-size: 0.95rem; }
    .run-time  { color: #8c959f; font-size: 0.82rem; font-family: monospace; }
    .run-body, .run-footer { display: flex; }
    .run-footer { border-top: 1px solid #d0d7de; }
    .panel { padding: 20px; flex: 1; min-width: 0; }
    .panel + .panel { border-left: 1px solid #d0d7de; }
    .code-block { background: #ffffff; border: 1px solid #d8dee4; border-radius: 6px;
                  padding: 14px; font-family: 'Cascadia Code','Fira Code',monospace;
                  font-size: 0.8rem; color: #24292f; margin: 0; overflow-x: auto;
                  white-space: pre; line-height: 1.65; }
    table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
    th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #d0d7de; }
    th { color: #656d76; font-weight: 600; }
    td:first-child { color: #656d76; font-family: monospace; font-size: 0.8rem; }
    td:last-child  { color: #1f2328; font-family: monospace; }
    .prob-bar-cell { display: flex; align-items: center; gap: 8px; }
    .prob-bar { height: 6px; border-radius: 3px; background: #0969da; }
    .phase-badge { display: inline-block; padding: 1px 6px; border-radius: 4px;
                   font-size: 0.75rem; font-family: monospace; font-weight: 600; }
  </style>
</head>
<body>
  <h1><!-- FLOW_TITLE --></h1>
  <!-- RUNS_START -->
</body>
</html>
"""


def _fmt_complex(val):
    real, imag = val.real, val.imag
    r = abs(real) >= 1e-4
    i = abs(imag) >= 1e-4
    if not r and not i:
        return "0"
    if not i:
        return f"{real:.4f}"
    if not r:
        return f"{imag:+.4f}i"
    sign = "+" if imag >= 0 else "−"
    return f"{real:.4f} {sign} {abs(imag):.4f}i"


class QrispStatevectorSimulator(FlowFileTransform):
    """
    Computes the exact statevector of an unmeasured quantum circuit using Qrisp.
    Outputs probability distribution as JSON content (q0_left convention) and
    writes an HTML report card with amplitude and phase per basis state.
    """

    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "0.1.0"
        tags = [
            "quantum",
            "qrisp",
            "statevector",
            "debug",
            "simulation",
            "amplitude",
            "phase",
        ]
        dependencies = ["qrisp==0.9.5", "qiskit>=2.0.0,<2.5"]
        description = (
            "Computes the exact statevector of an unmeasured quantum circuit using "
            "Qrisp. Outputs probability distribution as JSON (q0_left) and writes "
            "an HTML card with amplitude and phase per basis state."
        )

    def __init__(self, **kwargs):
        JvmHolder.jvm = kwargs.get("jvm")
        super().__init__()

        self.reports_dir = PropertyDescriptor(
            name="Reports Directory",
            description="Folder where the HTML statevector report is written.",
            required=True,
            default_value="reports",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
        )
        self.flow_name = PropertyDescriptor(
            name="Flow Name",
            description="Used as the page title and report filename: {flow_name}-statevector.html.",
            required=True,
            default_value="circuit",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
        )
        self.report_file_name = PropertyDescriptor(
            name="Report File Name",
            description="Optional override for the report filename without .html extension.",
            required=False,
            default_value="",
        )
        self.descriptors = [
            self.reports_dir,
            self.flow_name,
            self.report_file_name,
        ]

    def getPropertyDescriptors(self):
        return self.descriptors

    def _failure(self, flowFile, msg):
        self.logger.error("QrispStatevectorSimulator: " + msg)
        return FlowFileTransformResult(
            relationship="failure",
            contents=bytes(flowFile.getContentsAsBytes() or b""),
            attributes={"sim.error": msg},
        )

    def _append_html(self, reports_dir, flow_name, file_name, card_html):
        out_dir = Path(reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = (file_name.strip() if file_name else f"{flow_name}-statevector") + ".html"
        out_path = out_dir / fname

        if out_path.exists():
            existing = out_path.read_text(encoding="utf-8")
        else:
            existing = TEMPLATE.replace("<!-- FLOW_TITLE -->", html_lib.escape(flow_name))

        if SENTINEL in existing:
            new_html = existing.replace(SENTINEL, f"{card_html}\n  {SENTINEL}", 1)
        else:
            new_html = existing + "\n" + card_html
        out_path.write_text(new_html, encoding="utf-8")

    def transform(self, context, flowFile):
        def prop(d):
            return context.getProperty(d).getValue()

        reports_dir = prop(self.reports_dir)
        flow_name = prop(self.flow_name)
        report_file_name = prop(self.report_file_name)

        raw_bytes = bytes(flowFile.getContentsAsBytes() or b"")
        if not raw_bytes:
            return self._failure(flowFile, "FlowFile contains no circuit content.")

        try:
            from qiskit import qasm2
            from qrisp import QuantumCircuit as QrispCircuit

            raw_str = raw_bytes.decode("utf-8")
            # Parse circuit via qasm2
            qc_qiskit = qasm2.loads(
                raw_str, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
            )
            # Remove any terminal measurements for statevector simulation
            qc_unmeasured = qc_qiskit.remove_final_measurements(inplace=False)
            n = qc_unmeasured.num_qubits

            # Load into Qrisp circuit and simulate statevector:
            qc_qrisp = QrispCircuit.from_qiskit(qc_unmeasured)
            sv = qc_qrisp.statevector_array()

        except Exception as exc:
            return self._failure(
                flowFile, f"Qrisp statevector simulation failed: {exc}"
            )

        probs = np.abs(sv) ** 2
        total_p = float(np.sum(probs))
        if total_p > 0:
            probs = probs / total_p

        # Qrisp statevector indices format(i, '0{n}b') match q0_left canonical convention!
        dist = {}
        sv_table = []
        for i in range(len(sv)):
            amp = sv[i]
            p = float(probs[i])
            bits = format(i, f"0{n}b")
            if p > 1e-7:
                dist[bits] = p
            deg = math.degrees(math.atan2(amp.imag, amp.real)) % 360
            sv_table.append({
                "state": bits,
                "real": float(amp.real),
                "imag": float(amp.imag),
                "prob": p,
                "phase_deg": deg,
            })

        sorted_dist = dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True))
        top_result, top_prob = next(iter(sorted_dist.items())) if sorted_dist else ("", 0.0)

        # Build HTML card
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        card = [f'<div class="run-card">']
        card.append(f'  <div class="run-header"><span class="run-title">{html_lib.escape(flow_name)}</span><span class="run-time">{now}</span></div>')
        card.append('  <div class="run-body">')
        card.append('    <div class="panel"><h3>Statevector &amp; Probabilities (q0-left)</h3><table>')
        card.append('      <thead><tr><th>State</th><th>Amplitude</th><th>Phase</th><th>Prob</th></tr></thead><tbody>')
        for item in sorted(sv_table, key=lambda x: x["prob"], reverse=True)[:16]:
            if item["prob"] < 1e-4:
                continue
            pct = item["prob"] * 100
            card.append(f'      <tr><td>|{item["state"]}⟩</td><td>{_fmt_complex(complex(item["real"], item["imag"]))}</td>')
            card.append(f'          <td>{item["phase_deg"]:.1f}°</td>')
            card.append(f'          <td><div class="prob-bar-cell"><div class="prob-bar" style="width:{min(100, int(pct*1.5))}px"></div>{pct:.2f}%</div></td></tr>')
        card.append('    </tbody></table></div>')
        card.append('  </div>')
        card.append('</div>')
        card_html = "\n".join(card)

        try:
            self._append_html(reports_dir, flow_name, report_file_name, card_html)
        except Exception as exc:
            self.logger.warn(f"Could not write statevector report: {exc}")

        content_bytes = json.dumps(sorted_dist, indent=2).encode("utf-8")
        attrs = {
            "circuit.num_qubits": str(n),
            "sim.bit_order": "q0_left",
            "sim.top_result": top_result,
            "sim.top_probability": f"{top_prob:.6f}",
            "sim.component": "QrispStatevectorSimulator",
            "sim.framework": "qrisp",
            "sim.qubit_count": str(n),
            "mime.type": "application/json",
        }

        return FlowFileTransformResult(
            relationship="success",
            contents=content_bytes,
            attributes=attrs,
        )
