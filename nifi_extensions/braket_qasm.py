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

"""qasm2/qasm3 -> Braket-native OpenQASM 3 translation, shared by
BraketSimulator and BraketDevice.

The old adapter inlined Qiskit's stdgates.inc into OpenQASM 3. Differential
execution exposed a four-qubit Grover peak split between the marked state
and its neighbour. A later isolation reproduced the error with only
``ctrl @ x q[1], q[3]`` on four qubits: no QASM2 translation or gphase is
needed. Braket's small controlled-gate numerical update can reshape a
non-contiguous output slice into a detached copy and lose the write.

The original phase-convention explanation is superseded for this case.
The defect reproduces in amazon-braket-default-simulator 1.39.4 and 1.43.1;
upstream PR #415 proposes the corresponding repair (open on 2026-11-13):
https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/415
This evidence concerns the local simulator, not managed simulators or QPUs.

The workaround parses with Qiskit, transpiles to {h, cx, rz, x}, and emits
Braket-native names (cx as cnot). Native cnot avoids the faulty generic
controlled-gate path in the reproduced cases. Replication and isolation:
https://github.com/ramalhoneilson/quanifi-braket-phase-replication
"""
import re

# Universal basis whose every member is a native Braket gate (cx as cnot).
BRAKET_SAFE_BASIS = ["h", "cx", "rz", "x"]


def to_braket_qasm3(fmt, source):
    """Translate qasm2/qasm3 text into Braket-native OpenQASM 3.

    Returns (qasm3_src, translation_note). Raises ValueError on an
    unsupported format so callers can route to their failure relationship.
    """
    from qiskit import qasm2, qasm3, transpile

    if fmt == "qasm3":
        circuit = qasm3.loads(source)
        note = "qasm3 -> braket-native basis via qiskit"
    elif fmt == "qasm2":
        circuit = qasm2.loads(
            source, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        note = "qasm2 -> braket-native basis via qiskit"
    else:
        raise ValueError(
            "Unsupported circuit.format '{}'. Braket processors accept qasm3 "
            "or qasm2; set Output Format on the upstream circuit "
            "processor.".format(fmt))

    circuit.remove_final_measurements()
    circuit = transpile(
        circuit, basis_gates=BRAKET_SAFE_BASIS, optimization_level=0)
    src = qasm3.dumps(circuit)
    src = src.replace('include "stdgates.inc";', "")
    # Global phase is unobservable in counts; Braket rejects gphase outside
    # gate bodies, so drop any top-level statement the exporter emitted.
    src = re.sub(r"(?m)^\s*gphase\(.*?\);\s*$", "", src)
    src = re.sub(r"\bcx\b", "cnot", src)
    return src, note


def ensure_full_register_measure(qasm3_src):
    """Braket simulates only the qubits a program touches, so idle qubits
    silently vanish from the counts ('010' becomes '1'). Differential
    comparison needs every simulator to report the declared register width,
    so when the program has no measure statements append a register-level
    measure for each declared qubit register."""
    if re.search(r"\bmeasure\b", qasm3_src):
        return qasm3_src
    lines = [qasm3_src]
    for size, name in re.findall(r"qubit\[(\d+)\]\s+(\w+)\s*;", qasm3_src):
        lines.append(
            "bit[{0}] __meas_{1};\n__meas_{1} = measure {1};".format(size, name))
    return "\n".join(lines)
