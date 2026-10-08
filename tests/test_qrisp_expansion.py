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

import json
import math
import numpy as np
import pytest
from qiskit import qasm2
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from nifi_extensions.QrispPhaseOracle import QrispPhaseOracle
from nifi_extensions.QrispGroverOperator import QrispGroverOperator
from nifi_extensions.QrispAmplitudeAmplification import QrispAmplitudeAmplification
from nifi_extensions.QrispTeleportation import QrispTeleportation
from nifi_extensions.QrispStatevectorSimulator import QrispStatevectorSimulator
from nifi_extensions.QrispExpectation import QrispExpectation
from nifi_extensions.QiskitGroverOperator import QiskitGroverOperator


class MockFlowFile:
    def __init__(self, contents=b"", attributes=None):
        self._contents = contents if isinstance(contents, bytes) else contents.encode("utf-8")
        self._attributes = attributes or {}

    def getContentsAsBytes(self):
        return self._contents

    def getAttribute(self, key):
        return self._attributes.get(key)


class MockProcessContext:
    def __init__(self, properties=None):
        self._properties = properties or {}

    def getProperty(self, descriptor):
        val = self._properties.get(descriptor.name, descriptor.default_value)
        return MockPropertyValue(val)


class MockPropertyValue:
    def __init__(self, value):
        self._value = value

    def evaluateAttributeExpressions(self, flowfile=None):
        return self

    def getValue(self):
        return self._value


# ==============================================================================
# 1. QrispPhaseOracle Tests
# ==============================================================================
class TestQrispPhaseOracle:
    def test_standalone_oracle_diagonal(self):
        proc = QrispPhaseOracle()
        ctx = MockProcessContext({"Marked State": "110", "Output Format": "qasm2"})
        ff = MockFlowFile()

        result = proc.transform(ctx, ff)
        assert result.relationship == "success"
        assert result.attributes["circuit.format"] == "qasm2"
        assert result.attributes["circuit.marked_state"] == "110"
        assert result.attributes["circuit.num_qubits"] == "3"

        # Verify mathematical oracle action
        qc = qasm2.loads(result.contents.decode("utf-8"))
        op = Operator(qc).data
        diag = np.diagonal(op)
        # In Qiskit, index for q0=1, q1=1, q2=0 (LSB=q0) is index 3 (binary '011')
        ref_phase = diag[0]  # baseline phase
        for i, val in enumerate(diag):
            ratio = val / ref_phase
            if i == 3:  # 110 in q0-left
                assert np.isclose(ratio, -1.0, atol=1e-5), f"Expected -1 at index 3, got {ratio}"
            else:
                assert np.isclose(ratio, 1.0, atol=1e-5), f"Expected +1 at index {i}, got {ratio}"

    def test_compose_mode_appends_oracle(self):
        proc = QrispPhaseOracle()
        ctx = MockProcessContext({"Marked State": "10", "Output Format": "qasm2"})
        # Upstream circuit with H on both qubits
        h_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\nh q[1];\n'
        ff = MockFlowFile(h_qasm, {"circuit.format": "qasm2"})

        result = proc.transform(ctx, ff)
        assert result.relationship == "success"
        qc = qasm2.loads(result.contents.decode("utf-8"))
        assert qc.num_qubits == 2
        # Check that H gates exist
        ops = qc.count_ops()
        assert ops.get("h", 0) >= 2

    def test_invalid_marked_state_routes_to_failure(self):
        proc = QrispPhaseOracle()
        ctx = MockProcessContext({"Marked State": "102"})  # invalid non-binary
        ff = MockFlowFile()
        result = proc.transform(ctx, ff)
        assert result.relationship == "failure"
        assert "circuit.error" in result.attributes


# ==============================================================================
# 2. QrispGroverOperator Tests
# ==============================================================================
class TestQrispGroverOperator:
    def test_grover_operator_missing_input_fails(self):
        proc = QrispGroverOperator()
        ctx = MockProcessContext({"Num Iterations": "1"})
        ff = MockFlowFile()  # no circuit
        result = proc.transform(ctx, ff)
        assert result.relationship == "failure"

    def test_qrisp_oracle_to_qrisp_grover_operator(self):
        # 1. Oracle for 110
        oracle_proc = QrispPhaseOracle()
        o_ctx = MockProcessContext({"Marked State": "110", "Output Format": "qasm2"})
        o_res = oracle_proc.transform(o_ctx, MockFlowFile())

        # 2. Grover operator for 2 iterations
        grover_proc = QrispGroverOperator()
        g_ctx = MockProcessContext({"Num Iterations": "2", "Output Format": "qasm2"})
        g_ff = MockFlowFile(o_res.contents, o_res.attributes)
        g_res = grover_proc.transform(g_ctx, g_ff)

        assert g_res.relationship == "success"
        assert g_res.attributes["circuit.num_iterations"] == "2"
        assert g_res.attributes["circuit.marked_state"] == "110"

        # Simulate with Aer:
        qc = qasm2.loads(g_res.contents.decode("utf-8"))
        qc.measure_all()
        counts = AerSimulator().run(qc, shots=1024).result().get_counts()
        counts_q0 = {k[::-1]: v for k, v in counts.items()}
        top_state = max(counts_q0, key=counts_q0.get)
        assert top_state == "110"
        assert counts_q0["110"] > 900  # Grover peak > 90%

    def test_cross_framework_qrisp_oracle_to_qiskit_grover(self):
        # QrispPhaseOracle -> QiskitGroverOperator
        oracle_proc = QrispPhaseOracle()
        o_ctx = MockProcessContext({"Marked State": "110", "Output Format": "qasm2"})
        o_res = oracle_proc.transform(o_ctx, MockFlowFile())

        qiskit_grover = QiskitGroverOperator()
        q_ctx = MockProcessContext({"Num Iterations": "2", "Output Format": "qasm3"})
        q_ff = MockFlowFile(o_res.contents, o_res.attributes)
        q_res = qiskit_grover.transform(q_ctx, q_ff)

        assert q_res.relationship == "success"


# ==============================================================================
# 3. QrispAmplitudeAmplification Tests
# ==============================================================================
class TestQrispAmplitudeAmplification:
    def test_standalone_amplitude_amplification(self):
        proc = QrispAmplitudeAmplification()
        ctx = MockProcessContext({"Marked State": "101", "Num Iterations": "2"})
        ff = MockFlowFile()

        result = proc.transform(ctx, ff)
        assert result.relationship == "success"
        assert result.attributes["circuit.format"] == "qasm2"
        assert result.attributes["circuit.marked_state"] == "101"

        qc = qasm2.loads(result.contents.decode("utf-8"))
        qc.measure_all()
        counts = AerSimulator().run(qc, shots=1024).result().get_counts()
        counts_q0 = {k[::-1]: v for k, v in counts.items()}
        top_state = max(counts_q0, key=counts_q0.get)
        assert top_state == "101"
        assert counts_q0["101"] > 900


# ==============================================================================
# 4. QrispTeleportation Tests
# ==============================================================================
class TestQrispTeleportation:
    def test_teleportation_fidelity_ideal(self):
        proc = QrispTeleportation()
        ctx = MockProcessContext({
            "Theta": str(math.pi / 3),
            "Phi": str(math.pi / 4),
            "Shots": "1024",
        })
        ff = MockFlowFile()

        result = proc.transform(ctx, ff)
        assert result.relationship == "success"
        assert result.attributes["circuit.format"] == "qasm2"
        assert result.attributes["circuit.num_qubits"] == "3"
        fidelity = float(result.attributes["teleport.fidelity"])
        assert np.isclose(fidelity, 1.0, atol=1e-4)


# ==============================================================================
# 5. QrispStatevectorSimulator Tests
# ==============================================================================
class TestQrispStatevectorSimulator:
    def test_statevector_bell_pair(self, tmp_path):
        bell_qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'
        )
        proc = QrispStatevectorSimulator()
        ctx = MockProcessContext({
            "Reports Directory": str(tmp_path),
            "Flow Name": "test-bell",
        })
        ff = MockFlowFile(bell_qasm)

        result = proc.transform(ctx, ff)
        assert result.relationship == "success"
        assert result.attributes["sim.bit_order"] == "q0_left"
        assert result.attributes["sim.qubit_count"] == "2"

        dist = json.loads(result.contents.decode("utf-8"))
        assert np.isclose(dist["00"], 0.5, atol=1e-5)
        assert np.isclose(dist["11"], 0.5, atol=1e-5)
        assert "01" not in dist or dist["01"] < 1e-5
        assert "10" not in dist or dist["10"] < 1e-5

        # Check HTML card generation
        report_file = tmp_path / "test-bell-statevector.html"
        assert report_file.exists()
        html = report_file.read_text(encoding="utf-8")
        assert "Statevector &amp; Probabilities" in html


# ==============================================================================
# 6. QrispExpectation Tests
# ==============================================================================
class TestQrispExpectation:
    def test_exact_expectation_bell_z0z1(self):
        bell_qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'
        )
        proc = QrispExpectation()
        ctx = MockProcessContext({"Hamiltonian": "Z0 Z1", "Mode": "exact"})
        ff = MockFlowFile(bell_qasm)

        result = proc.transform(ctx, ff)
        assert result.relationship == "success"
        exp_val = float(result.attributes["sim.expectation"])
        assert np.isclose(exp_val, 1.0, atol=1e-5)
        assert np.isclose(float(result.attributes["sim.variance"]), 0.0, atol=1e-5)

    def test_exact_expectation_superposition_x0(self):
        h_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n'
        proc = QrispExpectation()
        ctx = MockProcessContext({"Hamiltonian": "X0", "Mode": "exact"})
        ff = MockFlowFile(h_qasm)

        result = proc.transform(ctx, ff)
        assert result.relationship == "success"
        exp_val = float(result.attributes["sim.expectation"])
        assert np.isclose(exp_val, 1.0, atol=1e-5)

    def test_sampled_mode_runs_successfully(self):
        bell_qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n'
        )
        proc = QrispExpectation()
        ctx = MockProcessContext({"Hamiltonian": "Z0", "Mode": "sampled", "Shots": "2048"})
        ff = MockFlowFile(bell_qasm)

        result = proc.transform(ctx, ff)
        assert result.relationship == "success"
        body = json.loads(result.contents.decode("utf-8"))
        assert body["mode"] == "sampled"
        assert np.isclose(body["expectation"], 0.0, atol=0.15)  # E[Z0] for Bell pair is 0
