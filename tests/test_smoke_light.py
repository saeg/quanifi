"""Light smoke test suite for Quanifi processors.

Designed for CI/CD runs on every push: exercises key representative components
across all supported quantum frameworks (Qiskit, Cirq, PennyLane, Qrisp, PyQuil)
and pipeline stages (circuit generation, simulation, QAOA solving, Hamiltonian encoding,
mutation testing, and reporting) within seconds.
"""

import json
import tempfile
import pytest

from conftest import MockContext, MockFlowFile, result_to_flowfile

pytestmark = pytest.mark.smoke_light


class TestKeyCircuitBuilders:
    """Smoke tests for primary circuit builders across frameworks."""

    def test_qiskit_builders(self):
        from QiskitGroverCircuit import QiskitGroverCircuit
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QiskitBellState import QiskitBellState

        ctx = MockContext(**{"Marked State": "10", "Num Iterations": "1"})
        res_g = QiskitGroverCircuit().transform(ctx, MockFlowFile())
        assert res_g.relationship == "success"
        assert res_g.attributes["circuit.format"] in ("qasm2", "qasm3")
        assert len(res_g.contents) > 0

        res_h = QiskitHadamardTransform().transform(MockContext(), MockFlowFile())
        assert res_h.relationship == "success"

        res_b = QiskitBellState().transform(MockContext(), MockFlowFile())
        assert res_b.relationship == "success"

    def test_cirq_builders(self):
        from CirqGroverCircuit import CirqGroverCircuit
        from CirqHadamardTransform import CirqHadamardTransform

        ctx = MockContext(**{"Marked State": "10", "Num Iterations": "1"})
        res_g = CirqGroverCircuit().transform(ctx, MockFlowFile())
        assert res_g.relationship == "success"
        assert res_g.attributes["circuit.format"] == "cirq_json"

        res_h = CirqHadamardTransform().transform(MockContext(), MockFlowFile())
        assert res_h.relationship == "success"

    def test_pennylane_builders(self):
        from PennylaneGroverCircuit import PennylaneGroverCircuit
        from PennylaneQuantumArithmetic import PennylaneQuantumArithmetic

        res_g = PennylaneGroverCircuit().transform(MockContext(), MockFlowFile())
        assert res_g.relationship == "success"
        assert res_g.attributes["circuit.format"] == "qasm2"

        res_a = PennylaneQuantumArithmetic().transform(MockContext(), MockFlowFile())
        assert res_a.relationship == "success"

    def test_qrisp_builders(self):
        from QrispGroverCircuit import QrispGroverCircuit
        from QrispHadamardTransform import QrispHadamardTransform
        from QrispTypeEncoder import QrispTypeEncoder

        ctx = MockContext(**{"Marked State": "10", "Num Iterations": "1"})
        res_g = QrispGroverCircuit().transform(ctx, MockFlowFile())
        assert res_g.relationship == "success"

        res_h = QrispHadamardTransform().transform(MockContext(), MockFlowFile())
        assert res_h.relationship == "success"

        res_t = QrispTypeEncoder().transform(MockContext(), MockFlowFile())
        assert res_t.relationship == "success"

    def test_pyquil_builders(self):
        from PyquilGroverCircuit import PyquilGroverCircuit

        ctx = MockContext(**{"Marked State": "10", "Num Iterations": "1"})
        res_g = PyquilGroverCircuit().transform(ctx, MockFlowFile())
        assert res_g.relationship == "success"
        assert res_g.attributes["circuit.format"] in ("qasm2", "quil")


class TestKeySimulators:
    """Smoke tests for simulation engines across frameworks."""

    def test_qiskit_aer_simulator(self):
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QiskitAerSimulator import QiskitAerSimulator

        c = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        res = QiskitAerSimulator().transform(MockContext(**{"Shots": "128"}), c)
        assert res.relationship == "success"
        counts = json.loads(res.contents)
        assert sum(counts.values()) == 128
        assert res.attributes["sim.framework"] == "qiskit"

    def test_cirq_simulator(self):
        from CirqHadamardTransform import CirqHadamardTransform
        from CirqSimulator import CirqSimulator

        c = result_to_flowfile(CirqHadamardTransform().transform(MockContext(), MockFlowFile()))
        res = CirqSimulator().transform(MockContext(**{"Shots": "100"}), c)
        assert res.relationship == "success"
        counts = json.loads(res.contents)
        assert sum(counts.values()) == 100
        assert res.attributes["sim.framework"] == "cirq"

    def test_pennylane_simulator(self):
        from PennylaneGroverCircuit import PennylaneGroverCircuit
        from PennylaneSimulator import PennylaneSimulator

        c = result_to_flowfile(PennylaneGroverCircuit().transform(MockContext(), MockFlowFile()))
        res = PennylaneSimulator().transform(MockContext(**{"Shots": "128"}), c)
        assert res.relationship == "success"
        counts = json.loads(res.contents)
        assert sum(counts.values()) == 128

    def test_qrisp_simulator(self):
        from QrispHadamardTransform import QrispHadamardTransform
        from QrispSimulator import QrispSimulator

        c = result_to_flowfile(QrispHadamardTransform().transform(MockContext(), MockFlowFile()))
        res = QrispSimulator().transform(MockContext(**{"Shots": "64"}), c)
        assert res.relationship == "success"
        counts = json.loads(res.contents)
        assert sum(counts.values()) == 64

    def test_pyquil_simulator(self):
        from PyquilGroverCircuit import PyquilGroverCircuit
        from PyquilSimulator import PyquilSimulator

        c = result_to_flowfile(PyquilGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()))
        res = PyquilSimulator().transform(MockContext(**{"Shots": "64"}), c)
        assert res.relationship == "success"
        counts = json.loads(res.contents)
        assert sum(counts.values()) == 64


class TestKeySolvers:
    """Smoke tests for QAOA and Search algorithms."""

    def test_qaoa_cross_framework(self):
        from QiskitHamiltonian import QiskitHamiltonian
        from QiskitQAOA import QiskitQAOA
        from CirqQAOA import CirqQAOA
        from PennylaneQAOA import PennylaneQAOA
        from PyquilQAOA import PyquilQAOA
        from QrispQAOA import QrispQAOA

        triangle = "0.5 Z0 Z1 + 0.5 Z1 Z2 + 0.5 Z0 Z2"
        h_ff = result_to_flowfile(
            QiskitHamiltonian().transform(MockContext(**{"Hamiltonian": triangle}), MockFlowFile())
        )

        for solver_cls in [QiskitQAOA, CirqQAOA, PennylaneQAOA, PyquilQAOA, QrispQAOA]:
            res = solver_cls().transform(MockContext(**{"Max Iterations": "1"}), h_ff)
            assert res.relationship == "success", f"{solver_cls.__name__} failed: {res.attributes}"
            assert res.attributes.get("qaoa.optimal_value") != ""

    def test_grover_search(self):
        from QiskitGroverSearch import QiskitGroverSearch

        res = QiskitGroverSearch().transform(MockContext(**{"Marked State": "01"}), MockFlowFile())
        assert res.relationship == "success"
        assert res.attributes["grover.top_result"] == "01"
        counts = json.loads(res.contents)
        assert "01" in counts


class TestKeyHamiltoniansAndEncoders:
    """Smoke tests for problem encoders and molecular Hamiltonians."""

    def test_problem_encoders(self):
        from MaxCutProblem import MaxCutProblem
        from MoleculeHamiltonian import MoleculeHamiltonian

        res_mc = MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile())
        assert res_mc.relationship == "success"
        assert res_mc.attributes["hamiltonian.format"] == "sparse_pauli_op_json"

        res_mol = MoleculeHamiltonian().transform(MockContext(), MockFlowFile())
        assert res_mol.relationship == "success"
        assert res_mol.attributes["hamiltonian.molecule"] == "H2"
        assert res_mol.attributes["hamiltonian.format"] == "sparse_pauli_op_json"


class TestKeyTestingInfrastructure:
    """Smoke tests for mutation testing, assertion, and unitary metrics."""

    def test_mutator_and_assertion(self):
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QuantumMutator import QuantumMutator
        from QuantumAssertion import QuantumAssertion
        from QuanifiUnitary import QuanifiUnitary

        c = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))

        # Mutation
        res_m = QuantumMutator().transform(MockContext(**{"Mutation Operator": "bitflip"}), c)
        assert res_m.relationship == "success"
        assert res_m.attributes["mut.applied"] in ("true", "false")

        # Unitary calculation
        res_u = QuanifiUnitary().transform(MockContext(**{"Max Qubits": "4"}), c)
        assert res_u.relationship == "success"
        assert res_u.attributes["unitary.dimension"] == "4"

        # Assertion
        with tempfile.TemporaryDirectory() as td:
            counts_ff = MockFlowFile(
                content=json.dumps({"00": 50, "11": 50}),
                attributes={"case_id": "c1", "partition": "p1", "marked_state": "00"}
            )
            ctx = MockContext(**{"Reports Directory": td, "Flow Name": "test_flow", "Check Ground Truth": "false"})
            res_a = QuantumAssertion().transform(ctx, counts_ff)
            assert res_a.relationship in ("pass", "fail")


class TestKeyReporting:
    """Smoke tests for report generation."""

    def test_circuit_report(self):
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QiskitCircuitReport import QiskitCircuitReport

        c = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        with tempfile.TemporaryDirectory() as td:
            res = QiskitCircuitReport().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "test_flow"}), c
            )
            assert res.relationship == "success"
            assert res.attributes["report.flow_name"] == "test_flow"
            assert "report.path" in res.attributes
