"""Comprehensive smoke test suite exercising every Quanifi processor at least once.

Covers all 122 processor classes across all 9 architectural categories:
  - Circuit builders (48)
  - Simulators & expectation engines (13)
  - Algorithms & all-in-one solvers (20)
  - Hamiltonians & problem encoders (9)
  - Differential & mutation-testing infrastructure (14)
  - PennyLane QML lane (3)
  - Reporting & utility (3)
  - Hardware execution gateways (3)
  - Hardware batch lane submitters & pollers (9)

Guarantees 100% component smoke coverage offline with fast parameters.
A dynamic completeness assertion validates that every processor discovered by
tools/add_all_components_group.py is exercised by this suite.
"""

import datetime
import importlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "nifi_extensions"))

import add_all_components_group as tool
from conftest import (
    MockContext,
    MockFlowFile,
    result_to_flowfile,
    result_to_flowfile_merged,
)

pytestmark = pytest.mark.smoke_comprehensive

# Global set to track all processor class names exercised across the tests
EXERCISED_PROCESSORS = set()


def _mark(name):
    """Register a processor class name as exercised in this suite."""
    EXERCISED_PROCESSORS.add(name)


def _qasm_ff(qasm='OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n', fmt="qasm2"):
    return MockFlowFile(content=qasm.encode(), attributes={"circuit.format": fmt})


# ===========================================================================
# 1. Circuit Builders (48 processors)
# ===========================================================================


class TestCircuitBuilders:
    """Exercising every circuit builder processor."""

    def test_qiskit_builders(self):
        from QiskitAmplitudeAmplification import QiskitAmplitudeAmplification
        from QiskitAnsatz import QiskitAnsatz
        from QiskitBellState import QiskitBellState
        from QiskitGroverCircuit import QiskitGroverCircuit
        from QiskitGroverOperator import QiskitGroverOperator
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QiskitPhaseEstimation import QiskitPhaseEstimation
        from QiskitQAOACircuit import QiskitQAOACircuit
        from QiskitQFTCircuit import QiskitQFTCircuit
        from QiskitQuantumArithmetic import QiskitQuantumArithmetic
        from QiskitQuantumHelloWorld import QiskitQuantumHelloWorld
        from QiskitStatePreparation import QiskitStatePreparation
        from QiskitTeleportation import QiskitTeleportation
        from MaxCutProblem import MaxCutProblem

        # Base transforms
        for name, cls in [
            ("QiskitAmplitudeAmplification", QiskitAmplitudeAmplification),
            ("QiskitAnsatz", QiskitAnsatz),
            ("QiskitBellState", QiskitBellState),
            ("QiskitHadamardTransform", QiskitHadamardTransform),
            ("QiskitPhaseEstimation", QiskitPhaseEstimation),
            ("QiskitQFTCircuit", QiskitQFTCircuit),
            ("QiskitQuantumArithmetic", QiskitQuantumArithmetic),
            ("QiskitQuantumHelloWorld", QiskitQuantumHelloWorld),
            ("QiskitStatePreparation", QiskitStatePreparation),
            ("QiskitTeleportation", QiskitTeleportation),
        ]:
            _mark(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        # Grover circuit & operator
        _mark("QiskitGroverCircuit")
        res_gc = QiskitGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()
        )
        assert res_gc.relationship == "success"

        _mark("QiskitGroverOperator")
        h_ff = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_go = QiskitGroverOperator().transform(MockContext(), h_ff)
        assert res_go.relationship == "success"

        # QAOA Circuit
        _mark("QiskitQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = QiskitQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_cirq_builders(self):
        from CirqAnsatz import CirqAnsatz
        from CirqGroverCircuit import CirqGroverCircuit
        from CirqGroverOperator import CirqGroverOperator
        from CirqHadamardTransform import CirqHadamardTransform
        from CirqPhaseEstimation import CirqPhaseEstimation
        from CirqQAOACircuit import CirqQAOACircuit
        from CirqQFTCircuit import CirqQFTCircuit
        from CirqQuantumArithmetic import CirqQuantumArithmetic
        from CirqStatePreparation import CirqStatePreparation
        from MaxCutProblem import MaxCutProblem

        for name, cls in [
            ("CirqAnsatz", CirqAnsatz),
            ("CirqHadamardTransform", CirqHadamardTransform),
            ("CirqPhaseEstimation", CirqPhaseEstimation),
            ("CirqQFTCircuit", CirqQFTCircuit),
            ("CirqQuantumArithmetic", CirqQuantumArithmetic),
            ("CirqStatePreparation", CirqStatePreparation),
        ]:
            _mark(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        _mark("CirqGroverCircuit")
        res_gc = CirqGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()
        )
        assert res_gc.relationship == "success"

        _mark("CirqGroverOperator")
        h_ff = result_to_flowfile(CirqHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_go = CirqGroverOperator().transform(MockContext(), h_ff)
        assert res_go.relationship == "success"

        _mark("CirqQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = CirqQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_pennylane_builders(self):
        from PennylaneGroverCircuit import PennylaneGroverCircuit
        from PennylanePhaseEstimation import PennylanePhaseEstimation
        from PennylaneQAOACircuit import PennylaneQAOACircuit
        from PennylaneQuantumArithmetic import PennylaneQuantumArithmetic
        from PennylaneVariationalAnsatz import PennylaneVariationalAnsatz
        from MaxCutProblem import MaxCutProblem

        for name, cls in [
            ("PennylaneGroverCircuit", PennylaneGroverCircuit),
            ("PennylanePhaseEstimation", PennylanePhaseEstimation),
            ("PennylaneQuantumArithmetic", PennylaneQuantumArithmetic),
            ("PennylaneVariationalAnsatz", PennylaneVariationalAnsatz),
        ]:
            _mark(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        _mark("PennylaneQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = PennylaneQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_pyquil_builders(self):
        from PyquilAnsatz import PyquilAnsatz
        from PyquilGroverCircuit import PyquilGroverCircuit
        from PyquilGroverOperator import PyquilGroverOperator
        from PyquilQAOACircuit import PyquilQAOACircuit
        from PyquilQFT import PyquilQFT
        from MaxCutProblem import MaxCutProblem

        for name, cls in [
            ("PyquilAnsatz", PyquilAnsatz),
            ("PyquilQFT", PyquilQFT),
        ]:
            _mark(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        _mark("PyquilGroverCircuit")
        res_gc = PyquilGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()
        )
        assert res_gc.relationship == "success"

        _mark("PyquilGroverOperator")
        p_ff = result_to_flowfile(
            PyquilGroverCircuit().transform(MockContext(**{"Marked State": "10", "Num Iterations": "0"}), MockFlowFile())
        )
        res_go = PyquilGroverOperator().transform(MockContext(), p_ff)
        assert res_go.relationship == "success"

        _mark("PyquilQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = PyquilQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_qrisp_builders(self):
        from QrispAmplitudeAmplification import QrispAmplitudeAmplification
        from QrispAnsatz import QrispAnsatz
        from QrispGroverCircuit import QrispGroverCircuit
        from QrispGroverOperator import QrispGroverOperator
        from QrispHadamardTransform import QrispHadamardTransform
        from QrispPhaseEstimation import QrispPhaseEstimation
        from QrispQAOACircuit import QrispQAOACircuit
        from QrispQCCSDAnsatz import QrispQCCSDAnsatz
        from QrispQFTCircuit import QrispQFTCircuit
        from QrispQuantumArithmetic import QrispQuantumArithmetic
        from QrispStatePreparation import QrispStatePreparation
        from QrispTeleportation import QrispTeleportation
        from QrispTruthTableSynthesis import QrispTruthTableSynthesis
        from QrispTypeEncoder import QrispTypeEncoder
        from MaxCutProblem import MaxCutProblem

        for name, cls in [
            ("QrispAmplitudeAmplification", QrispAmplitudeAmplification),
            ("QrispAnsatz", QrispAnsatz),
            ("QrispHadamardTransform", QrispHadamardTransform),
            ("QrispPhaseEstimation", QrispPhaseEstimation),
            ("QrispQCCSDAnsatz", QrispQCCSDAnsatz),
            ("QrispQFTCircuit", QrispQFTCircuit),
            ("QrispQuantumArithmetic", QrispQuantumArithmetic),
            ("QrispStatePreparation", QrispStatePreparation),
            ("QrispTeleportation", QrispTeleportation),
            ("QrispTruthTableSynthesis", QrispTruthTableSynthesis),
            ("QrispTypeEncoder", QrispTypeEncoder),
        ]:
            _mark(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        _mark("QrispGroverCircuit")
        res_gc = QrispGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()
        )
        assert res_gc.relationship == "success"

        _mark("QrispGroverOperator")
        qr_h = result_to_flowfile(QrispHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_go = QrispGroverOperator().transform(MockContext(), qr_h)
        assert res_go.relationship == "success"

        _mark("QrispQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = QrispQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_evaluator_and_baseline(self):
        from ReadoutBaselineCircuit import ReadoutBaselineCircuit
        from QuantumQAOAEvaluator import QuantumQAOAEvaluator

        _mark("ReadoutBaselineCircuit")
        res_rb = ReadoutBaselineCircuit().transform(MockContext(), MockFlowFile())
        assert res_rb.relationship == "success"

        _mark("QuantumQAOAEvaluator")
        ff = MockFlowFile(
            content=json.dumps({"01": 500, "10": 500}),
            attributes={
                "sim.bit_order": "q0_left",
                "builder.component": "X",
                "sim.component": "Y",
                "hamiltonian.json": json.dumps({"num_qubits": 2, "terms": [["ZZ", 1.0, 0.0]]}),
            },
        )
        res_qe = QuantumQAOAEvaluator().transform(MockContext(**{"Hamiltonian": "Z0 Z1"}), ff)
        assert res_qe.relationship == "success", res_qe.attributes


# ===========================================================================
# 2. Simulators & Expectation Engines (13 processors)
# ===========================================================================


class TestSimulators:
    """Exercising all 13 quantum simulators and expectation evaluators."""

    def test_simulators(self):
        from QiskitAerSimulator import QiskitAerSimulator
        from QiskitStatevectorSimulator import QiskitStatevectorSimulator
        from CirqSimulator import CirqSimulator
        from CirqStatevectorSimulator import CirqStatevectorSimulator
        from PennylaneSimulator import PennylaneSimulator
        from PyquilSimulator import PyquilSimulator
        from QrispSimulator import QrispSimulator
        from QrispStatevectorSimulator import QrispStatevectorSimulator
        from BraketSimulator import BraketSimulator
        from QSharpSimulator import QSharpSimulator

        from QiskitHadamardTransform import QiskitHadamardTransform
        from CirqHadamardTransform import CirqHadamardTransform
        from PennylaneGroverCircuit import PennylaneGroverCircuit
        from PyquilGroverCircuit import PyquilGroverCircuit
        from QrispHadamardTransform import QrispHadamardTransform

        qiskit_ff = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        cirq_ff = result_to_flowfile(CirqHadamardTransform().transform(MockContext(), MockFlowFile()))
        pl_ff = result_to_flowfile(PennylaneGroverCircuit().transform(MockContext(), MockFlowFile()))
        pyquil_ff = result_to_flowfile(PyquilGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()))
        qrisp_ff = result_to_flowfile(QrispHadamardTransform().transform(MockContext(), MockFlowFile()))

        sim_targets = [
            ("QiskitAerSimulator", QiskitAerSimulator, qiskit_ff, {"Shots": "64"}),
            ("QiskitStatevectorSimulator", QiskitStatevectorSimulator, qiskit_ff, {}),
            ("CirqSimulator", CirqSimulator, cirq_ff, {"Shots": "64"}),
            ("CirqStatevectorSimulator", CirqStatevectorSimulator, cirq_ff, {}),
            ("PennylaneSimulator", PennylaneSimulator, pl_ff, {"Shots": "64"}),
            ("PyquilSimulator", PyquilSimulator, pyquil_ff, {"Shots": "64"}),
            ("QrispSimulator", QrispSimulator, qrisp_ff, {"Shots": "64"}),
            ("QrispStatevectorSimulator", QrispStatevectorSimulator, qrisp_ff, {}),
            ("BraketSimulator", BraketSimulator, qiskit_ff, {"Shots": "64"}),
            ("QSharpSimulator", QSharpSimulator, qiskit_ff, {"Shots": "64"}),
        ]

        for name, cls, ff, props in sim_targets:
            _mark(name)
            res = cls().transform(MockContext(**props), ff)
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

    def test_expectation_evaluators(self):
        from PennylaneFeatureEmbedding import PennylaneFeatureEmbedding
        from PennylaneExpectation import PennylaneExpectation
        from PyquilGroverCircuit import PyquilGroverCircuit
        from PyquilExpectation import PyquilExpectation
        from QrispHadamardTransform import QrispHadamardTransform
        from QrispExpectation import QrispExpectation

        # PennylaneExpectation
        _mark("PennylaneExpectation")
        emb_ff = result_to_flowfile(
            PennylaneFeatureEmbedding().transform(
                MockContext(**{"Embedding": "angle", "Features": "[0.1, 0.2]"}), MockFlowFile()
            )
        )
        res_pe = PennylaneExpectation().transform(MockContext(), emb_ff)
        assert res_pe.relationship == "success"

        # PyquilExpectation
        _mark("PyquilExpectation")
        pyq_ff = result_to_flowfile(
            PyquilGroverCircuit().transform(MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile())
        )
        res_pye = PyquilExpectation().transform(MockContext(), pyq_ff)
        assert res_pye.relationship == "success"

        # QrispExpectation
        _mark("QrispExpectation")
        qr_ff = result_to_flowfile(QrispHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_qre = QrispExpectation().transform(MockContext(), qr_ff)
        assert res_qre.relationship == "success"


# ===========================================================================
# 3. Algorithms & Solvers (20 processors)
# ===========================================================================


class TestAlgorithmsAndSolvers:
    """Exercising all 20 solver and algorithm processors."""

    def test_amplitude_estimation(self):
        from CirqAmplitudeEstimation import CirqAmplitudeEstimation
        from PennylaneAmplitudeEstimation import PennylaneAmplitudeEstimation
        from QiskitAmplitudeEstimation import QiskitAmplitudeEstimation
        from QrispAmplitudeEstimation import QrispAmplitudeEstimation

        for name, cls, props in [
            ("CirqAmplitudeEstimation", CirqAmplitudeEstimation, {"Evaluation Qubits": "2"}),
            ("PennylaneAmplitudeEstimation", PennylaneAmplitudeEstimation, {"Evaluation Qubits": "2"}),
            ("QiskitAmplitudeEstimation", QiskitAmplitudeEstimation, {"Evaluation Qubits": "2"}),
            ("QrispAmplitudeEstimation", QrispAmplitudeEstimation, {"Precision": "2"}),
        ]:
            _mark(name)
            res = cls().transform(MockContext(**props), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

    def test_grover_search(self):
        from QiskitGroverSearch import QiskitGroverSearch
        from QrispGroverSearch import QrispGroverSearch

        for name, cls in [
            ("QiskitGroverSearch", QiskitGroverSearch),
            ("QrispGroverSearch", QrispGroverSearch),
        ]:
            _mark(name)
            res = cls().transform(MockContext(**{"Marked State": "01"}), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

    def test_qrisp_algorithms(self):
        from QrispBernsteinVazirani import QrispBernsteinVazirani
        from QrispDeutschJozsa import QrispDeutschJozsa
        from QrispShor import QrispShor
        from QrispSwapTest import QrispSwapTest

        for name, cls, props in [
            ("QrispBernsteinVazirani", QrispBernsteinVazirani, {"Secret String": "10"}),
            ("QrispDeutschJozsa", QrispDeutschJozsa, {"Function Type": "constant_zero"}),
            ("QrispShor", QrispShor, {"Target": "15"}),
            ("QrispSwapTest", QrispSwapTest, {}),
        ]:
            _mark(name)
            res = cls().transform(MockContext(**props), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

    def test_qaoa_solvers(self):
        from QiskitHamiltonian import QiskitHamiltonian
        from CirqQAOA import CirqQAOA
        from PennylaneQAOA import PennylaneQAOA
        from PyquilQAOA import PyquilQAOA
        from QiskitQAOA import QiskitQAOA
        from QrispQAOA import QrispQAOA

        triangle = "0.5 Z0 Z1 + 0.5 Z1 Z2 + 0.5 Z0 Z2"
        h_ff = result_to_flowfile(
            QiskitHamiltonian().transform(MockContext(**{"Hamiltonian": triangle}), MockFlowFile())
        )

        for name, cls in [
            ("CirqQAOA", CirqQAOA),
            ("PennylaneQAOA", PennylaneQAOA),
            ("PyquilQAOA", PyquilQAOA),
            ("QiskitQAOA", QiskitQAOA),
            ("QrispQAOA", QrispQAOA),
        ]:
            _mark(name)
            res = cls().transform(MockContext(**{"Max Iterations": "1"}), h_ff)
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

    def test_vqe_solvers(self):
        from QiskitHamiltonian import QiskitHamiltonian
        from QiskitAnsatz import QiskitAnsatz
        from QiskitVQE import QiskitVQE

        from CirqHamiltonian import CirqHamiltonian
        from CirqAnsatz import CirqAnsatz
        from CirqVQE import CirqVQE

        from PyquilAnsatz import PyquilAnsatz
        from PyquilVQE import PyquilVQE

        from QrispHamiltonian import QrispHamiltonian
        from QrispAnsatz import QrispAnsatz
        from QrispVQE import QrispVQE

        # Qiskit VQE
        _mark("QiskitVQE")
        q_h = result_to_flowfile(QiskitHamiltonian().transform(MockContext(), MockFlowFile()))
        q_a = result_to_flowfile(QiskitAnsatz().transform(MockContext(), q_h))
        res_qv = QiskitVQE().transform(MockContext(**{"Max Iterations": "1"}), q_a)
        assert res_qv.relationship == "success"

        # Cirq VQE
        _mark("CirqVQE")
        c_h = result_to_flowfile(CirqHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0"}), MockFlowFile()))
        c_a = result_to_flowfile(CirqAnsatz().transform(MockContext(**{"Reps": "1"}), c_h))
        res_cv = CirqVQE().transform(MockContext(**{"Max Iterations": "1", "Shots": "32"}), c_a)
        assert res_cv.relationship == "success"

        # Pyquil VQE (with merged upstream flowfile for format retention)
        _mark("PyquilVQE")
        p_h = result_to_flowfile_merged(
            QiskitHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0 + Z1"}), MockFlowFile()),
            MockFlowFile()
        )
        p_a = result_to_flowfile_merged(
            PyquilAnsatz().transform(MockContext(**{"Output Mode": "attach"}), p_h),
            p_h
        )
        res_pv = PyquilVQE().transform(MockContext(**{"Max Iterations": "1", "Shots": "32"}), p_a)
        assert res_pv.relationship == "success"

        # Qrisp VQE
        _mark("QrispVQE")
        qr_h = result_to_flowfile(QrispHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0"}), MockFlowFile()))
        qr_a = result_to_flowfile(QrispAnsatz().transform(MockContext(**{"Reps": "1"}), qr_h))
        res_qrv = QrispVQE().transform(MockContext(**{"Max Iterations": "1"}), qr_a)
        assert res_qrv.relationship == "success"

    def test_qiro_solver(self):
        from MaxIndependentSetProblem import MaxIndependentSetProblem
        from QrispQIRO import QrispQIRO

        _mark("QrispQIRO")
        mis_ff = result_to_flowfile(
            MaxIndependentSetProblem().transform(MockContext(**{"Edges": "[[0, 1]]"}), MockFlowFile())
        )
        res = QrispQIRO().transform(
            MockContext(**{"Max Iterations": "1", "Recursions": "1", "Layers": "1"}), mis_ff
        )
        assert res.relationship == "success"


# ===========================================================================
# 4. Hamiltonians & Problem Encoders (9 processors)
# ===========================================================================


class TestHamiltoniansAndProblemEncoders:
    """Exercising all 9 Hamiltonian and problem encoder processors."""

    def test_hamiltonians_and_problem_encoders(self):
        from CirqHamiltonian import CirqHamiltonian
        from MaxCliqueProblem import MaxCliqueProblem
        from MaxCutProblem import MaxCutProblem
        from MaxIndependentSetProblem import MaxIndependentSetProblem
        from MoleculeHamiltonian import MoleculeHamiltonian
        from PortfolioRebalancingProblem import PortfolioRebalancingProblem
        from QiskitHamiltonian import QiskitHamiltonian
        from QrispHamiltonian import QrispHamiltonian
        from QuboToHamiltonian import QuboToHamiltonian

        encoders = [
            ("CirqHamiltonian", CirqHamiltonian, {}),
            ("MaxCliqueProblem", MaxCliqueProblem, {}),
            ("MaxCutProblem", MaxCutProblem, {}),
            ("MaxIndependentSetProblem", MaxIndependentSetProblem, {}),
            ("MoleculeHamiltonian", MoleculeHamiltonian, {}),
            ("PortfolioRebalancingProblem", PortfolioRebalancingProblem, {}),
            ("QiskitHamiltonian", QiskitHamiltonian, {}),
            ("QrispHamiltonian", QrispHamiltonian, {}),
            ("QuboToHamiltonian", QuboToHamiltonian, {"QUBO Matrix": "[[1, -1], [-1, 1]]"}),
        ]

        for name, cls, props in encoders:
            _mark(name)
            res = cls().transform(MockContext(**props), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"


# ===========================================================================
# 5. Differential & Mutation Testing Infrastructure (14 processors)
# ===========================================================================


class TestDifferentialAndMutation:
    """Exercising all 14 differential, mutation, and oracle processors."""

    def test_phase_oracles(self):
        from CirqHadamardTransform import CirqHadamardTransform
        from CirqPhaseOracle import CirqPhaseOracle
        from PyquilGroverCircuit import PyquilGroverCircuit
        from PyquilPhaseOracle import PyquilPhaseOracle
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QiskitPhaseOracle import QiskitPhaseOracle
        from QrispHadamardTransform import QrispHadamardTransform
        from QrispPhaseOracle import QrispPhaseOracle

        _mark("CirqPhaseOracle")
        c_h = result_to_flowfile(CirqHadamardTransform().transform(MockContext(), MockFlowFile()))
        assert CirqPhaseOracle().transform(MockContext(**{"Marked State": "01"}), c_h).relationship == "success"

        _mark("PyquilPhaseOracle")
        p_h = result_to_flowfile(
            PyquilGroverCircuit().transform(MockContext(**{"Marked State": "10", "Num Iterations": "0"}), MockFlowFile())
        )
        assert PyquilPhaseOracle().transform(MockContext(**{"Marked State": "10"}), p_h).relationship == "success"

        _mark("QiskitPhaseOracle")
        q_h = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        assert QiskitPhaseOracle().transform(MockContext(**{"Marked State": "01"}), q_h).relationship == "success"

        _mark("QrispPhaseOracle")
        qr_h = result_to_flowfile(QrispHadamardTransform().transform(MockContext(), MockFlowFile()))
        assert QrispPhaseOracle().transform(MockContext(**{"Marked State": "01"}), qr_h).relationship == "success"

    def test_mutator_and_testcase_source(self):
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QuantumMutator import QuantumMutator
        from QuantumTestCaseSource import QuantumTestCaseSource

        _mark("QuantumMutator")
        q_h = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_m = QuantumMutator().transform(MockContext(**{"Mutation Operator": "bitflip"}), q_h)
        assert res_m.relationship == "success"

        _mark("QuantumTestCaseSource")
        res_tcs = QuantumTestCaseSource().transform(MockContext(), MockFlowFile())
        assert res_tcs.relationship == "success"

    def test_unitaries_and_comparison(self):
        from QiskitHadamardTransform import QiskitHadamardTransform
        from CirqHadamardTransform import CirqHadamardTransform
        from QuanifiUnitary import QuanifiUnitary
        from QuantumUnitaryComparison import QuantumUnitaryComparison

        _mark("QuanifiUnitary")
        q_h = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        c_h = result_to_flowfile(CirqHadamardTransform().transform(MockContext(), MockFlowFile()))

        res_uq = QuanifiUnitary().transform(MockContext(**{"Max Qubits": "4"}), q_h)
        res_uc = QuanifiUnitary().transform(MockContext(**{"Max Qubits": "4"}), c_h)
        assert res_uq.relationship == "success"
        assert res_uc.relationship == "success"

        _mark("QuantumUnitaryComparison")
        with tempfile.TemporaryDirectory() as td:
            ctx = MockContext(**{
                "State Directory": td, "Comparison Label": "label-1",
                "Framework Label": "qiskit", "Tolerance": "1e-6"
            })
            cmp_proc = QuantumUnitaryComparison()
            # First unitary buffers (status waiting)
            r1 = cmp_proc.transform(ctx, result_to_flowfile(res_uq))
            assert r1.relationship == "failure" and r1.attributes["equiv.status"] == "waiting"
            # Second unitary completes comparison
            ctx2 = MockContext(**{
                "State Directory": td, "Comparison Label": "label-1",
                "Framework Label": "cirq", "Tolerance": "1e-6"
            })
            r2 = cmp_proc.transform(ctx2, result_to_flowfile(res_uc))
            assert r2.relationship == "success" and r2.attributes["equiv.status"] == "complete"

    def test_statistical_and_consensus_oracles(self):
        from QuantumAssertion import QuantumAssertion
        from QuantumCalibratedOracle import QuantumCalibratedOracle
        from QuantumConsensusOracle import QuantumConsensusOracle
        from QuantumDistributionComparison import QuantumDistributionComparison
        from QuantumDistributionOracle import QuantumDistributionOracle
        from QuantumSuccessProbabilityOracle import QuantumSuccessProbabilityOracle
        from MutationScoreReport import MutationScoreReport

        counts_content = json.dumps({"00": 50, "11": 50})

        # QuantumAssertion
        _mark("QuantumAssertion")
        with tempfile.TemporaryDirectory() as td:
            res_a = QuantumAssertion().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "f", "Check Ground Truth": "false"}),
                MockFlowFile(content=counts_content, attributes={"case_id": "c1", "partition": "p1", "marked_state": "00"})
            )
            assert res_a.relationship in ("pass", "fail")

        # QuantumSuccessProbabilityOracle
        _mark("QuantumSuccessProbabilityOracle")
        res_spo = QuantumSuccessProbabilityOracle().transform(
            MockContext(**{"Expected Outcome": "00"}),
            MockFlowFile(content=counts_content)
        )
        assert res_spo.relationship == "success"

        # QuantumCalibratedOracle
        _mark("QuantumCalibratedOracle")
        with tempfile.TemporaryDirectory() as td:
            res_cal = QuantumCalibratedOracle().transform(
                MockContext(**{"State Directory": td, "Consensus Label": "slot1", "Framework Label": "qiskit"}),
                MockFlowFile(content=counts_content, attributes={"test.run_id": "r1", "test.case_id": "c1"})
            )
            assert res_cal.relationship in ("pass", "fail", "failure")

        # QuantumConsensusOracle
        _mark("QuantumConsensusOracle")
        with tempfile.TemporaryDirectory() as td:
            res_con = QuantumConsensusOracle().transform(
                MockContext(**{"State Directory": td, "Consensus Label": "slot1", "Framework Label": "qiskit", "Expected Branches": "1"}),
                MockFlowFile(content=counts_content, attributes={"test.run_id": "r1", "test.case_id": "c1"})
            )
            assert res_con.relationship in ("pass", "fail", "waiting")

        # QuantumDistributionComparison
        _mark("QuantumDistributionComparison")
        with tempfile.TemporaryDirectory() as td:
            cmp_proc = QuantumDistributionComparison()
            c_ctx = MockContext(**{"Reports Directory": td, "State Directory": td, "Comparison Label": "pair1", "Framework Label": "qiskit"})
            ff_a = MockFlowFile(content=counts_content, attributes={"sim.framework": "qiskit"})
            ff_b = MockFlowFile(content=counts_content, attributes={"sim.framework": "cirq"})
            r1 = cmp_proc.transform(c_ctx, ff_a)
            assert r1.relationship == "failure" and r1.attributes.get("compare.status") == "waiting"
            c_ctx2 = MockContext(**{"Reports Directory": td, "State Directory": td, "Comparison Label": "pair1", "Framework Label": "cirq"})
            r2 = cmp_proc.transform(c_ctx2, ff_b)
            assert r2.relationship == "success" and r2.attributes.get("compare.status") == "complete"

        # QuantumDistributionOracle
        _mark("QuantumDistributionOracle")
        with tempfile.TemporaryDirectory() as td:
            res_dist = QuantumDistributionOracle().transform(
                MockContext(**{"Reports Directory": td, "State Directory": td, "Consensus Label": "s1", "Expected Branches": "1", "Multiple Comparison Correction": "holm"}),
                MockFlowFile(content=counts_content, attributes={"sim.framework": "qiskit", "test.run_id": "r", "test.case_id": "c"})
            )
            assert res_dist.relationship in ("pass", "fail", "waiting", "control_dissent")

        # MutationScoreReport
        _mark("MutationScoreReport")
        with tempfile.TemporaryDirectory() as td:
            res_ms = MutationScoreReport().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "test_flow"}),
                MockFlowFile(attributes={"test.run_id": "R1", "test.case_id": "c0", "mut.applied": "false", "assert.verdict": "PASS"})
            )
            assert res_ms.relationship == "success"


# ===========================================================================
# 6. PennyLane QML Lane (3 processors)
# ===========================================================================


class TestPennyLaneQML:
    """Exercising all 3 PennyLane QML processors."""

    def test_pennylane_qml_lane(self):
        from PennylaneDatasetLoader import PennylaneDatasetLoader
        from PennylaneFeatureEmbedding import PennylaneFeatureEmbedding
        from PennylaneVariationalClassifier import PennylaneVariationalClassifier

        _mark("PennylaneDatasetLoader")
        res_d = PennylaneDatasetLoader().transform(MockContext(**{"Dataset": "iris", "Batch Size": "2"}), MockFlowFile())
        assert res_d.relationship == "success"

        _mark("PennylaneFeatureEmbedding")
        res_e = PennylaneFeatureEmbedding().transform(
            MockContext(**{"Embedding": "angle", "Features": "[0.1, 0.2]"}), MockFlowFile()
        )
        assert res_e.relationship == "success"

        _mark("PennylaneVariationalClassifier")
        res_c = PennylaneVariationalClassifier().transform(
            MockContext(**{"Num Qubits": "2", "Layers": "1", "Max Iterations": "1", "Shots": "32"}),
            result_to_flowfile(res_d)
        )
        assert res_c.relationship == "success"


# ===========================================================================
# 7. Reporting & Utility (3 processors)
# ===========================================================================


class TestReportingAndUtility:
    """Exercising all 3 reporting and utility processors."""

    def test_reporting_and_utility(self):
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QiskitCircuitReport import QiskitCircuitReport
        from QuanifiReport import QuanifiReport
        from QuanifiUnitary import QuanifiUnitary

        q_h = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))

        with tempfile.TemporaryDirectory() as td:
            _mark("QiskitCircuitReport")
            res_cr = QiskitCircuitReport().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "f"}), q_h
            )
            assert res_cr.relationship == "success"

            _mark("QuanifiReport")
            res_qr = QuanifiReport().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "f"}), q_h
            )
            assert res_qr.relationship == "success"

            _mark("QuanifiUnitary")
            res_qu = QuanifiUnitary().transform(MockContext(**{"Max Qubits": "4"}), q_h)
            assert res_qu.relationship == "success"


# ===========================================================================
# 8. Hardware Execution Gateways (3 processors)
# ===========================================================================


class TestHardwareExecution:
    """Exercising all 3 hardware gateway processors in offline mode."""

    def test_hardware_gateways(self):
        from BraketDevice import BraketDevice
        from QiskitRuntimeSampler import QiskitRuntimeSampler
        from QrispIQMDevice import QrispIQMDevice

        ff = _qasm_ff()

        _mark("BraketDevice")
        res_b = BraketDevice().transform(MockContext(**{"Device": "local", "Shots": "64"}), ff)
        assert res_b.relationship == "success"

        _mark("QiskitRuntimeSampler")
        res_q = QiskitRuntimeSampler().transform(MockContext(**{"Backend": "fake_manila", "Shots": "64"}), ff)
        assert res_q.relationship == "success"

        _mark("QrispIQMDevice")
        res_i = QrispIQMDevice().transform(MockContext(**{"Device Instance": "local", "Shots": "64"}), ff)
        assert res_i.relationship == "success"


# ===========================================================================
# 9. Hardware Batch Lane (9 processors)
# ===========================================================================


class TestHardwareBatchLane:
    """Exercising all 9 hardware batch submitters and pollers offline."""

    def test_quantum_batch_result_expander(self):
        from QuantumBatchResultExpander import QuantumBatchResultExpander

        _mark("QuantumBatchResultExpander")
        exp_data = {
            "job_id": "job-1", "device": "test", "bit_order": "q0_right",
            "entries": [{"label": "e1", "kind": "control", "num_qubits": 2, "counts": {"00": 100}}]
        }
        res = QuantumBatchResultExpander().transform(MockContext(), MockFlowFile(content=json.dumps(exp_data)))
        assert res.relationship == "success"

    def test_ibm_batch_lane(self):
        from QuantumIBMBatchSubmitter import QuantumIBMBatchSubmitter
        from QuantumIBMBatchPoller import QuantumIBMBatchPoller

        _mark("QuantumIBMBatchSubmitter")

        class OfflineIBMSubmitter(QuantumIBMBatchSubmitter):
            def prepare(self, *args, **kwargs):
                return {
                    "backend": None, "transpiled": [], "layout": [0, 1], "two_qubit_gates": 2,
                    "pending": 0, "estimated_usage": 0.1, "widths": [2], "padded_width": 2, "capacity": 127
                }
            def run_job(self, plan, shots):
                return "ibm-job-123"
            def serialize_isa(self, plan, expected):
                return ["OPENQASM 2.0;"] * expected
            def service_for(self, token, instance):
                now = datetime.datetime.now(datetime.timezone.utc)
                class S:
                    @staticmethod
                    def usage():
                        return {"usage": {"seconds": 10}, "usage_limit": {"seconds": 600}, "usage_period": {"start_time": now, "end_time": now}}
                return S()

        with tempfile.TemporaryDirectory() as td:
            ctx = MockContext(**{
                "Batch Label": "b1", "Expected Circuits": "1", "Device": "ibm_test", "API Token": "tok",
                "State Directory": td, "Manifest Directory": td, "Submit Mode": "armed",
                "Fixed Layout": "0,1", "Excluded Physical Qubits": "", "Shots": "100", "Attribute Prefix": "arithmetic."
            })
            res_sub = OfflineIBMSubmitter().transform(ctx, _qasm_ff())
            assert res_sub.relationship == "submitted"

        _mark("QuantumIBMBatchPoller")

        class OfflineIBMPoller(QuantumIBMBatchPoller):
            def service(self, token, instance):
                class Pub:
                    class Data:
                        meas = type("Counts", (), {"get_counts": lambda self: {"00": 50, "11": 50}})()
                        def keys(self): return ["meas"]
                    data = Data()
                class Job:
                    def status(self): return "DONE"
                    def metrics(self): return {"usage": {"quantum_seconds": 1, "seconds": 1, "status": "complete"}}
                    def result(self): return [Pub()]
                return type("Service", (), {"job": lambda self, jid: Job()})()

        poller_manifest = {"job_id": "ibm-job-123", "device": "ibm_test", "entries": [{"label": "e1", "kind": "control"}]}
        res_pol = OfflineIBMPoller().transform(
            MockContext(**{"API Token": "tok", "Instance": "test", "Poll Timeout Seconds": "1"}),
            MockFlowFile(content=json.dumps(poller_manifest))
        )
        assert res_pol.relationship == "success"

    def test_iqm_batch_lane(self, monkeypatch):
        from QuantumIQMBatchSubmitter import QuantumIQMBatchSubmitter
        from QuantumIQMBatchPoller import QuantumIQMBatchPoller
        from IQMJobPoller import IQMJobPoller

        _mark("QuantumIQMBatchSubmitter")

        class OfflineIQMSubmitter(QuantumIQMBatchSubmitter):
            def prepare(self, *args, **kwargs):
                return {
                    "backend": None, "transpiled": [], "layout": [0, 1], "two_qubit_gates": 2,
                    "estimated_usage": 0.0, "widths": [2], "padded_width": 2, "capacity": 20
                }
            def run_job(self, plan, shots):
                return "iqm-job-1"
            def read_provider_credits(self, server, token):
                return 100.0

        with tempfile.TemporaryDirectory() as td:
            ctx = MockContext(**{
                "Batch Label": "b1", "Expected Circuits": "1", "API Token": "tok",
                "State Directory": td, "Device": "garnet", "Submit Mode": "armed", "Shots": "64"
            })
            res_sub = OfflineIQMSubmitter().transform(ctx, _qasm_ff())
            assert res_sub.relationship == "submitted"

        _mark("QuantumIQMBatchPoller")
        manifest = {"job_id": "iqm-job-1", "device": "garnet", "entries": [{"label": "e1", "kind": "control"}]}

        class MockResp:
            def __init__(self, data):
                self._data = data
                self.status_code = 200
                self.ok = True
            def json(self): return self._data
            def raise_for_status(self): pass

        def mock_get(self, url, *args, **kwargs):
            if url.endswith("/measurement_counts"):
                return MockResp([{"counts": {"00": 100}}])
            return MockResp({"status": "completed", "timeline": [{"status": "completed", "timestamp": "2026-01-01T00:00:00Z"}]})

        monkeypatch.setattr("requests.Session.get", mock_get)

        res_p = QuantumIQMBatchPoller().transform(
            MockContext(**{"API Token": "tok", "Job ID": "iqm-job-1", "Poll Timeout Seconds": "1"}),
            MockFlowFile(content=json.dumps(manifest))
        )
        assert res_p.relationship == "success"

        _mark("IQMJobPoller")
        poller = IQMJobPoller()
        iqm_calls = [
            MockResp({"status": "completed", "queue_position": None}),
            MockResp([{"measurement_keys": ["m"], "counts": {"00": 100}}]),
        ]
        monkeypatch.setattr(IQMJobPoller, "_get", lambda self, session, url, timeout=30: iqm_calls.pop(0))
        ff_iqm = MockFlowFile(content=b'{"job_id": "iqm-job-1"}', attributes={"hw.job_id": "iqm-job-1"})
        res_ijp = poller.transform(
            MockContext(**{"API Token": "tok", "Job ID": "iqm-job-1", "Poll Timeout Seconds": "1"}),
            ff_iqm
        )
        assert res_ijp.relationship == "success"

    def test_generic_batch_poller(self, monkeypatch):
        from QuantumBatchPoller import QuantumBatchPoller

        _mark("QuantumBatchPoller")
        poller = QuantumBatchPoller()

        class _Resp:
            def __init__(self, code, payload):
                self.status_code = code
                self._payload = payload
            def json(self): return self._payload

        monkeypatch.setattr(
            poller, "_get",
            lambda s, url, timeout=30: _Resp(200, [{"counts": {"00": 100}}])
            if url.endswith("measurement_counts") else _Resp(200, {"status": "completed"})
        )
        manifest = {"job_id": "j1", "entries": [{"label": "e1", "kind": "control"}]}
        res = poller.transform(
            MockContext(**{"API Token": "tok", "Job ID": "j1", "Poll Timeout Seconds": "1", "Poll Interval Seconds": "1"}),
            MockFlowFile(content=json.dumps(manifest))
        )
        assert res.relationship == "success"

    def test_quantum_inspire_batch_lane(self):
        from QuantumInspireBatchSubmitter import QuantumInspireBatchSubmitter
        from QuantumInspireBatchPoller import QuantumInspireBatchPoller
        from qiskit.providers.jobstatus import JobStatus
        import qiskit_quantuminspire.qi_jobs

        _mark("QuantumInspireBatchSubmitter")

        class StubQISubmitter(QuantumInspireBatchSubmitter):
            def prepare(self, *args, **kwargs):
                return {
                    "backend": None, "transpiled": [0], "layout": [0, 1], "two_qubit_gates": 2,
                    "widths": [2], "padded_width": 2, "capacity": 5, "native_basis_gates": "cz,h",
                    "chunks": [(0, 1)], "chunk_size": 5, "provider_batchjobs_per_queue_limit": 5,
                    "isa_profile": [{"two_qubit_gates": 1, "depth": 3, "gate_count": 5}],
                    "provider_backend_type": {"name": "Tuna-17"}, "provider_max_batch_jobs": 5,
                    "estimated_usage": 0.0,
                }
            def run_chunk(self, plan, shots, circuits):
                class Job:
                    batch_job_id = "qi-job-1"
                    circuits_run_data = [type("C", (), {"job_id": "cj-1"})()]
                return Job()
            def persist_handle(self, manifest_dir, job):
                p = f"{manifest_dir}/handle.json"
                with open(p, "w") as f:
                    f.write("{}")
                return (p, "")

        with tempfile.TemporaryDirectory() as td:
            ctx = MockContext(**{
                "Batch Label": "b1", "Expected Circuits": "1", "Device": "Tuna-17",
                "State Directory": td, "Manifest Directory": td, "Submit Mode": "armed",
                "Shots": "100"
            })
            res_sub = StubQISubmitter().transform(ctx, _qasm_ff())
            assert res_sub.relationship == "submitted"

        _mark("QuantumInspireBatchPoller")

        class FakeQIJob:
            batch_job_id = "qi-1"
            def status(self): return JobStatus.DONE
            def result(self, *args, **kwargs):
                class Res:
                    def get_counts(self, index=None): return {"00": 100}
                    results = [type("Exp", (), {"shots": 100})()]
                return Res()

        orig_deserialize = getattr(qiskit_quantuminspire.qi_jobs.QIJob, "deserialize", None)
        try:
            qiskit_quantuminspire.qi_jobs.QIJob.deserialize = staticmethod(lambda *args, **kwargs: FakeQIJob())
            manifest = {"job_id": "qi-1", "entries": [{"label": "e1", "kind": "control"}], "provider_handle_path": "/tmp/dummy"}
            res_p = QuantumInspireBatchPoller().transform(MockContext(), MockFlowFile(content=json.dumps(manifest)))
            assert res_p.relationship == "success"
        finally:
            if orig_deserialize:
                qiskit_quantuminspire.qi_jobs.QIJob.deserialize = orig_deserialize


# ===========================================================================
# 10. Dynamic Completeness Verification
# ===========================================================================


# All 122 processor classes implemented in the codebase and covered across the test classes above
ALL_SUITE_PROCESSORS = {
    # 1. Circuit Builders (48)
    "CirqAnsatz", "CirqGroverCircuit", "CirqGroverOperator", "CirqHadamardTransform",
    "CirqPhaseEstimation", "CirqPhaseOracle", "CirqQAOACircuit", "CirqQFTCircuit",
    "CirqQuantumArithmetic", "CirqStatePreparation", "PennylaneGroverCircuit",
    "PennylanePhaseEstimation", "PennylaneQAOACircuit", "PennylaneQuantumArithmetic",
    "PennylaneVariationalAnsatz", "PyquilAnsatz", "PyquilGroverCircuit",
    "PyquilGroverOperator", "PyquilPhaseOracle", "PyquilQAOACircuit", "PyquilQFT",
    "QiskitAmplitudeAmplification", "QiskitAnsatz", "QiskitBellState",
    "QiskitGroverCircuit", "QiskitGroverOperator", "QiskitHadamardTransform",
    "QiskitPhaseEstimation", "QiskitPhaseOracle", "QiskitQAOACircuit",
    "QiskitQFTCircuit", "QiskitQuantumArithmetic", "QiskitQuantumHelloWorld",
    "QiskitStatePreparation", "QiskitTeleportation", "QrispAmplitudeAmplification",
    "QrispAnsatz", "QrispBernsteinVazirani", "QrispDeutschJozsa", "QrispGroverCircuit",
    "QrispGroverOperator", "QrispHadamardTransform", "QrispPhaseEstimation",
    "QrispPhaseOracle", "QrispQAOACircuit", "QrispQCCSDAnsatz", "QrispQFTCircuit",
    "QrispQuantumArithmetic", "QrispShor", "QrispStatePreparation", "QrispSwapTest",
    "QrispTeleportation", "QrispTruthTableSynthesis", "QrispTypeEncoder",
    "ReadoutBaselineCircuit",
    # 2. Simulators & Expectation (13)
    "BraketSimulator", "CirqSimulator", "CirqStatevectorSimulator", "PennylaneExpectation",
    "PennylaneSimulator", "PyquilExpectation", "PyquilSimulator", "QSharpSimulator",
    "QiskitAerSimulator", "QiskitStatevectorSimulator", "QrispExpectation",
    "QrispSimulator", "QrispStatevectorSimulator",
    # 3. Algorithms & Solvers (20)
    "CirqAmplitudeEstimation", "CirqQAOA", "CirqVQE", "PennylaneAmplitudeEstimation",
    "PennylaneQAOA", "PyquilQAOA", "PyquilVQE", "QiskitAmplitudeEstimation",
    "QiskitGroverSearch", "QiskitQAOA", "QiskitVQE", "QrispAmplitudeEstimation",
    "QrispGroverSearch", "QrispQAOA", "QrispQIRO", "QrispVQE",
    # 4. Hamiltonians & Encoders (9)
    "CirqHamiltonian", "MaxCliqueProblem", "MaxCutProblem", "MaxIndependentSetProblem",
    "MoleculeHamiltonian", "PortfolioRebalancingProblem", "QiskitHamiltonian",
    "QrispHamiltonian", "QuboToHamiltonian",
    # 5. Differential & Mutation (14)
    "MutationScoreReport", "QuanifiUnitary", "QuantumAssertion", "QuantumCalibratedOracle",
    "QuantumConsensusOracle", "QuantumDistributionComparison", "QuantumDistributionOracle",
    "QuantumMutator", "QuantumQAOAEvaluator", "QuantumSuccessProbabilityOracle",
    "QuantumTestCaseSource", "QuantumUnitaryComparison",
    # 6. PennyLane QML Lane (3)
    "PennylaneDatasetLoader", "PennylaneFeatureEmbedding", "PennylaneVariationalClassifier",
    # 7. Reporting & Utility (3)
    "QiskitCircuitReport", "QuanifiReport", "QuantumBatchResultExpander",
    # 8. Hardware Gateways (3)
    "BraketDevice", "QiskitRuntimeSampler", "QrispIQMDevice",
    # 9. Hardware Batch Lane (9)
    "IQMJobPoller", "QuantumBatchPoller", "QuantumIBMBatchPoller",
    "QuantumIBMBatchSubmitter", "QuantumIQMBatchPoller", "QuantumIQMBatchSubmitter",
    "QuantumInspireBatchPoller", "QuantumInspireBatchSubmitter",
}


def test_every_single_processor_is_exercised():
    """Verify that every single processor discovered in the repository is accounted for and exercised."""
    discovered = {p["name"] for p in tool.categorise(tool.discover())}
    assert len(discovered) == 122, f"Expected 122 processors on saeg/main, found {len(discovered)}"
    uncovered = discovered - ALL_SUITE_PROCESSORS
    assert not uncovered, f"Processors missing from test suite: {uncovered}"

    # If the full suite was executed in this session, verify runtime registration matches 100%
    if EXERCISED_PROCESSORS:
        untested = discovered - EXERCISED_PROCESSORS
        assert not untested, (
            f"{len(untested)} processors were not exercised in this comprehensive suite:\n"
            + "\n".join(f"  - {name}" for name in sorted(untested))
        )

