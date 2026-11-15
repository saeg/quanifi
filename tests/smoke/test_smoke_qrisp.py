"""Smoke test suite exercising all 29 Qrisp processors."""

import pytest

from tests.smoke.smoke_helpers import (
    MockContext,
    MockFlowFile,
    mark_exercised,
    qasm_ff,
    result_to_flowfile,
)

pytestmark = pytest.mark.smoke_comprehensive


class TestQrispSubsystem:
    """Smoke tests covering all 29 Qrisp processors."""

    def test_qrisp_builders(self):
        from MaxCutProblem import MaxCutProblem
        from QrispAmplitudeAmplification import QrispAmplitudeAmplification
        from QrispAnsatz import QrispAnsatz
        from QrispBernsteinVazirani import QrispBernsteinVazirani
        from QrispDeutschJozsa import QrispDeutschJozsa
        from QrispGroverCircuit import QrispGroverCircuit
        from QrispGroverOperator import QrispGroverOperator
        from QrispHadamardTransform import QrispHadamardTransform
        from QrispPhaseEstimation import QrispPhaseEstimation
        from QrispPhaseOracle import QrispPhaseOracle
        from QrispQAOACircuit import QrispQAOACircuit
        from QrispQCCSDAnsatz import QrispQCCSDAnsatz
        from QrispQFTCircuit import QrispQFTCircuit
        from QrispQuantumArithmetic import QrispQuantumArithmetic
        from QrispShor import QrispShor
        from QrispStatePreparation import QrispStatePreparation
        from QrispSwapTest import QrispSwapTest
        from QrispTeleportation import QrispTeleportation
        from QrispTruthTableSynthesis import QrispTruthTableSynthesis
        from QrispTypeEncoder import QrispTypeEncoder

        # Base transforms
        for name, cls in [
            ("QrispAnsatz", QrispAnsatz),
            ("QrispBernsteinVazirani", QrispBernsteinVazirani),
            ("QrispDeutschJozsa", QrispDeutschJozsa),
            ("QrispHadamardTransform", QrispHadamardTransform),
            ("QrispPhaseEstimation", QrispPhaseEstimation),
            ("QrispQFTCircuit", QrispQFTCircuit),
            ("QrispQuantumArithmetic", QrispQuantumArithmetic),
            ("QrispShor", QrispShor),
            ("QrispStatePreparation", QrispStatePreparation),
            ("QrispSwapTest", QrispSwapTest),
            ("QrispTeleportation", QrispTeleportation),
            ("QrispTruthTableSynthesis", QrispTruthTableSynthesis),
        ]:
            mark_exercised(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        # Grover circuit & operator & phase oracle & QAA
        mark_exercised("QrispGroverCircuit")
        res_gc = QrispGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()
        )
        assert res_gc.relationship == "success"

        mark_exercised("QrispAmplitudeAmplification")
        res_aa = QrispAmplitudeAmplification().transform(
            MockContext(**{"Marked State": "10", "Iterations": "1"}), MockFlowFile()
        )
        assert res_aa.relationship == "success"

        h_ff = result_to_flowfile(QrispHadamardTransform().transform(MockContext(), MockFlowFile()))

        mark_exercised("QrispGroverOperator")
        res_go = QrispGroverOperator().transform(MockContext(), h_ff)
        assert res_go.relationship == "success"

        mark_exercised("QrispPhaseOracle")
        res_po = QrispPhaseOracle().transform(
            MockContext(**{"Marked State": "10"}), h_ff
        )
        assert res_po.relationship == "success"

        # Type Encoder
        mark_exercised("QrispTypeEncoder")
        res_te = QrispTypeEncoder().transform(
            MockContext(**{"Variable Name": "x", "Type": "int", "Value": "3"}), MockFlowFile()
        )
        assert res_te.relationship == "success"

        # QCCSD Ansatz
        mark_exercised("QrispQCCSDAnsatz")
        res_qccsd = QrispQCCSDAnsatz().transform(
            MockContext(**{"Num Spin Orbitals": "4", "Num Particles": "2"}), MockFlowFile()
        )
        assert res_qccsd.relationship == "success"

        # QAOA Circuit
        mark_exercised("QrispQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = QrispQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_qrisp_simulators_and_expectation(self):
        from QrispExpectation import QrispExpectation
        from QrispHamiltonian import QrispHamiltonian
        from QrispSimulator import QrispSimulator
        from QrispStatevectorSimulator import QrispStatevectorSimulator

        mark_exercised("QrispSimulator")
        res_sim = QrispSimulator().transform(
            MockContext(**{"Shots": "50"}), qasm_ff()
        )
        assert res_sim.relationship == "success"

        mark_exercised("QrispStatevectorSimulator")
        res_sv = QrispStatevectorSimulator().transform(
            MockContext(), qasm_ff()
        )
        assert res_sv.relationship == "success"

        # Expectation
        from QrispHadamardTransform import QrispHadamardTransform
        mark_exercised("QrispExpectation")
        qr_ff = result_to_flowfile(QrispHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_exp = QrispExpectation().transform(MockContext(), qr_ff)
        assert res_exp.relationship == "success"

    def test_qrisp_solvers_and_algorithms(self):
        from MaxIndependentSetProblem import MaxIndependentSetProblem
        from QrispAmplitudeEstimation import QrispAmplitudeEstimation
        from QrispAnsatz import QrispAnsatz
        from QrispGroverSearch import QrispGroverSearch
        from QrispHamiltonian import QrispHamiltonian
        from QrispQAOA import QrispQAOA
        from QrispQIRO import QrispQIRO
        from QrispVQE import QrispVQE

        # QrispAmplitudeEstimation
        mark_exercised("QrispAmplitudeEstimation")
        res_ae = QrispAmplitudeEstimation().transform(MockContext(), MockFlowFile())
        assert res_ae.relationship == "success"

        # QrispGroverSearch
        mark_exercised("QrispGroverSearch")
        res_gs = QrispGroverSearch().transform(
            MockContext(**{"Target Bitstring": "10"}), MockFlowFile()
        )
        assert res_gs.relationship == "success"

        # QrispHamiltonian
        mark_exercised("QrispHamiltonian")
        qr_h = result_to_flowfile(QrispHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0"}), MockFlowFile()))

        # QrispQAOA
        mark_exercised("QrispQAOA")
        res_qaoa = QrispQAOA().transform(MockContext(**{"Max Iterations": "1"}), qr_h)
        assert res_qaoa.relationship == "success"

        # QrispVQE
        mark_exercised("QrispVQE")
        qr_a = result_to_flowfile(QrispAnsatz().transform(MockContext(**{"Reps": "1"}), qr_h))
        res_vqe = QrispVQE().transform(MockContext(**{"Max Iterations": "1"}), qr_a)
        assert res_vqe.relationship == "success"

        # QrispQIRO
        mark_exercised("QrispQIRO")
        mis_ff = result_to_flowfile(
            MaxIndependentSetProblem().transform(MockContext(**{"Edges": "[[0, 1]]"}), MockFlowFile())
        )
        res_qiro = QrispQIRO().transform(
            MockContext(**{"Max Iterations": "1", "Recursions": "1", "Layers": "1"}), mis_ff
        )
        assert res_qiro.relationship == "success"

    def test_qrisp_hardware_gateway(self):
        from QrispIQMDevice import QrispIQMDevice

        mark_exercised("QrispIQMDevice")
        res = QrispIQMDevice().transform(
            MockContext(**{"Device Instance": "local", "Shots": "20"}),
            qasm_ff(),
        )
        assert res.relationship == "success"
