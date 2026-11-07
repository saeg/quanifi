"""Smoke test suite exercising all 22 Qiskit processors."""

import pytest

from tests.smoke.smoke_helpers import (
    MockContext,
    MockFlowFile,
    mark_exercised,
    qasm_ff,
    result_to_flowfile,
    result_to_flowfile_merged,
)

pytestmark = pytest.mark.smoke_comprehensive


class TestQiskitSubsystem:
    """Smoke tests covering all 22 Qiskit processors."""

    def test_qiskit_builders(self):
        from MaxCutProblem import MaxCutProblem
        from QiskitAmplitudeAmplification import QiskitAmplitudeAmplification
        from QiskitAnsatz import QiskitAnsatz
        from QiskitBellState import QiskitBellState
        from QiskitGroverCircuit import QiskitGroverCircuit
        from QiskitGroverOperator import QiskitGroverOperator
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QiskitPhaseEstimation import QiskitPhaseEstimation
        from QiskitPhaseOracle import QiskitPhaseOracle
        from QiskitQAOACircuit import QiskitQAOACircuit
        from QiskitQFTCircuit import QiskitQFTCircuit
        from QiskitQuantumArithmetic import QiskitQuantumArithmetic
        from QiskitQuantumHelloWorld import QiskitQuantumHelloWorld
        from QiskitStatePreparation import QiskitStatePreparation
        from QiskitTeleportation import QiskitTeleportation

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
            mark_exercised(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        # Grover circuit & operator & phase oracle
        mark_exercised("QiskitGroverCircuit")
        res_gc = QiskitGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()
        )
        assert res_gc.relationship == "success"

        mark_exercised("QiskitGroverOperator")
        h_ff = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_go = QiskitGroverOperator().transform(MockContext(), h_ff)
        assert res_go.relationship == "success"

        mark_exercised("QiskitPhaseOracle")
        res_po = QiskitPhaseOracle().transform(
            MockContext(**{"Marked State": "10"}), h_ff
        )
        assert res_po.relationship == "success"

        # QAOA Circuit
        mark_exercised("QiskitQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = QiskitQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_qiskit_simulators(self):
        from QiskitAerSimulator import QiskitAerSimulator
        from QiskitStatevectorSimulator import QiskitStatevectorSimulator

        mark_exercised("QiskitAerSimulator")
        res = QiskitAerSimulator().transform(
            MockContext(**{"Shots": "100"}), qasm_ff()
        )
        assert res.relationship == "success"

        mark_exercised("QiskitStatevectorSimulator")
        res = QiskitStatevectorSimulator().transform(
            MockContext(), qasm_ff()
        )
        assert res.relationship == "success"

    def test_qiskit_solvers_and_algorithms(self):
        from QiskitAmplitudeEstimation import QiskitAmplitudeEstimation
        from QiskitAnsatz import QiskitAnsatz
        from QiskitGroverSearch import QiskitGroverSearch
        from QiskitHamiltonian import QiskitHamiltonian
        from QiskitQAOA import QiskitQAOA
        from QiskitVQE import QiskitVQE

        # QiskitAmplitudeEstimation
        mark_exercised("QiskitAmplitudeEstimation")
        res = QiskitAmplitudeEstimation().transform(MockContext(), MockFlowFile())
        assert res.relationship == "success"

        # QiskitGroverSearch
        mark_exercised("QiskitGroverSearch")
        res = QiskitGroverSearch().transform(
            MockContext(**{"Target Bitstring": "10"}), MockFlowFile()
        )
        assert res.relationship == "success"

        # QiskitHamiltonian
        mark_exercised("QiskitHamiltonian")
        ff_ham = result_to_flowfile(
            QiskitHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0 + Z1"}), MockFlowFile())
        )

        # QiskitQAOA
        mark_exercised("QiskitQAOA")
        res_qaoa = QiskitQAOA().transform(MockContext(**{"Max Iterations": "1"}), ff_ham)
        assert res_qaoa.relationship == "success"

        # QiskitVQE
        mark_exercised("QiskitVQE")
        q_h = result_to_flowfile(QiskitHamiltonian().transform(MockContext(), MockFlowFile()))
        q_a = result_to_flowfile(QiskitAnsatz().transform(MockContext(), q_h))
        res_vqe = QiskitVQE().transform(MockContext(**{"Max Iterations": "1"}), q_a)
        assert res_vqe.relationship == "success", res_vqe.attributes

    def test_qiskit_hardware_gateway(self):
        from QiskitRuntimeSampler import QiskitRuntimeSampler

        mark_exercised("QiskitRuntimeSampler")
        res = QiskitRuntimeSampler().transform(
            MockContext(**{"Backend": "fake_manila", "Shots": "50"}),
            qasm_ff(),
        )
        assert res.relationship == "success"
