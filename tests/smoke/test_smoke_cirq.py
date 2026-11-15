"""Smoke test suite exercising all 16 Cirq processors."""

import pytest

from tests.smoke.smoke_helpers import (
    MockContext,
    MockFlowFile,
    mark_exercised,
    qasm_ff,
    result_to_flowfile,
)

pytestmark = pytest.mark.smoke_comprehensive


class TestCirqSubsystem:
    """Smoke tests covering all 16 Cirq processors."""

    def test_cirq_builders(self):
        from CirqAnsatz import CirqAnsatz
        from CirqGroverCircuit import CirqGroverCircuit
        from CirqGroverOperator import CirqGroverOperator
        from CirqHadamardTransform import CirqHadamardTransform
        from CirqPhaseEstimation import CirqPhaseEstimation
        from CirqPhaseOracle import CirqPhaseOracle
        from CirqQAOACircuit import CirqQAOACircuit
        from CirqQFTCircuit import CirqQFTCircuit
        from CirqQuantumArithmetic import CirqQuantumArithmetic
        from CirqStatePreparation import CirqStatePreparation
        from MaxCutProblem import MaxCutProblem

        # Base transforms
        for name, cls in [
            ("CirqAnsatz", CirqAnsatz),
            ("CirqHadamardTransform", CirqHadamardTransform),
            ("CirqPhaseEstimation", CirqPhaseEstimation),
            ("CirqQFTCircuit", CirqQFTCircuit),
            ("CirqQuantumArithmetic", CirqQuantumArithmetic),
            ("CirqStatePreparation", CirqStatePreparation),
        ]:
            mark_exercised(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        # Grover circuit & operator & phase oracle
        mark_exercised("CirqGroverCircuit")
        res_gc = CirqGroverCircuit().transform(
            MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile()
        )
        assert res_gc.relationship == "success"

        c_h_ff = result_to_flowfile(CirqHadamardTransform().transform(MockContext(), MockFlowFile()))

        mark_exercised("CirqPhaseOracle")
        res_po = CirqPhaseOracle().transform(
            MockContext(**{"Marked State": "10"}), c_h_ff
        )
        assert res_po.relationship == "success"

        mark_exercised("CirqGroverOperator")
        res_go = CirqGroverOperator().transform(MockContext(), c_h_ff)
        assert res_go.relationship == "success"

        # QAOA circuit
        mark_exercised("CirqQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = CirqQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_cirq_simulators(self):
        from CirqSimulator import CirqSimulator
        from CirqStatevectorSimulator import CirqStatevectorSimulator

        mark_exercised("CirqSimulator")
        res_sim = CirqSimulator().transform(
            MockContext(**{"Shots": "50"}), qasm_ff()
        )
        assert res_sim.relationship == "success"

        mark_exercised("CirqStatevectorSimulator")
        res_sv = CirqStatevectorSimulator().transform(
            MockContext(), qasm_ff()
        )
        assert res_sv.relationship == "success"

    def test_cirq_solvers_and_algorithms(self):
        from CirqAmplitudeEstimation import CirqAmplitudeEstimation
        from CirqAnsatz import CirqAnsatz
        from CirqHamiltonian import CirqHamiltonian
        from CirqQAOA import CirqQAOA
        from CirqVQE import CirqVQE

        # CirqAmplitudeEstimation
        mark_exercised("CirqAmplitudeEstimation")
        res_ae = CirqAmplitudeEstimation().transform(MockContext(), MockFlowFile())
        assert res_ae.relationship == "success"

        # CirqHamiltonian
        mark_exercised("CirqHamiltonian")
        c_h = result_to_flowfile(CirqHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0"}), MockFlowFile()))

        # CirqQAOA
        mark_exercised("CirqQAOA")
        res_qaoa = CirqQAOA().transform(MockContext(**{"Max Iterations": "1"}), c_h)
        assert res_qaoa.relationship == "success"

        # CirqVQE
        mark_exercised("CirqVQE")
        c_a = result_to_flowfile(CirqAnsatz().transform(MockContext(**{"Reps": "1"}), c_h))
        res_vqe = CirqVQE().transform(MockContext(**{"Max Iterations": "1", "Shots": "32"}), c_a)
        assert res_vqe.relationship == "success"
