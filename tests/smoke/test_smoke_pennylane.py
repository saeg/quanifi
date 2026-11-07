"""Smoke test suite exercising all 12 PennyLane processors."""

import pytest

from tests.smoke.smoke_helpers import (
    MockContext,
    MockFlowFile,
    mark_exercised,
    qasm_ff,
    result_to_flowfile,
)

pytestmark = pytest.mark.smoke_comprehensive


class TestPennylaneSubsystem:
    """Smoke tests covering all 12 PennyLane processors."""

    def test_pennylane_builders(self):
        from MaxCutProblem import MaxCutProblem
        from PennylaneGroverCircuit import PennylaneGroverCircuit
        from PennylanePhaseEstimation import PennylanePhaseEstimation
        from PennylaneQAOACircuit import PennylaneQAOACircuit
        from PennylaneQuantumArithmetic import PennylaneQuantumArithmetic
        from PennylaneVariationalAnsatz import PennylaneVariationalAnsatz

        for name, cls in [
            ("PennylaneGroverCircuit", PennylaneGroverCircuit),
            ("PennylanePhaseEstimation", PennylanePhaseEstimation),
            ("PennylaneQuantumArithmetic", PennylaneQuantumArithmetic),
            ("PennylaneVariationalAnsatz", PennylaneVariationalAnsatz),
        ]:
            mark_exercised(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        # QAOA Circuit
        mark_exercised("PennylaneQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = PennylaneQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_pennylane_simulators_and_expectation(self):
        from PennylaneExpectation import PennylaneExpectation
        from PennylaneFeatureEmbedding import PennylaneFeatureEmbedding
        from PennylaneSimulator import PennylaneSimulator

        mark_exercised("PennylaneSimulator")
        res_sim = PennylaneSimulator().transform(
            MockContext(**{"Shots": "50"}), qasm_ff()
        )
        assert res_sim.relationship == "success"

        mark_exercised("PennylaneExpectation")
        emb_ff = result_to_flowfile(
            PennylaneFeatureEmbedding().transform(
                MockContext(**{"Embedding": "angle", "Features": "[0.1, 0.2]"}), MockFlowFile()
            )
        )
        res_exp = PennylaneExpectation().transform(MockContext(), emb_ff)
        assert res_exp.relationship == "success"

    def test_pennylane_solvers_and_algorithms(self):
        from CirqHamiltonian import CirqHamiltonian
        from PennylaneAmplitudeEstimation import PennylaneAmplitudeEstimation
        from PennylaneQAOA import PennylaneQAOA

        mark_exercised("PennylaneAmplitudeEstimation")
        res_ae = PennylaneAmplitudeEstimation().transform(MockContext(), MockFlowFile())
        assert res_ae.relationship == "success"

        mark_exercised("PennylaneQAOA")
        c_h = result_to_flowfile(CirqHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0"}), MockFlowFile()))
        res_qaoa = PennylaneQAOA().transform(MockContext(**{"Max Iterations": "1"}), c_h)
        assert res_qaoa.relationship == "success"

    def test_pennylane_qml_lane(self):
        from PennylaneDatasetLoader import PennylaneDatasetLoader
        from PennylaneFeatureEmbedding import PennylaneFeatureEmbedding
        from PennylaneVariationalClassifier import PennylaneVariationalClassifier

        mark_exercised("PennylaneDatasetLoader")
        res_dl = PennylaneDatasetLoader().transform(
            MockContext(**{"Dataset": "iris", "Batch Size": "2"}), MockFlowFile()
        )
        assert res_dl.relationship == "success"

        mark_exercised("PennylaneFeatureEmbedding")
        res_fe = PennylaneFeatureEmbedding().transform(
            MockContext(**{"Embedding": "angle", "Features": "[0.1, 0.2]"}), MockFlowFile()
        )
        assert res_fe.relationship == "success"

        mark_exercised("PennylaneVariationalClassifier")
        res_vc = PennylaneVariationalClassifier().transform(
            MockContext(**{"Num Qubits": "2", "Layers": "1", "Max Iterations": "1", "Shots": "32"}),
            result_to_flowfile(res_dl),
        )
        assert res_vc.relationship == "success"
