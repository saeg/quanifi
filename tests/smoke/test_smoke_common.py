"""Smoke test suite exercising all 21 Common processors (problem encoders, differential & mutation, reporting)."""

import json
import tempfile
import pytest

from tests.smoke.smoke_helpers import (
    MockContext,
    MockFlowFile,
    mark_exercised,
    result_to_flowfile,
)

pytestmark = pytest.mark.smoke_comprehensive


class TestCommonSubsystem:
    """Smoke tests covering all 21 common/shared processors."""

    def test_problem_encoders_and_hamiltonians(self):
        from MaxCliqueProblem import MaxCliqueProblem
        from MaxCutProblem import MaxCutProblem
        from MaxIndependentSetProblem import MaxIndependentSetProblem
        from MoleculeHamiltonian import MoleculeHamiltonian
        from PortfolioRebalancingProblem import PortfolioRebalancingProblem
        from QuboToHamiltonian import QuboToHamiltonian

        encoders = [
            ("MaxCliqueProblem", MaxCliqueProblem, {}),
            ("MaxCutProblem", MaxCutProblem, {}),
            ("MaxIndependentSetProblem", MaxIndependentSetProblem, {}),
            ("MoleculeHamiltonian", MoleculeHamiltonian, {}),
            ("PortfolioRebalancingProblem", PortfolioRebalancingProblem, {}),
            ("QuboToHamiltonian", QuboToHamiltonian, {"QUBO Matrix": "[[1, -1], [-1, 1]]"}),
        ]

        for name, cls, props in encoders:
            mark_exercised(name)
            res = cls().transform(MockContext(**props), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

    def test_baseline_circuit(self):
        from ReadoutBaselineCircuit import ReadoutBaselineCircuit

        mark_exercised("ReadoutBaselineCircuit")
        res_rb = ReadoutBaselineCircuit().transform(MockContext(), MockFlowFile())
        assert res_rb.relationship == "success"

    def test_differential_mutation_and_evaluators(self):
        from CirqHadamardTransform import CirqHadamardTransform
        from MutationScoreReport import MutationScoreReport
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QuanifiUnitary import QuanifiUnitary
        from QuantumAssertion import QuantumAssertion
        from QuantumCalibratedOracle import QuantumCalibratedOracle
        from QuantumConsensusOracle import QuantumConsensusOracle
        from QuantumDistributionComparison import QuantumDistributionComparison
        from QuantumDistributionOracle import QuantumDistributionOracle
        from QuantumMutator import QuantumMutator
        from QuantumQAOAEvaluator import QuantumQAOAEvaluator
        from QuantumSuccessProbabilityOracle import QuantumSuccessProbabilityOracle
        from QuantumTestCaseSource import QuantumTestCaseSource
        from QuantumUnitaryComparison import QuantumUnitaryComparison

        # Mutator & TestCaseSource
        mark_exercised("QuantumMutator")
        q_h = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_m = QuantumMutator().transform(MockContext(**{"Mutation Operator": "bitflip"}), q_h)
        assert res_m.relationship == "success"

        mark_exercised("QuantumTestCaseSource")
        res_tcs = QuantumTestCaseSource().transform(MockContext(), MockFlowFile())
        assert res_tcs.relationship == "success"

        # Unitary & Unitary Comparison
        mark_exercised("QuanifiUnitary")
        c_h = result_to_flowfile(CirqHadamardTransform().transform(MockContext(), MockFlowFile()))
        res_uq = QuanifiUnitary().transform(MockContext(**{"Max Qubits": "4"}), q_h)
        res_uc = QuanifiUnitary().transform(MockContext(**{"Max Qubits": "4"}), c_h)
        assert res_uq.relationship == "success"
        assert res_uc.relationship == "success"

        mark_exercised("QuantumUnitaryComparison")
        with tempfile.TemporaryDirectory() as td:
            ctx = MockContext(**{"State Directory": td, "Comparison Label": "label-1", "Framework Label": "qiskit", "Tolerance": "1e-6"})
            cmp_proc = QuantumUnitaryComparison()
            r1 = cmp_proc.transform(ctx, result_to_flowfile(res_uq))
            assert r1.relationship == "failure" and r1.attributes["equiv.status"] == "waiting"
            ctx2 = MockContext(**{"State Directory": td, "Comparison Label": "label-1", "Framework Label": "cirq", "Tolerance": "1e-6"})
            r2 = cmp_proc.transform(ctx2, result_to_flowfile(res_uc))
            assert r2.relationship == "success" and r2.attributes["equiv.status"] == "complete"

        # Oracles & Assertion
        counts_content = json.dumps({"00": 50, "11": 50})

        mark_exercised("QuantumAssertion")
        with tempfile.TemporaryDirectory() as td:
            res_a = QuantumAssertion().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "f", "Check Ground Truth": "false"}),
                MockFlowFile(content=counts_content, attributes={"case_id": "c1", "partition": "p1", "marked_state": "00"}),
            )
            assert res_a.relationship in ("pass", "fail")

        mark_exercised("QuantumSuccessProbabilityOracle")
        res_spo = QuantumSuccessProbabilityOracle().transform(
            MockContext(**{"Expected Outcome": "00"}), MockFlowFile(content=counts_content)
        )
        assert res_spo.relationship == "success"

        mark_exercised("QuantumQAOAEvaluator")
        ff_counts = MockFlowFile(
            content=counts_content,
            attributes={
                "sim.bit_order": "q0_left",
                "builder.component": "X",
                "sim.component": "Y",
                "hamiltonian.json": json.dumps({"num_qubits": 2, "terms": [["ZZ", 1.0, 0.0]]}),
            },
        )
        res_qe = QuantumQAOAEvaluator().transform(MockContext(**{"Hamiltonian": "Z0 Z1"}), ff_counts)
        assert res_qe.relationship == "success", res_qe.attributes

        mark_exercised("QuantumCalibratedOracle")
        with tempfile.TemporaryDirectory() as td:
            res_cal = QuantumCalibratedOracle().transform(
                MockContext(**{"State Directory": td, "Consensus Label": "slot1", "Framework Label": "qiskit"}),
                MockFlowFile(content=counts_content, attributes={"test.run_id": "r1", "test.case_id": "c1"}),
            )
            assert res_cal.relationship in ("pass", "fail", "failure")

        mark_exercised("QuantumConsensusOracle")
        with tempfile.TemporaryDirectory() as td:
            res_con = QuantumConsensusOracle().transform(
                MockContext(**{"State Directory": td, "Consensus Label": "slot1", "Framework Label": "qiskit", "Expected Branches": "1"}),
                MockFlowFile(content=counts_content, attributes={"test.run_id": "r1", "test.case_id": "c1"}),
            )
            assert res_con.relationship in ("pass", "fail", "waiting")

        mark_exercised("QuantumDistributionComparison")
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

        mark_exercised("QuantumDistributionOracle")
        with tempfile.TemporaryDirectory() as td:
            res_dist = QuantumDistributionOracle().transform(
                MockContext(**{"Reports Directory": td, "State Directory": td, "Consensus Label": "s1", "Expected Branches": "1", "Multiple Comparison Correction": "holm"}),
                MockFlowFile(content=counts_content, attributes={"sim.framework": "qiskit", "test.run_id": "r", "test.case_id": "c"}),
            )
            assert res_dist.relationship in ("pass", "fail", "waiting", "control_dissent")

        mark_exercised("MutationScoreReport")
        with tempfile.TemporaryDirectory() as td:
            res_ms = MutationScoreReport().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "test_flow"}),
                MockFlowFile(attributes={"test.run_id": "R1", "test.case_id": "c0", "mut.applied": "false", "assert.verdict": "PASS"}),
            )
            assert res_ms.relationship == "success"

    def test_reporting(self):
        from QiskitCircuitReport import QiskitCircuitReport
        from QiskitHadamardTransform import QiskitHadamardTransform
        from QuanifiReport import QuanifiReport

        q_h = result_to_flowfile(QiskitHadamardTransform().transform(MockContext(), MockFlowFile()))

        with tempfile.TemporaryDirectory() as td:
            mark_exercised("QiskitCircuitReport")
            res_cr = QiskitCircuitReport().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "f"}), q_h
            )
            assert res_cr.relationship == "success"

            mark_exercised("QuanifiReport")
            res_qr = QuanifiReport().transform(
                MockContext(**{"Reports Directory": td, "Flow Name": "f"}), q_h
            )
            assert res_qr.relationship == "success"
