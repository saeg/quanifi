"""Smoke test suite exercising all 10 PyQuil processors."""

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


class TestPyquilSubsystem:
    """Smoke tests covering all 10 PyQuil processors."""

    def test_pyquil_builders(self):
        from MaxCutProblem import MaxCutProblem
        from PyquilAnsatz import PyquilAnsatz
        from PyquilGroverCircuit import PyquilGroverCircuit
        from PyquilGroverOperator import PyquilGroverOperator
        from PyquilPhaseOracle import PyquilPhaseOracle
        from PyquilQAOACircuit import PyquilQAOACircuit
        from PyquilQFT import PyquilQFT

        for name, cls in [
            ("PyquilAnsatz", PyquilAnsatz),
            ("PyquilGroverCircuit", PyquilGroverCircuit),
            ("PyquilQFT", PyquilQFT),
        ]:
            mark_exercised(name)
            res = cls().transform(MockContext(), MockFlowFile())
            assert res.relationship == "success", f"{name} failed: {res.attributes}"

        # Grover operator & phase oracle (quil format input)
        mark_exercised("PyquilGroverOperator")
        p_gc = result_to_flowfile(
            PyquilGroverCircuit().transform(MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile())
        )
        res_go = PyquilGroverOperator().transform(MockContext(), p_gc)
        assert res_go.relationship == "success"

        mark_exercised("PyquilPhaseOracle")
        res_po = PyquilPhaseOracle().transform(MockContext(**{"Marked State": "10"}), p_gc)
        assert res_po.relationship == "success"

        # QAOA Circuit
        mark_exercised("PyquilQAOACircuit")
        mc_ff = result_to_flowfile(MaxCutProblem().transform(MockContext(**{"Num Nodes": "3"}), MockFlowFile()))
        res_qaoa = PyquilQAOACircuit().transform(MockContext(), mc_ff)
        assert res_qaoa.relationship == "success"

    def test_pyquil_simulators_and_expectation(self):
        from PyquilExpectation import PyquilExpectation
        from PyquilGroverCircuit import PyquilGroverCircuit
        from PyquilSimulator import PyquilSimulator

        pyq_ff = result_to_flowfile(
            PyquilGroverCircuit().transform(MockContext(**{"Marked State": "10", "Num Iterations": "1"}), MockFlowFile())
        )

        mark_exercised("PyquilSimulator")
        res_sim = PyquilSimulator().transform(
            MockContext(**{"Shots": "50"}), pyq_ff
        )
        assert res_sim.relationship == "success", res_sim.attributes

        mark_exercised("PyquilExpectation")
        res_exp = PyquilExpectation().transform(MockContext(), pyq_ff)
        assert res_exp.relationship == "success"

    def test_pyquil_solvers_and_algorithms(self):
        from CirqHamiltonian import CirqHamiltonian
        from PyquilAnsatz import PyquilAnsatz
        from PyquilQAOA import PyquilQAOA
        from PyquilVQE import PyquilVQE
        from QiskitHamiltonian import QiskitHamiltonian

        # PyquilQAOA
        mark_exercised("PyquilQAOA")
        c_h = result_to_flowfile(CirqHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0"}), MockFlowFile()))
        res_qaoa = PyquilQAOA().transform(MockContext(**{"Max Iterations": "1"}), c_h)
        assert res_qaoa.relationship == "success"

        # PyquilVQE (with merged upstream flowfile for format retention)
        mark_exercised("PyquilVQE")
        p_h = result_to_flowfile_merged(
            QiskitHamiltonian().transform(MockContext(**{"Hamiltonian": "Z0 + Z1"}), MockFlowFile()),
            MockFlowFile(),
        )
        p_a = result_to_flowfile_merged(
            PyquilAnsatz().transform(MockContext(**{"Output Mode": "attach"}), p_h),
            p_h,
        )
        res_vqe = PyquilVQE().transform(MockContext(**{"Max Iterations": "1", "Shots": "32"}), p_a)
        assert res_vqe.relationship == "success"
