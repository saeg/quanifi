import json
import pytest

from QrispHamiltonian import QrispHamiltonian
from QrispQCCSDAnsatz import QrispQCCSDAnsatz
from QrispVQE import QrispVQE
from conftest import MockContext, MockFlowFile, result_to_flowfile


class TestQrispQCCSDAnsatz:
    def test_standalone_spec(self):
        proc = QrispQCCSDAnsatz()
        ctx = MockContext(**{
            "Num Spin Orbitals": "4",
            "Num Electrons": "2",
        })
        res = proc.transform(ctx, MockFlowFile())
        assert res.relationship == "success"
        attrs = res.attributes
        assert attrs["ansatz.format"] == "qrisp_spec"
        assert attrs["ansatz.type"] == "qccsd"
        assert attrs["ansatz.num_spin_orbitals"] == "4"
        assert attrs["ansatz.num_electrons"] == "2"
        assert attrs["ansatz.num_qubits"] == "4"
        assert int(attrs["ansatz.num_parameters"]) > 0

    def test_chain_mode(self):
        # 1. Hamiltonian
        h_proc = QrispHamiltonian()
        h_ctx = MockContext(**{
            "Hamiltonian": "Z0 + Z1 + 0.5 Z2 + 0.5 Z3",
            "Num Qubits": "4",
        })
        h_res = h_proc.transform(h_ctx, MockFlowFile())
        assert h_res.relationship == "success"

        # 2. QCCSD Ansatz
        ansatz_proc = QrispQCCSDAnsatz()
        ansatz_ctx = MockContext(**{
            "Num Spin Orbitals": "4",
            "Num Electrons": "2",
        })
        ansatz_res = ansatz_proc.transform(ansatz_ctx, result_to_flowfile(h_res))
        assert ansatz_res.relationship == "success"
        assert ansatz_res.attributes["hamiltonian.format"] == "sparse_pauli_op_json"
        assert ansatz_res.attributes["ansatz.type"] == "qccsd"

        # 3. QrispVQE Solver
        vqe_proc = QrispVQE()
        vqe_ctx = MockContext(**{
            "Optimizer": "COBYLA",
            "Max Iterations": "20",
            "Shots": "256",
            "Initial Parameters": "zeros",
        })
        vqe_res = vqe_proc.transform(vqe_ctx, result_to_flowfile(ansatz_res))
        assert vqe_res.relationship == "success"
        vqe_attrs = vqe_res.attributes
        assert vqe_attrs["vqe.framework"] == "qrisp"
        assert "vqe.optimal_value" in vqe_attrs
        assert float(vqe_attrs["vqe.optimal_value"]) is not None


class TestQrispVQEEnergyPrecision:
    def _h2_ansatz_flowfile(self):
        from MoleculeHamiltonian import MoleculeHamiltonian

        ham = MoleculeHamiltonian().transform(MockContext(**{
            "Molecule Geometry": "H 0 0 0; H 0 0 0.735",
            "Basis Set": "sto-3g", "Charge": "0", "Multiplicity": "1",
            "Compute Reference Energies": "true",
        }), MockFlowFile())
        assert ham.relationship == "success"
        ff = result_to_flowfile(ham)
        ans = QrispQCCSDAnsatz().transform(
            MockContext(**{"Num Spin Orbitals": "0", "Num Electrons": "2"}), ff)
        assert ans.relationship == "success"
        return ff, result_to_flowfile(ans)

    def test_tight_precision_reaches_fci(self):
        ham_ff, ff = self._h2_ansatz_flowfile()
        fci = float(ham_ff.getAttribute("hamiltonian.fci_energy"))
        res = QrispVQE().transform(MockContext(**{
            "Optimizer": "COBYLA", "Max Iterations": "100", "Shots": "256",
            "Initial Parameters": "zeros", "Random Seed": "3",
            "Energy Precision": "0.0005",
        }), ff)
        assert res.relationship == "success"
        assert res.attributes["vqe.energy_precision"] == "0.0005"
        assert abs(float(res.attributes["vqe.optimal_value"]) - fci) < 0.003

    def test_invalid_precision_fails(self):
        _, ff = self._h2_ansatz_flowfile()
        res = QrispVQE().transform(MockContext(**{
            "Optimizer": "COBYLA", "Max Iterations": "10", "Shots": "256",
            "Initial Parameters": "zeros", "Energy Precision": "0",
        }), ff)
        assert res.relationship == "failure"
        assert "vqe.error" in res.attributes
