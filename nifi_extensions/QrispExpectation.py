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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import (
    PropertyDescriptor,
    StandardValidators,
    ExpressionLanguageScope,
)
from nifiapi.__jvm__ import JvmHolder


class QrispExpectation(FlowFileTransform):
    """
    Evaluates the expectation value of a Hermitian Pauli-sum observable on a quantum
    circuit using Qrisp. Reads OpenQASM 2.0 circuit content and a Hamiltonian property
    (or upstream hamiltonian.json), computing exact analytical or sampled expectations.
    """

    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "0.1.0"
        tags = [
            "quantum",
            "qrisp",
            "expectation",
            "observable",
            "hamiltonian",
            "simulation",
            "vqe",
        ]
        dependencies = ["qrisp==0.9.5", "qiskit>=2.0.0,<2.5"]
        description = (
            "Evaluates the expectation value of a Hamiltonian on a quantum circuit "
            "using Qrisp. Reads a qasm2 circuit and an indexed Pauli expression or "
            "neutral Hamiltonian JSON property. Outputs expectation JSON and sets "
            "sim.expectation attributes."
        )

    def __init__(self, **kwargs):
        JvmHolder.jvm = kwargs.get("jvm")
        super().__init__()

        self.hamiltonian = PropertyDescriptor(
            name="Hamiltonian",
            description=(
                "Indexed Pauli expression (e.g. 'Z0 + 0.5 X1') or sparse_pauli_op_json. "
                "By default uses upstream hamiltonian.json or attribute, otherwise Z0."
            ),
            required=True,
            default_value="Z0",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.mode = PropertyDescriptor(
            name="Mode",
            description="Evaluation mode: 'exact' (analytical statevector) or 'sampled'.",
            required=True,
            default_value="exact",
            allowable_values=["exact", "sampled"],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.shots = PropertyDescriptor(
            name="Shots",
            description="Number of measurement shots when Mode is 'sampled'.",
            required=True,
            default_value="1024",
            validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.descriptors = [self.hamiltonian, self.mode, self.shots]

    def getPropertyDescriptors(self):
        return self.descriptors

    def _failure(self, flowFile, msg):
        self.logger.error("QrispExpectation: " + msg)
        return FlowFileTransformResult(
            relationship="failure",
            contents=bytes(flowFile.getContentsAsBytes() or b""),
            attributes={"sim.error": msg},
        )

    def transform(self, context, flowFile):
        def get(prop):
            return (
                context.getProperty(prop)
                .evaluateAttributeExpressions(flowFile)
                .getValue()
            )

        raw_bytes = bytes(flowFile.getContentsAsBytes() or b"")
        if not raw_bytes:
            return self._failure(flowFile, "FlowFile contains no circuit content.")

        h_expr = get(self.hamiltonian)
        mode = (get(self.mode) or "exact").lower()
        try:
            shots = int(get(self.shots))
        except (TypeError, ValueError) as exc:
            return self._failure(flowFile, f"Invalid shots value: {exc}")

        try:
            from pauli_dsl import parse_pauli_sum, PauliDSLError
            from qiskit import qasm2
            import qrisp.operators as ops
            from qrisp import QuantumCircuit as QrispCircuit

            # Parse incoming circuit
            qc_qiskit = qasm2.loads(
                raw_bytes.decode("utf-8"),
                custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            )
            qc_unmeasured = qc_qiskit.remove_final_measurements(inplace=False)
            n_circuit = qc_unmeasured.num_qubits

            # Parse Hamiltonian
            terms, n_ham = parse_pauli_sum(h_expr, n_circuit)
            n = max(n_circuit, n_ham)

            P_map = {"X": ops.X, "Y": ops.Y, "Z": ops.Z}
            total_H = 0
            for pauli_str, indices, coeff in terms:
                if not pauli_str:
                    total_H += coeff
                else:
                    term_op = coeff
                    for p, idx in zip(pauli_str, indices):
                        term_op = term_op * P_map[p](idx)
                    total_H = total_H + term_op

            # Statevector via Qrisp
            qc_qrisp = QrispCircuit.from_qiskit(qc_unmeasured)
            sv = qc_qrisp.statevector_array()

            if len(sv) != (1 << n):
                # Pad statevector if Hamiltonian uses higher qubit index
                padded = np.zeros(1 << n, dtype=complex)
                padded[: len(sv)] = sv
                sv = padded

            # Build matrix representation
            if hasattr(total_H, "to_sparse_matrix"):
                mat = total_H.to_sparse_matrix(n).toarray()
            else:
                mat = np.eye(1 << n) * float(total_H)

            exact_val = float(np.real(sv.conj().T @ mat @ sv))
            mat_sq = mat @ mat
            exact_val_sq = float(np.real(sv.conj().T @ mat_sq @ sv))
            variance = max(0.0, float(exact_val_sq - exact_val**2))

            if mode == "sampled":
                # Add sampling noise according to normal distribution of sample mean
                std_err = np.sqrt(variance / max(1, shots))
                sampled_val = float(np.random.normal(exact_val, std_err))
                reported_val = sampled_val
            else:
                reported_val = exact_val

            result_obj = {
                "expectation": reported_val,
                "variance": variance,
                "exact_expectation": exact_val,
                "observable": h_expr,
                "mode": mode,
                "shots": shots if mode == "sampled" else 0,
            }
            content_bytes = json.dumps(result_obj, indent=2).encode("utf-8")

        except Exception as exc:
            return self._failure(flowFile, f"Qrisp expectation evaluation failed: {exc}")

        attrs = {
            "sim.expectation": f"{reported_val:.6f}",
            "sim.variance": f"{variance:.6f}",
            "sim.framework": "qrisp",
            "sim.component": "QrispExpectation",
            "circuit.num_qubits": str(n_circuit),
            "mime.type": "application/json",
        }

        return FlowFileTransformResult(
            relationship="success",
            contents=content_bytes,
            attributes=attrs,
        )
