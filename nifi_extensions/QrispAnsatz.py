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

from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import PropertyDescriptor, StandardValidators, ExpressionLanguageScope
from nifiapi.__jvm__ import JvmHolder

from qrisp_ansatz import (  # noqa: F401  (re-exported for tests)
    _ENTANGLER,
    _ROTATIONS,
    _build_ansatz_function,
    _describe,
    _entangle_pairs,
    _num_params_per_layer,
)


class QrispAnsatz(FlowFileTransform):
    """
    Declares a parameterised Qrisp ansatz for variational algorithms.

    State-preparation stage of QrispHamiltonian -> QrispAnsatz -> QrispVQE ->
    QuanifiReport. Unlike Qiskit (QPY) and Cirq (Cirq JSON), a Qrisp ansatz is a
    Python *callable* ``ansatz_function(qv, theta)`` consumed by ``VQEProblem``,
    not a serialisable circuit. So the cross-processor carrier is a compact
    **spec** (``ansatz.type`` / ``ansatz.reps`` / ``ansatz.entanglement`` /
    ``ansatz.num_qubits``); QrispVQE reconstructs the callable from it. The
    ``ansatz.*`` attribute contract is otherwise the same as the other frameworks.

    Two modes:
      Chain mode — a Hamiltonian is on the FlowFile: its content passes through
        and the qubit count comes from ``hamiltonian.num_qubits``.
      Standalone — no Hamiltonian: writes the spec as JSON content; sized by
        ``Num Qubits``.
    """

    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']

    class ProcessorDetails:
        version = "0.1.0"
        description = (
            "Declares a parameterised Qrisp variational ansatz (efficient_su2, "
            "real_amplitudes, or two_local) as a spec carried in ansatz.* attributes. "
            "A Qrisp ansatz is a callable, not a serialisable circuit, so QrispVQE "
            "reconstructs it from the spec. In chain mode the incoming Hamiltonian passes "
            "through."
        )
        tags = ["quantum", "qrisp", "vqe", "ansatz", "variational"]
        # _build_ansatz_function lazily imports qrisp gates, so the package is
        # required at runtime even though this processor only emits a spec.
        dependencies = ["qrisp==0.9.5", "qiskit>=2.0.0,<2.5"]

    def __init__(self, **kwargs):
        JvmHolder.jvm = kwargs.get('jvm')
        super().__init__()

        self.ansatz_type = PropertyDescriptor(
            name="Ansatz Type",
            description=(
                "efficient_su2 = RY+RZ rotations + CZ entanglers (default). "
                "real_amplitudes = RY rotations + CZ entanglers. "
                "two_local = RY rotations + CX entanglers."
            ),
            required=True,
            default_value="efficient_su2",
            allowable_values=["efficient_su2", "real_amplitudes", "two_local"],
        )
        self.reps = PropertyDescriptor(
            name="Reps",
            description="Number of ansatz layers (VQEProblem 'depth'). Total parameters = params-per-layer * reps.",
            required=True,
            default_value="2",
            validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.entanglement = PropertyDescriptor(
            name="Entanglement",
            description="Entanglement pattern between qubits in each entangling layer.",
            required=True,
            default_value="full",
            allowable_values=["full", "linear", "circular"],
        )
        self.num_qubits = PropertyDescriptor(
            name="Num Qubits",
            description="Qubit count in standalone mode. Ignored in chain mode (taken from hamiltonian.num_qubits).",
            required=True,
            default_value="2",
            validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.descriptors = [self.ansatz_type, self.reps, self.entanglement, self.num_qubits]

    def getPropertyDescriptors(self):
        return self.descriptors

    def transform(self, context, flowFile):
        get = lambda prop: (
            context.getProperty(prop)
            .evaluateAttributeExpressions(flowFile)
            .getValue()
        )

        atype = get(self.ansatz_type)
        try:
            reps = int(get(self.reps))
        except (TypeError, ValueError) as exc:
            msg = "bad numeric property value: {}".format(exc)
            self.logger.error("QrispAnsatz: " + msg)
            return FlowFileTransformResult(
                relationship="failure", contents=b"",
                attributes={"ansatz.error": msg},
            )
        entanglement = get(self.entanglement)

        ham_format = flowFile.getAttribute("hamiltonian.format")
        ham_qubits = flowFile.getAttribute("hamiltonian.num_qubits")
        chain_mode = ham_format is not None and ham_qubits is not None
        try:
            n = int(ham_qubits) if chain_mode else int(get(self.num_qubits))
        except (TypeError, ValueError) as exc:
            msg = "bad numeric property value: {}".format(exc)
            self.logger.error("QrispAnsatz: " + msg)
            return FlowFileTransformResult(
                relationship="failure", contents=b"",
                attributes={"ansatz.error": msg},
            )

        per_layer = _num_params_per_layer(atype, n)
        total = per_layer * reps
        description = _describe(atype, n, entanglement)

        spec = {
            "type": atype, "reps": reps, "entanglement": entanglement,
            "num_qubits": n, "num_params_per_layer": per_layer, "num_parameters": total,
        }

        self.logger.warn("QrispAnsatz ({} qubits, {} reps, {} params): {}".format(
            n, reps, total, description))

        attrs = {
            "ansatz.format": "qrisp_spec",
            "ansatz.type": atype,
            "ansatz.reps": str(reps),
            "ansatz.entanglement": entanglement,
            "ansatz.num_qubits": str(n),
            "ansatz.num_params_per_layer": str(per_layer),
            "ansatz.num_parameters": str(total),
            "ansatz.framework": "qrisp",
            "ansatz.diagram": description,
        }

        if chain_mode:
            content = bytes(flowFile.getContentsAsBytes() or b"")
            for key in ("hamiltonian.format", "hamiltonian.num_qubits",
                        "hamiltonian.num_terms", "hamiltonian.framework",
                        "hamiltonian.expression"):
                val = flowFile.getAttribute(key)
                if val is not None:
                    attrs[key] = val
        else:
            content = json.dumps(spec, indent=2).encode("utf-8")
            attrs["circuit.framework"] = "qrisp"
            attrs["circuit.algorithm"] = "ansatz"
            attrs["circuit.num_qubits"] = str(n)
            attrs["circuit.num_parameters"] = str(total)

        return FlowFileTransformResult(
            relationship="success",
            contents=content,
            attributes=attrs,
        )
