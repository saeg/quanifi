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

import contextlib
import io

from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import (
    PropertyDescriptor,
    StandardValidators,
    ExpressionLanguageScope,
)
from nifiapi.__jvm__ import JvmHolder

MAX_QUBITS = 8
BASIS_GATES = ["h", "cx", "rz", "x"]


class QrispPhaseOracle(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "0.1.0"
        tags = ["quantum", "qrisp", "oracle", "grover", "circuit", "qasm2", "builder"]
        dependencies = ["qrisp==0.9.5", "qiskit>=2.0.0,<2.5"]
        description = (
            "Builds a phase oracle circuit for a given target bitstring using Qrisp. "
            "The oracle applies a -1 phase to |target⟩ and leaves all other states unchanged. "
            "In standalone mode it outputs a bare oracle circuit (no state preparation). "
            "In compose mode it appends the oracle to an existing quantum circuit. "
            "Connect to QrispGroverOperator to build the full Grover search, or feed into "
            "QrispAmplitudeAmplification. Mirrors QiskitPhaseOracle and CirqPhaseOracle so "
            "the processors are interchangeable on the canvas."
        )

    def __init__(self, **kwargs):
        JvmHolder.jvm = kwargs.get("jvm")
        super().__init__()

        self.marked_state = PropertyDescriptor(
            name="Marked State",
            description=(
                "Target bitstring the oracle marks with a -1 phase, e.g. '110'. "
                "Length sets the number of qubits. Bit order is left-to-right "
                "(qubit 0 = leftmost character)."
            ),
            required=True,
            default_value="11",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.insert_barriers = PropertyDescriptor(
            name="Insert Barriers",
            description="Wrap the oracle in barriers so its boundary is visible in circuit diagrams.",
            required=True,
            default_value="false",
            allowable_values=["true", "false"],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.output_format = PropertyDescriptor(
            name="Output Format",
            description="Output circuit serialization. Qrisp exports portable OpenQASM 2.0.",
            required=True,
            default_value="qasm2",
            allowable_values=["qasm2"],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.descriptors = [
            self.marked_state,
            self.insert_barriers,
            self.output_format,
        ]

    def getPropertyDescriptors(self):
        return self.descriptors

    def _failure(self, flowFile, msg):
        self.logger.error("QrispPhaseOracle: " + msg)
        return FlowFileTransformResult(
            relationship="failure",
            contents=bytes(flowFile.getContentsAsBytes() or b""),
            attributes={"circuit.error": msg},
        )

    def transform(self, context, flowFile):
        def get(prop):
            return (
                context.getProperty(prop)
                .evaluateAttributeExpressions(flowFile)
                .getValue()
            )

        target = (get(self.marked_state) or "").strip()
        if not target or set(target) - {"0", "1"}:
            return self._failure(
                flowFile,
                "Marked State must be a nonempty binary string, got {!r}".format(target),
            )
        n = len(target)
        if n > MAX_QUBITS:
            return self._failure(
                flowFile,
                "Marked State must have 1..8 bits, got {}".format(n),
            )

        insert_barriers = get(self.insert_barriers).lower() == "true"
        fmt = get(self.output_format)
        if fmt != "qasm2":
            return self._failure(
                flowFile,
                "Output Format must be qasm2, got {!r}".format(fmt),
            )

        try:
            from qiskit import QuantumCircuit, qasm2, transpile
            from qrisp import QuantumVariable
            from qrisp.grover import tag_state

            # Build the Qrisp phase oracle:
            with contextlib.redirect_stdout(io.StringIO()):
                qv = QuantumVariable(n)
                # target[::-1] maps q0-left convention to Qrisp little-endian variable order
                tag_state({qv: target[::-1]}, binary_values=True)
                compiled = qv.qs.compile().to_qiskit()

            single_oracle = QuantumCircuit(n)
            single_oracle.compose(compiled, qubits=list(range(n)), inplace=True)

            incoming_fmt = flowFile.getAttribute("circuit.format")
            raw_bytes = bytes(flowFile.getContentsAsBytes() or b"")

            if incoming_fmt and raw_bytes:
                # Compose mode: append oracle to existing circuit
                if incoming_fmt != "qasm2":
                    return self._failure(
                        flowFile,
                        "Unsupported incoming circuit format '{}'; expected 'qasm2'".format(
                            incoming_fmt
                        ),
                    )
                upstream = qasm2.loads(
                    raw_bytes.decode("utf-8"),
                    custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
                )
                if upstream.num_qubits != n:
                    return self._failure(
                        flowFile,
                        "Incoming circuit qubit count ({}) does not match Marked State length ({})".format(
                            upstream.num_qubits, n
                        ),
                    )
                combined = QuantumCircuit(n)
                combined.compose(upstream, inplace=True)
                if insert_barriers:
                    combined.barrier()
                combined.compose(single_oracle, inplace=True)
                if insert_barriers:
                    combined.barrier()
                target_circuit = combined
            else:
                # Standalone mode: bare oracle
                target_circuit = QuantumCircuit(n)
                if insert_barriers:
                    target_circuit.barrier()
                target_circuit.compose(single_oracle, inplace=True)
                if insert_barriers:
                    target_circuit.barrier()

            emitted = transpile(
                target_circuit, basis_gates=BASIS_GATES, optimization_level=0
            )
            source = qasm2.dumps(emitted)
            parsed = qasm2.loads(
                source, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
            )
            diagram = str(target_circuit.draw("text"))
        except Exception as exc:
            return self._failure(
                flowFile, "Qrisp phase oracle construction failed: {}".format(exc)
            )

        ops = parsed.count_ops()
        attrs = {
            "circuit.format": "qasm2",
            "circuit.qasm2": source,
            "circuit.qasm3": "",
            "circuit.cirq_json": "",
            "circuit.svg": "",
            "circuit.num_qubits": str(n),
            "circuit.marked_state": target,
            "circuit.bit_order": "q0_left",
            "circuit.diagram": diagram,
            "circuit.depth": str(parsed.depth()),
            "circuit.gate_count": str(
                sum(v for k, v in ops.items() if k not in ("barrier", "measure"))
            ),
            "circuit.nonlocal_gates": str(parsed.num_nonlocal_gates()),
            "circuit.t_count": str(ops.get("t", 0) + ops.get("tdg", 0)),
            "builder.component": "QrispPhaseOracle",
            "builder.framework": "qrisp",
            "mime.type": "text/plain",
        }

        return FlowFileTransformResult(
            relationship="success",
            contents=source.encode("utf-8"),
            attributes=attrs,
        )
