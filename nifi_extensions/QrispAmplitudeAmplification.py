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


class QrispAmplitudeAmplification(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "0.1.0"
        tags = [
            "quantum",
            "qrisp",
            "amplitude-amplification",
            "grover",
            "qaa",
            "circuit",
            "qasm2",
        ]
        dependencies = ["qrisp==0.9.5", "qiskit>=2.0.0,<2.5"]
        description = (
            "Generalised Quantum Amplitude Amplification (QAA) using Qrisp. "
            "In standalone mode it builds a phase oracle for the Marked State bitstring and "
            "applies uniform-superposition state preparation A = H⊗n and repeated Grover "
            "amplification Q = A·S₀·A†·S_f. "
            "In compose mode it treats the incoming FlowFile as the oracle circuit and wraps "
            "amplitude amplification around it. No measurement is added — connect to a simulator."
        )

    def __init__(self, **kwargs):
        JvmHolder.jvm = kwargs.get("jvm")
        super().__init__()

        self.marked_state = PropertyDescriptor(
            name="Marked State",
            description=(
                "Target bitstring for the standalone phase oracle (e.g. '101' marks |101⟩). "
                "Ignored in compose mode when the incoming circuit defines the oracle. "
                "Bit order is left-to-right (qubit 0 = leftmost character)."
            ),
            required=True,
            default_value="11",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.num_iterations = PropertyDescriptor(
            name="Num Iterations",
            description="Number of amplitude amplification applications Q.",
            required=True,
            default_value="1",
            validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.insert_barriers = PropertyDescriptor(
            name="Insert Barriers",
            description="Add barriers between state preparation, oracle, and diffuser stages.",
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
            self.num_iterations,
            self.insert_barriers,
            self.output_format,
        ]

    def getPropertyDescriptors(self):
        return self.descriptors

    def _failure(self, flowFile, msg):
        self.logger.error("QrispAmplitudeAmplification: " + msg)
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

        try:
            num_iterations = int(get(self.num_iterations))
        except (TypeError, ValueError) as exc:
            return self._failure(flowFile, "bad numeric property value: {}".format(exc))

        if num_iterations < 1:
            return self._failure(
                flowFile, "Num Iterations must be >= 1, got {}".format(num_iterations)
            )

        insert_barriers = get(self.insert_barriers).lower() == "true"
        fmt = get(self.output_format)
        if fmt != "qasm2":
            return self._failure(
                flowFile, "Output Format must be qasm2, got {!r}".format(fmt)
            )

        incoming_fmt = flowFile.getAttribute("circuit.format")
        raw_bytes = bytes(flowFile.getContentsAsBytes() or b"")

        try:
            from qiskit import QuantumCircuit, qasm2, transpile
            from qrisp import QuantumVariable
            from qrisp.grover import tag_state, diffuser

            if incoming_fmt and raw_bytes:
                # Compose mode: incoming circuit is the oracle
                if incoming_fmt != "qasm2":
                    return self._failure(
                        flowFile,
                        "Unsupported incoming circuit format '{}'; expected 'qasm2'".format(
                            incoming_fmt
                        ),
                    )
                oracle_qc = qasm2.loads(
                    raw_bytes.decode("utf-8"),
                    custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
                )
                n = oracle_qc.num_qubits
                target = flowFile.getAttribute("circuit.marked_state") or ""
            else:
                # Standalone mode: build oracle for Marked State
                target = (get(self.marked_state) or "").strip()
                if not target or set(target) - {"0", "1"}:
                    return self._failure(
                        flowFile,
                        "Marked State must be a nonempty binary string, got {!r}".format(
                            target
                        ),
                    )
                n = len(target)
                if n > MAX_QUBITS:
                    return self._failure(
                        flowFile, "Marked State must have 1..8 bits, got {}".format(n)
                    )

                with contextlib.redirect_stdout(io.StringIO()):
                    qv_o = QuantumVariable(n)
                    tag_state({qv_o: target[::-1]}, binary_values=True)
                    compiled_oracle = qv_o.qs.compile().to_qiskit()

                oracle_qc = QuantumCircuit(n)
                oracle_qc.compose(compiled_oracle, qubits=list(range(n)), inplace=True)

            # Build Qrisp diffuser
            with contextlib.redirect_stdout(io.StringIO()):
                qv_d = QuantumVariable(n)
                diffuser(qv_d)
                compiled_diffuser = qv_d.qs.compile().to_qiskit()

            single_diffuser = QuantumCircuit(n)
            single_diffuser.compose(compiled_diffuser, qubits=list(range(n)), inplace=True)

            # State preparation A = H on all qubits
            full_qc = QuantumCircuit(n)
            for i in range(n):
                full_qc.h(i)

            # Apply Q = A S0 A_dag Sf
            for _ in range(num_iterations):
                if insert_barriers:
                    full_qc.barrier()
                full_qc.compose(oracle_qc, inplace=True)
                if insert_barriers:
                    full_qc.barrier()
                full_qc.compose(single_diffuser, inplace=True)

            if insert_barriers:
                full_qc.barrier()

            emitted = transpile(
                full_qc, basis_gates=BASIS_GATES, optimization_level=0
            )
            source = qasm2.dumps(emitted)
            parsed = qasm2.loads(
                source, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
            )
            diagram = str(full_qc.draw("text"))
        except Exception as exc:
            return self._failure(
                flowFile, "Qrisp amplitude amplification assembly failed: {}".format(exc)
            )

        ops = parsed.count_ops()
        attrs = {
            "circuit.format": "qasm2",
            "circuit.qasm2": source,
            "circuit.qasm3": "",
            "circuit.cirq_json": "",
            "circuit.svg": "",
            "circuit.num_qubits": str(n),
            "circuit.num_iterations": str(num_iterations),
            "circuit.bit_order": "q0_left",
            "circuit.diagram": diagram,
            "circuit.depth": str(parsed.depth()),
            "circuit.gate_count": str(
                sum(v for k, v in ops.items() if k not in ("barrier", "measure"))
            ),
            "circuit.nonlocal_gates": str(parsed.num_nonlocal_gates()),
            "circuit.t_count": str(ops.get("t", 0) + ops.get("tdg", 0)),
            "builder.component": "QrispAmplitudeAmplification",
            "builder.framework": "qrisp",
            "mime.type": "text/plain",
        }
        if target:
            attrs["circuit.marked_state"] = target

        return FlowFileTransformResult(
            relationship="success",
            contents=source.encode("utf-8"),
            attributes=attrs,
        )
