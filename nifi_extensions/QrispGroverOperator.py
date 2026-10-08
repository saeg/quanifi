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

BASIS_GATES = ["h", "cx", "rz", "x"]


class QrispGroverOperator(FlowFileTransform):
    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "0.1.0"
        tags = ["quantum", "qrisp", "grover", "operator", "diffuser", "circuit", "qasm2"]
        dependencies = ["qrisp==0.9.5", "qiskit>=2.0.0,<2.5"]
        description = (
            "Reads an oracle circuit from the FlowFile content (produced by QrispPhaseOracle, "
            "or by QiskitPhaseOracle / CirqPhaseOracle with Output Format 'qasm2'), prepends "
            "the uniform superposition H⊗n, and applies the full Grover operator (oracle + diffuser) "
            "the requested number of times via Qrisp's native diffuser. "
            "Mirrors QiskitGroverOperator and CirqGroverOperator so they are interchangeable "
            "on the canvas. No measurement is added — connect to QrispSimulator, QiskitAerSimulator, "
            "or CirqSimulator to run the circuit."
        )

    def __init__(self, **kwargs):
        JvmHolder.jvm = kwargs.get("jvm")
        super().__init__()

        self.num_iterations = PropertyDescriptor(
            name="Num Iterations",
            description=(
                "Number of times the Grover operator (oracle + diffuser) is applied. "
                "Optimal is roughly floor(pi/4 * sqrt(2^n)) for a single marked state."
            ),
            required=True,
            default_value="1",
            validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.insert_barriers = PropertyDescriptor(
            name="Insert Barriers",
            description="Add barriers between oracle and diffuser stages.",
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
            self.num_iterations,
            self.insert_barriers,
            self.output_format,
        ]

    def getPropertyDescriptors(self):
        return self.descriptors

    def _failure(self, flowFile, msg):
        self.logger.error("QrispGroverOperator: " + msg)
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
        if not incoming_fmt or not raw_bytes:
            return self._failure(
                flowFile,
                "No incoming circuit found on FlowFile. QrispGroverOperator requires "
                "an oracle circuit from an upstream processor (e.g. QrispPhaseOracle).",
            )

        if incoming_fmt != "qasm2":
            return self._failure(
                flowFile,
                "Unsupported incoming circuit format '{}'; expected 'qasm2'".format(
                    incoming_fmt
                ),
            )

        try:
            from qiskit import QuantumCircuit, qasm2, transpile
            from qrisp import QuantumVariable
            from qrisp.grover import diffuser

            oracle_qc = qasm2.loads(
                raw_bytes.decode("utf-8"),
                custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            )
            n = oracle_qc.num_qubits

            # Build Qrisp diffuser:
            with contextlib.redirect_stdout(io.StringIO()):
                qv = QuantumVariable(n)
                diffuser(qv)
                compiled_diffuser = qv.qs.compile().to_qiskit()

            single_diffuser = QuantumCircuit(n)
            single_diffuser.compose(compiled_diffuser, qubits=list(range(n)), inplace=True)

            # Assemble full Grover circuit:
            # 1. Uniform superposition H on all qubits
            full_qc = QuantumCircuit(n)
            for i in range(n):
                full_qc.h(i)

            # 2. Iterate (oracle + diffuser) num_iterations times
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
                flowFile, "Qrisp Grover operator assembly failed: {}".format(exc)
            )

        ops = parsed.count_ops()
        marked_state = flowFile.getAttribute("circuit.marked_state") or ""
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
            "builder.component": "QrispGroverOperator",
            "builder.framework": "qrisp",
            "mime.type": "text/plain",
        }
        if marked_state:
            attrs["circuit.marked_state"] = marked_state

        return FlowFileTransformResult(
            relationship="success",
            contents=source.encode("utf-8"),
            attributes=attrs,
        )
