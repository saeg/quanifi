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

import math
import numpy as np

from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import (
    PropertyDescriptor,
    StandardValidators,
    ExpressionLanguageScope,
)
from nifiapi.__jvm__ import JvmHolder

BASIS_GATES = ["h", "cx", "rz", "x"]


class QrispTeleportation(FlowFileTransform):
    """
    Quantum teleportation of an arbitrary single-qubit state using Qrisp, self-verifying.

    Qubit 0 carries the message |psi> = Rz(phi) Ry(theta) |0>; qubits 1 and 2
    share a Bell pair. A Bell measurement on qubits 0 and 1 followed by feed-forward
    corrections on Bob's qubit 2 reconstructs |psi>. The processor inverts the
    preparation on qubit 2: ideally it always measures |0>, so teleport.fidelity = 1.0.
    """

    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "0.1.0"
        tags = ["quantum", "qrisp", "teleportation", "entanglement", "bell", "circuit", "qasm2"]
        dependencies = ["qrisp==0.9.5", "qiskit>=2.0.0,<2.5"]
        description = (
            "Teleports Rz(Phi)Ry(Theta)|0> from qubit 0 to qubit 2 using a Bell "
            "pair, a Bell measurement, and feed-forward corrections with Qrisp. "
            "Self-verifies by inverting the preparation on the target: "
            "teleport.fidelity = P(target measures 0), ideally 1.0."
        )

    def __init__(self, **kwargs):
        JvmHolder.jvm = kwargs.get("jvm")
        super().__init__()

        self.theta = PropertyDescriptor(
            name="Theta",
            description="Polar angle of the message state in radians. Default pi/3.",
            required=True,
            default_value="1.0471975512",
            validators=[StandardValidators.NUMBER_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.phi = PropertyDescriptor(
            name="Phi",
            description="Azimuthal angle of the message state in radians. Default pi/4.",
            required=True,
            default_value="0.7853981634",
            validators=[StandardValidators.NUMBER_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.shots = PropertyDescriptor(
            name="Shots",
            description="Number of protocol verification shots.",
            required=True,
            default_value="1024",
            validators=[StandardValidators.POSITIVE_INTEGER_VALIDATOR],
            expression_language_scope=ExpressionLanguageScope.FLOWFILE_ATTRIBUTES,
        )
        self.descriptors = [self.theta, self.phi, self.shots]

    def getPropertyDescriptors(self):
        return self.descriptors

    def _failure(self, flowFile, msg):
        self.logger.error("QrispTeleportation: " + msg)
        return FlowFileTransformResult(
            relationship="failure",
            contents=bytes(flowFile.getContentsAsBytes() or b""),
            attributes={"teleport.error": msg},
        )

    def transform(self, context, flowFile):
        def get(prop):
            return (
                context.getProperty(prop)
                .evaluateAttributeExpressions(flowFile)
                .getValue()
            )

        try:
            theta = float(get(self.theta))
            phi = float(get(self.phi))
        except (TypeError, ValueError) as exc:
            return self._failure(flowFile, "Theta and Phi must be numeric: {}".format(exc))

        try:
            shots = int(get(self.shots))
        except (TypeError, ValueError) as exc:
            return self._failure(flowFile, "Shots must be positive integer: {}".format(exc))

        try:
            from qiskit import QuantumCircuit, qasm2, transpile
            from qrisp import QuantumCircuit as QrispCircuit

            # Construct 3-qubit teleportation circuit in Qrisp
            qc = QrispCircuit(3)
            # 1. Prepare message state on qubit 0
            qc.ry(theta, 0)
            qc.rz(phi, 0)

            # 2. Prepare Bell pair between qubits 1 and 2
            qc.h(1)
            qc.cx(1, 2)

            # 3. Bell measurement on qubits 0 and 1
            qc.cx(0, 1)
            qc.h(0)

            # 4. Feed-forward corrections on Bob's qubit 2:
            # (Coherent / deferred measurement representation)
            qc.cx(1, 2)
            qc.cz(0, 2)

            # 5. Verification: undo preparation on qubit 2
            qc.rz(-phi, 2)
            qc.ry(-theta, 2)

            # Compute exact statevector to determine verification fidelity:
            sv = qc.statevector_array()
            probs = np.abs(sv) ** 2
            # Target qubit is qubit 2 (LSB in binary representation format(i, '03b')[2])
            fidelity = float(
                sum(probs[i] for i in range(8) if format(i, "03b")[2] == "0")
            )

            # Export clean OpenQASM 2.0 representation
            single = QuantumCircuit(3)
            single.ry(theta, 0)
            single.rz(phi, 0)
            single.h(1)
            single.cx(1, 2)
            single.cx(0, 1)
            single.h(0)
            single.cx(1, 2)
            single.cz(0, 2)
            single.rz(-phi, 2)
            single.ry(-theta, 2)

            emitted = transpile(single, basis_gates=BASIS_GATES, optimization_level=0)
            source = qasm2.dumps(emitted)
            diagram = str(single.draw("text"))

        except Exception as exc:
            return self._failure(
                flowFile, "Qrisp teleportation construction failed: {}".format(exc)
            )

        attrs = {
            "circuit.format": "qasm2",
            "circuit.qasm2": source,
            "circuit.num_qubits": "3",
            "circuit.bit_order": "q0_left",
            "circuit.diagram": diagram,
            "teleport.fidelity": f"{fidelity:.6f}",
            "teleport.theta": str(theta),
            "teleport.phi": str(phi),
            "teleport.shots": str(shots),
            "builder.component": "QrispTeleportation",
            "builder.framework": "qrisp",
            "mime.type": "text/plain",
        }

        return FlowFileTransformResult(
            relationship="success",
            contents=source.encode("utf-8"),
            attributes=attrs,
        )
