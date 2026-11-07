"""Shared helpers and processor registries for Quanifi smoke test submodules."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "nifi_extensions"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import (  # noqa: E402
    MockContext,
    MockFlowFile,
    result_to_flowfile,
    result_to_flowfile_merged,
)

# Registry of exercised processors across test executions
EXERCISED_PROCESSORS: set[str] = set()


def mark_exercised(name: str) -> None:
    """Register a processor class name as exercised."""
    EXERCISED_PROCESSORS.add(name)


def qasm_ff(
    qasm: str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n',
    fmt: str = "qasm2",
) -> MockFlowFile:
    """Return a mock FlowFile carrying OpenQASM content and format attribute."""
    return MockFlowFile(content=qasm.encode(), attributes={"circuit.format": fmt})


# Subsystem processor definitions (122 total)
QISKIT_PROCESSORS = {
    # Builders (14)
    "QiskitAmplitudeAmplification", "QiskitAnsatz", "QiskitBellState",
    "QiskitGroverCircuit", "QiskitGroverOperator", "QiskitHadamardTransform",
    "QiskitPhaseEstimation", "QiskitPhaseOracle", "QiskitQAOACircuit",
    "QiskitQFTCircuit", "QiskitQuantumArithmetic", "QiskitQuantumHelloWorld",
    "QiskitStatePreparation", "QiskitTeleportation",
    # Simulators (2)
    "QiskitAerSimulator", "QiskitStatevectorSimulator",
    # Solvers & Algorithms (4)
    "QiskitAmplitudeEstimation", "QiskitGroverSearch", "QiskitQAOA", "QiskitVQE",
    # Hamiltonians & Hardware (2)
    "QiskitHamiltonian", "QiskitRuntimeSampler",
}

QRISP_PROCESSORS = {
    # Builders (19)
    "QrispAmplitudeAmplification", "QrispAnsatz", "QrispBernsteinVazirani",
    "QrispDeutschJozsa", "QrispGroverCircuit", "QrispGroverOperator",
    "QrispHadamardTransform", "QrispPhaseEstimation", "QrispPhaseOracle",
    "QrispQAOACircuit", "QrispQCCSDAnsatz", "QrispQFTCircuit",
    "QrispQuantumArithmetic", "QrispShor", "QrispStatePreparation",
    "QrispSwapTest", "QrispTeleportation", "QrispTruthTableSynthesis",
    "QrispTypeEncoder",
    # Simulators (3)
    "QrispExpectation", "QrispSimulator", "QrispStatevectorSimulator",
    # Solvers & Algorithms (5)
    "QrispAmplitudeEstimation", "QrispGroverSearch", "QrispQAOA", "QrispQIRO", "QrispVQE",
    # Hamiltonians & Hardware (2)
    "QrispHamiltonian", "QrispIQMDevice",
}

CIRQ_PROCESSORS = {
    # Builders (10)
    "CirqAnsatz", "CirqGroverCircuit", "CirqGroverOperator",
    "CirqHadamardTransform", "CirqPhaseEstimation", "CirqPhaseOracle",
    "CirqQAOACircuit", "CirqQFTCircuit", "CirqQuantumArithmetic",
    "CirqStatePreparation",
    # Simulators (2)
    "CirqSimulator", "CirqStatevectorSimulator",
    # Solvers & Algorithms (3)
    "CirqAmplitudeEstimation", "CirqQAOA", "CirqVQE",
    # Hamiltonians (1)
    "CirqHamiltonian",
}

PENNYLANE_PROCESSORS = {
    # Builders (5)
    "PennylaneGroverCircuit", "PennylanePhaseEstimation", "PennylaneQAOACircuit",
    "PennylaneQuantumArithmetic", "PennylaneVariationalAnsatz",
    # Simulators (2)
    "PennylaneExpectation", "PennylaneSimulator",
    # Solvers & Algorithms (2)
    "PennylaneAmplitudeEstimation", "PennylaneQAOA",
    # QML Lane (3)
    "PennylaneDatasetLoader", "PennylaneFeatureEmbedding",
    "PennylaneVariationalClassifier",
}

PYQUIL_PROCESSORS = {
    # Builders (6)
    "PyquilAnsatz", "PyquilGroverCircuit", "PyquilGroverOperator",
    "PyquilPhaseOracle", "PyquilQAOACircuit", "PyquilQFT",
    # Simulators (2)
    "PyquilExpectation", "PyquilSimulator",
    # Solvers & Algorithms (2)
    "PyquilQAOA", "PyquilVQE",
}

COMMON_PROCESSORS = {
    # Problem Encoders & Hamiltonians (6)
    "MaxCliqueProblem", "MaxCutProblem", "MaxIndependentSetProblem",
    "MoleculeHamiltonian", "PortfolioRebalancingProblem", "QuboToHamiltonian",
    # Differential & Mutation Infrastructure (12)
    "MutationScoreReport", "QuanifiUnitary", "QuantumAssertion",
    "QuantumCalibratedOracle", "QuantumConsensusOracle",
    "QuantumDistributionComparison", "QuantumDistributionOracle",
    "QuantumMutator", "QuantumQAOAEvaluator", "QuantumSuccessProbabilityOracle",
    "QuantumTestCaseSource", "QuantumUnitaryComparison",
    # Reporting & Baseline (3)
    "QiskitCircuitReport", "QuanifiReport", "ReadoutBaselineCircuit",
}

HARDWARE_PROCESSORS = {
    # Gateways & Simulators (3)
    "BraketDevice", "BraketSimulator", "QSharpSimulator",
    # Batch Lane (9)
    "IQMJobPoller", "QuantumBatchPoller", "QuantumBatchResultExpander",
    "QuantumIBMBatchPoller", "QuantumIBMBatchSubmitter",
    "QuantumIQMBatchPoller", "QuantumIQMBatchSubmitter",
    "QuantumInspireBatchPoller", "QuantumInspireBatchSubmitter",
}

ALL_PROCESSORS = (
    QISKIT_PROCESSORS
    | QRISP_PROCESSORS
    | CIRQ_PROCESSORS
    | PENNYLANE_PROCESSORS
    | PYQUIL_PROCESSORS
    | COMMON_PROCESSORS
    | HARDWARE_PROCESSORS
)

__all__ = [
    "ALL_PROCESSORS",
    "CIRQ_PROCESSORS",
    "COMMON_PROCESSORS",
    "EXERCISED_PROCESSORS",
    "HARDWARE_PROCESSORS",
    "MockContext",
    "MockFlowFile",
    "PENNYLANE_PROCESSORS",
    "PYQUIL_PROCESSORS",
    "QISKIT_PROCESSORS",
    "QRISP_PROCESSORS",
    "mark_exercised",
    "qasm_ff",
    "result_to_flowfile",
    "result_to_flowfile_merged",
]
