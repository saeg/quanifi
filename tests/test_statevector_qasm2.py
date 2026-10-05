"""QASM2-only content must run without a companion QASM3 attribute."""

import pytest

from conftest import MockContext, MockFlowFile
from QiskitStatevectorSimulator import QiskitStatevectorSimulator


@pytest.mark.parametrize("format_name", ["qasm2", "qasm"])
def test_qasm2_only_content_preserves_declared_qubit_order(tmp_path, format_name):
    result = QiskitStatevectorSimulator().transform(
        MockContext(**{"Reports Directory": str(tmp_path), "Flow Name": "qasm2"}),
        MockFlowFile(
            content=b'OPENQASM 2.0; include "qelib1.inc"; qreg z_main[1]; qreg a_aux[1]; x z_main[0];',
            attributes={"circuit.format": format_name},
        ),
    )
    assert result.relationship == "success"
    assert result.attributes["sim.top_result"] == "10"
    assert result.attributes["sim.bit_order"] == "q0_left"


def test_qrisp_builder_qasm2_runs_without_qasm3_attribute(tmp_path):
    from QrispQFTCircuit import QrispQFTCircuit

    built = QrispQFTCircuit().transform(
        MockContext(**{"Qubit Count": "2", "Inverse": "false", "Do Swaps": "true"}),
        MockFlowFile(),
    )
    assert built.relationship == "success"
    assert "circuit.qasm3" not in built.attributes
    result = QiskitStatevectorSimulator().transform(
        MockContext(**{"Reports Directory": str(tmp_path), "Flow Name": "qrisp-qft"}),
        MockFlowFile(content=built.contents, attributes=built.attributes),
    )
    assert result.relationship == "success"
    assert result.attributes["circuit.num_qubits"] == "2"


def test_malformed_qasm2_routes_parser_error(tmp_path):
    result = QiskitStatevectorSimulator().transform(
        MockContext(**{"Reports Directory": str(tmp_path)}),
        MockFlowFile(content=b"OPENQASM 2.0; invalid;", attributes={"circuit.format": "qasm2"}),
    )
    assert result.relationship == "failure"
    assert "QASM2" in result.attributes["sim.error"]
