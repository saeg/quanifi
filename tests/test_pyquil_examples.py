"""The shipped canvas and the executed examples share real processor settings."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pyquil_examples", ROOT / "tools/build_pyquil_examples.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_canvas_is_current_stopped_connected_and_has_valid_properties():
    snapshot = json.loads((ROOT / "demo/pyquil/pyquil-examples.json").read_text())
    assert snapshot == tool.snapshot()
    group = snapshot["flowContents"]
    all_components = (
        group["processors"] + group["funnels"] + group["connections"] + group["labels"]
    )
    ids = [c["identifier"] for c in all_components]
    assert len(ids) == len(set(ids))
    node_ids = {c["identifier"] for c in group["processors"] + group["funnels"]}
    assert len(group["funnels"]) == 3
    for c in group["connections"]:
        assert c["source"]["id"] in node_ids and c["destination"]["id"] in node_ids
    for p in group["processors"]:
        assert p["scheduledState"] == "ENABLED"
        if p["bundle"]["artifact"] == "python-extensions":
            cls = tool.processor_class(p["type"])
            assert p["bundle"]["version"] == cls.ProcessorDetails.version
            descriptors = {d.name: d for d in cls().getPropertyDescriptors()}
            assert set(p["properties"]) <= set(descriptors)
            for name, value in p["properties"].items():
                choices = descriptors[name].allowable_values
                assert not choices or value in choices
            outgoing = {
                r
                for c in group["connections"]
                if c["source"]["id"] == p["identifier"]
                for r in c["selectedRelationships"]
            }
            assert outgoing | set(p["autoTerminatedRelationships"]) == {
                "success",
                "failure",
                "original",
            }
            assert "failure" in outgoing


def test_run_canvas_examples_and_generate_reports(tmp_path):
    results = tool.execute(tmp_path)
    assert results["grover"]["counts"] == {"10": 256}
    assert float(results["vqe"]["attributes"]["vqe.optimal_value"]) == pytest.approx(
        -1.11803398875, abs=1e-7
    )
    qaoa_attrs = results["qaoa"]["attributes"]
    assert float(qaoa_attrs["qaoa.optimal_value"]) == pytest.approx(-1, abs=1e-7)
    assert qaoa_attrs["qaoa.best_value"] == "-1.0"
    assert qaoa_attrs["qaoa.approximation_ratio"] == "1.0"
    qaoa_counts = results["qaoa"]["counts"]
    assert set(qaoa_counts) <= {"01", "10"}
    assert sum(qaoa_counts.values()) == 256
    for key, record in results.items():
        saved = json.loads((ROOT / "demo/pyquil" / f"{key}.json").read_text())
        if key == "vqe":
            assert record["description"] == saved["description"]
            assert record["stages"] == saved["stages"]
            assert record["counts"] == saved["counts"]
            vqe_floats = {"vqe.optimal_parameters", "vqe.optimal_value"}
            rec_attrs = {
                k: v for k, v in record["attributes"].items() if k not in vqe_floats
            }
            saved_attrs = {
                k: v for k, v in saved["attributes"].items() if k not in vqe_floats
            }
            assert rec_attrs == saved_attrs
            assert float(record["attributes"]["vqe.optimal_value"]) == pytest.approx(
                float(saved["attributes"]["vqe.optimal_value"]), abs=1e-5
            )
            import ast

            rec_params = ast.literal_eval(record["attributes"]["vqe.optimal_parameters"])
            saved_params = ast.literal_eval(saved["attributes"]["vqe.optimal_parameters"])
            assert rec_params == pytest.approx(saved_params, abs=1e-5)
        elif key == "qaoa":
            assert record["description"] == saved["description"]
            assert record["stages"] == saved["stages"]
            assert record["counts"] == saved["counts"]
            float_attrs = {
                "qaoa.optimal_value",
                "qaoa.optimal_parameters",
                "qaoa.betas",
                "qaoa.gammas",
            }
            rec_attrs = {
                k: v for k, v in record["attributes"].items() if k not in float_attrs
            }
            saved_attrs = {
                k: v for k, v in saved["attributes"].items() if k not in float_attrs
            }
            assert rec_attrs == saved_attrs
            assert float(record["attributes"]["qaoa.optimal_value"]) == pytest.approx(
                float(saved["attributes"]["qaoa.optimal_value"]), abs=1e-5
            )
        else:
            assert record == saved
        assert (tmp_path / f"pyquil-{key}.html").is_file()
        assert (tmp_path / f"{key}.qasm").is_file()
