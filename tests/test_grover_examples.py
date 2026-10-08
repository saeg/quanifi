"""The shipped Qiskit Grover quickstart canvas matches the generator, is
well formed, ships in the Docker image, and finds its target when run
headlessly with the nifiapi test stubs."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_grover_examples", ROOT / "tools/build_grover_examples.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

SIMPLE_TYPES = [
    "GenerateFlowFile",
    "QiskitPhaseOracle",
    "QiskitGroverOperator",
    "QiskitAerSimulator",
    "QuanifiReport",
]


def test_simple_snapshot_is_current():
    current = json.loads((ROOT / "demo/grover/qiskit-grover.json").read_text())
    assert current == tool.simple_snapshot()


def test_simple_snapshot_structure():
    group = tool.simple_snapshot()["flowContents"]
    procs = group["processors"]

    assert group["name"] == tool.SIMPLE_GROUP_NAME
    assert [p["type"].split(".")[-1] for p in procs] == SIMPLE_TYPES
    assert len(group["connections"]) == 8  # 4 success hops + 4 failure drains
    assert len(group["funnels"]) == 1
    assert len(group["labels"]) == 1

    all_components = (
        procs + group["funnels"] + group["connections"] + group["labels"]
    )
    ids = [c["identifier"] for c in all_components]
    assert len(ids) == len(set(ids))

    node_ids = {c["identifier"] for c in procs + group["funnels"]}
    for c in group["connections"]:
        assert c["source"]["id"] in node_ids
        assert c["destination"]["id"] in node_ids

    successes = [
        c for c in group["connections"] if c["selectedRelationships"] == ["success"]
    ]
    assert [(c["source"]["id"], c["destination"]["id"]) for c in successes] == [
        (a["identifier"], b["identifier"]) for a, b in zip(procs, procs[1:])
    ]

    assert all(p["scheduledState"] != "RUNNING" for p in procs)
    assert procs[0]["schedulingPeriod"] == "1 day"
    assert all(p["schedulingPeriod"] == "0 sec" for p in procs[1:])

    oracle, operator, _simulator, report = procs[1:]
    assert oracle["properties"]["Marked State"] == tool.SIMPLE_TARGET == "10"
    assert oracle["properties"]["Output Format"] == "qasm2"
    assert operator["properties"]["Num Iterations"] == "1"
    assert operator["properties"]["Output Format"] == "qasm2"
    assert report["properties"]["Flow Name"] == tool.SIMPLE_FLOW_NAME
    assert report["properties"]["Reports Directory"] == tool.REPORTS_DIR
    # Sensitive defaults are never written into the committed canvas.
    assert "Reports API Token" not in report["properties"]


def test_python_processors_match_classes():
    group = tool.simple_snapshot()["flowContents"]
    funnel_id = group["funnels"][0]["identifier"]
    python_procs = [
        p for p in group["processors"] if p["bundle"]["artifact"] == "python-extensions"
    ]
    assert len(python_procs) == 4

    for p in python_procs:
        cls = tool.processor_class(p["type"])
        assert p["bundle"]["version"] == cls.ProcessorDetails.version

        descriptors = {d.name: d for d in cls().getPropertyDescriptors()}
        for prop_name, value in p["properties"].items():
            assert prop_name in descriptors
            allowed = descriptors[prop_name].allowable_values
            if allowed:
                assert value in allowed or value.startswith("${")

        outgoing = set()
        failure_destinations = set()
        for c in group["connections"]:
            if c["source"]["id"] == p["identifier"]:
                outgoing.update(c["selectedRelationships"])
                if "failure" in c["selectedRelationships"]:
                    failure_destinations.add(c["destination"]["id"])
        autoterm = set(p["autoTerminatedRelationships"])
        assert (outgoing | autoterm) == {"success", "failure", "original"}
        assert failure_destinations == {funnel_id}


def test_canvas_types_are_in_processor_list():
    """Every Python processor used by the demo must ship in the runtime image."""
    qb_spec = importlib.util.spec_from_file_location(
        "quanifi_build", ROOT / "docker/nifi/quanifi_build.py"
    )
    qb = importlib.util.module_from_spec(qb_spec)
    qb_spec.loader.exec_module(qb)

    canvas_types = set(tool.python_processor_types(tool.simple_snapshot()))
    processor_list = set(
        qb.read_processor_list(
            str(ROOT / "docker/processors.txt"), ROOT / "nifi_extensions"
        )
    )
    assert canvas_types <= processor_list


def test_simple_flow_executes_and_reports(tmp_path):
    from conftest import MockContext, MockFlowFile, result_to_flowfile_merged

    flowfile = MockFlowFile()
    for proc in tool.simple_snapshot()["flowContents"]["processors"][1:]:
        properties = dict(proc["properties"])
        if proc["type"] == "QuanifiReport":
            properties["Reports Directory"] = str(tmp_path)
        result = tool.processor_class(proc["type"])().transform(
            MockContext(**properties), flowfile
        )
        assert result.relationship == "success", result.attributes
        flowfile = result_to_flowfile_merged(result, flowfile)
        if proc["type"] == "QiskitGroverOperator":
            assert result.attributes["circuit.marked_state"] == "10"
            assert result.attributes["circuit.format"] == "qasm2"
        if proc["type"] == "QiskitAerSimulator":
            assert result.attributes["sim.top_result"] == "10"
            assert float(result.attributes["sim.top_probability"]) == 1.0
    report = (tmp_path / "qiskit-grover.html").read_text()
    assert report.count('class="run-card"') == 1
    assert "<td>sim.top_result</td><td>10</td>" in report
