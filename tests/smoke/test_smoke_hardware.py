"""Smoke test suite exercising all 12 Hardware processors (gateways, simulators, submitters & pollers)."""

import datetime
import json
import tempfile
import pytest

from tests.smoke.smoke_helpers import (
    MockContext,
    MockFlowFile,
    mark_exercised,
    qasm_ff,
)

pytestmark = pytest.mark.smoke_comprehensive


class TestHardwareSubsystem:
    """Smoke tests covering all 12 hardware gateway, simulator, submitter and poller processors."""

    def test_hardware_simulators_and_gateways(self):
        from BraketDevice import BraketDevice
        from BraketSimulator import BraketSimulator
        from QSharpSimulator import QSharpSimulator

        ff = qasm_ff()

        # BraketDevice
        mark_exercised("BraketDevice")
        res_bd = BraketDevice().transform(MockContext(**{"Device": "local", "Shots": "64"}), ff)
        assert res_bd.relationship == "success"

        # BraketSimulator
        mark_exercised("BraketSimulator")
        res_bs = BraketSimulator().transform(MockContext(**{"Shots": "64"}), ff)
        assert res_bs.relationship == "success"

        # QSharpSimulator
        mark_exercised("QSharpSimulator")
        res_qs = QSharpSimulator().transform(MockContext(**{"Shots": "64"}), ff)
        assert res_qs.relationship == "success"

    def test_quantum_batch_result_expander(self):
        from QuantumBatchResultExpander import QuantumBatchResultExpander

        mark_exercised("QuantumBatchResultExpander")
        exp_data = {
            "job_id": "job-1", "device": "test", "bit_order": "q0_right",
            "entries": [{"label": "e1", "kind": "control", "num_qubits": 2, "counts": {"00": 100}}],
        }
        res = QuantumBatchResultExpander().transform(MockContext(), MockFlowFile(content=json.dumps(exp_data)))
        assert res.relationship == "success"

    def test_ibm_batch_lane(self):
        from QuantumIBMBatchSubmitter import QuantumIBMBatchSubmitter
        from QuantumIBMBatchPoller import QuantumIBMBatchPoller

        mark_exercised("QuantumIBMBatchSubmitter")

        class OfflineIBMSubmitter(QuantumIBMBatchSubmitter):
            def prepare(self, *args, **kwargs):
                return {
                    "backend": None, "transpiled": [], "layout": [0, 1], "two_qubit_gates": 2,
                    "pending": 0, "estimated_usage": 0.1, "widths": [2], "padded_width": 2, "capacity": 127,
                }
            def run_job(self, plan, shots):
                return "ibm-job-123"
            def serialize_isa(self, plan, expected):
                return ["OPENQASM 2.0;"] * expected
            def service_for(self, token, instance):
                now = datetime.datetime.now(datetime.timezone.utc)
                class S:
                    @staticmethod
                    def usage():
                        return {"usage": {"seconds": 10}, "usage_limit": {"seconds": 600}, "usage_period": {"start_time": now, "end_time": now}}
                return S()

        with tempfile.TemporaryDirectory() as td:
            ctx = MockContext(**{
                "Batch Label": "b1", "Expected Circuits": "1", "Device": "ibm_test", "API Token": "tok",
                "State Directory": td, "Manifest Directory": td, "Submit Mode": "armed",
                "Fixed Layout": "0,1", "Excluded Physical Qubits": "", "Shots": "100", "Attribute Prefix": "arithmetic.",
            })
            res_sub = OfflineIBMSubmitter().transform(ctx, qasm_ff())
            assert res_sub.relationship == "submitted"

        mark_exercised("QuantumIBMBatchPoller")

        class OfflineIBMPoller(QuantumIBMBatchPoller):
            def service(self, token, instance):
                class Pub:
                    class Data:
                        meas = type("Counts", (), {"get_counts": lambda self: {"00": 50, "11": 50}})()
                        def keys(self): return ["meas"]
                    data = Data()
                class Job:
                    def status(self): return "DONE"
                    def metrics(self): return {"usage": {"quantum_seconds": 1, "seconds": 1, "status": "complete"}}
                    def result(self): return [Pub()]
                return type("Service", (), {"job": lambda self, jid: Job()})()

        poller_manifest = {"job_id": "ibm-job-123", "device": "ibm_test", "entries": [{"label": "e1", "kind": "control"}]}
        res_pol = OfflineIBMPoller().transform(
            MockContext(**{"API Token": "tok", "Instance": "test", "Poll Timeout Seconds": "1"}),
            MockFlowFile(content=json.dumps(poller_manifest)),
        )
        assert res_pol.relationship == "success"

    def test_iqm_batch_lane(self, monkeypatch):
        from QuantumIQMBatchSubmitter import QuantumIQMBatchSubmitter
        from QuantumIQMBatchPoller import QuantumIQMBatchPoller
        from IQMJobPoller import IQMJobPoller

        mark_exercised("QuantumIQMBatchSubmitter")

        class OfflineIQMSubmitter(QuantumIQMBatchSubmitter):
            def prepare(self, *args, **kwargs):
                return {
                    "backend": None, "transpiled": [], "layout": [0, 1], "two_qubit_gates": 2,
                    "estimated_usage": 0.0, "widths": [2], "padded_width": 2, "capacity": 20,
                }
            def run_job(self, plan, shots):
                return "iqm-job-1"
            def read_provider_credits(self, server, token):
                return 100.0

        with tempfile.TemporaryDirectory() as td:
            ctx = MockContext(**{
                "Batch Label": "b1", "Expected Circuits": "1", "API Token": "tok",
                "State Directory": td, "Manifest Directory": td, "Device": "garnet", "Submit Mode": "armed", "Shots": "64",
            })
            res_sub = OfflineIQMSubmitter().transform(ctx, qasm_ff())
            assert res_sub.relationship == "submitted"

        mark_exercised("QuantumIQMBatchPoller")
        manifest = {"job_id": "iqm-job-1", "device": "garnet", "entries": [{"label": "e1", "kind": "control"}]}

        class MockResp:
            def __init__(self, data):
                self._data = data
                self.status_code = 200
                self.ok = True
            def json(self): return self._data
            def raise_for_status(self): pass

        def mock_get(self, url, *args, **kwargs):
            if url.endswith("/measurement_counts"):
                return MockResp([{"counts": {"00": 100}}])
            return MockResp({"status": "completed", "timeline": [{"status": "completed", "timestamp": "2026-01-01T00:00:00Z"}]})

        monkeypatch.setattr("requests.Session.get", mock_get)

        res_p = QuantumIQMBatchPoller().transform(
            MockContext(**{"API Token": "tok", "Job ID": "iqm-job-1", "Poll Timeout Seconds": "1"}),
            MockFlowFile(content=json.dumps(manifest)),
        )
        assert res_p.relationship == "success"

        mark_exercised("IQMJobPoller")
        poller = IQMJobPoller()
        iqm_calls = [
            MockResp({"status": "completed", "queue_position": None}),
            MockResp([{"measurement_keys": ["m"], "counts": {"00": 100}}]),
        ]
        monkeypatch.setattr(IQMJobPoller, "_get", lambda self, session, url, timeout=30: iqm_calls.pop(0))
        ff_iqm = MockFlowFile(content=b'{"job_id": "iqm-job-1"}', attributes={"hw.job_id": "iqm-job-1"})
        res_ijp = poller.transform(
            MockContext(**{"API Token": "tok", "Job ID": "iqm-job-1", "Poll Timeout Seconds": "1"}),
            ff_iqm,
        )
        assert res_ijp.relationship == "success"

    def test_generic_batch_poller(self, monkeypatch):
        from QuantumBatchPoller import QuantumBatchPoller

        mark_exercised("QuantumBatchPoller")
        poller = QuantumBatchPoller()

        class _Resp:
            def __init__(self, code, payload):
                self.status_code = code
                self._payload = payload
            def json(self): return self._payload

        monkeypatch.setattr(
            poller, "_get",
            lambda s, url, timeout=30: _Resp(200, [{"counts": {"00": 100}}])
            if url.endswith("measurement_counts") else _Resp(200, {"status": "completed"}),
        )
        manifest = {"job_id": "j1", "entries": [{"label": "e1", "kind": "control"}]}
        res = poller.transform(
            MockContext(**{"API Token": "tok", "Job ID": "j1", "Poll Timeout Seconds": "1", "Poll Interval Seconds": "1"}),
            MockFlowFile(content=json.dumps(manifest)),
        )
        assert res.relationship == "success"

    def test_quantum_inspire_batch_lane(self):
        from QuantumInspireBatchSubmitter import QuantumInspireBatchSubmitter
        from QuantumInspireBatchPoller import QuantumInspireBatchPoller
        from qiskit.providers.jobstatus import JobStatus
        import qiskit_quantuminspire.qi_jobs
        import qiskit_quantuminspire.qi_provider

        mark_exercised("QuantumInspireBatchSubmitter")

        class StubQISubmitter(QuantumInspireBatchSubmitter):
            def prepare(self, *args, **kwargs):
                return {
                    "backend": None, "transpiled": [0], "layout": [0, 1], "two_qubit_gates": 2,
                    "widths": [2], "padded_width": 2, "capacity": 5, "native_basis_gates": "cz,h",
                    "chunks": [(0, 1)], "chunk_size": 5, "provider_batchjobs_per_queue_limit": 5,
                    "isa_profile": [{"two_qubit_gates": 1, "depth": 3, "gate_count": 5}],
                    "provider_backend_type": {"name": "Tuna-17"}, "provider_max_batch_jobs": 5,
                    "estimated_usage": 0.0,
                }
            def run_chunk(self, plan, shots, circuits):
                class Job:
                    batch_job_id = "qi-job-1"
                    circuits_run_data = [type("C", (), {"job_id": "cj-1"})()]
                return Job()
            def persist_handle(self, manifest_dir, job):
                p = f"{manifest_dir}/handle.json"
                with open(p, "w") as f:
                    f.write("{}")
                return (p, "")

        with tempfile.TemporaryDirectory() as td:
            ctx = MockContext(**{
                "Batch Label": "b1", "Expected Circuits": "1", "Device": "Tuna-17",
                "State Directory": td, "Manifest Directory": td, "Submit Mode": "armed",
                "Shots": "100",
            })
            res_sub = StubQISubmitter().transform(ctx, qasm_ff())
            assert res_sub.relationship == "submitted"

        mark_exercised("QuantumInspireBatchPoller")

        class FakeQIJob:
            batch_job_id = "qi-1"
            def status(self): return JobStatus.DONE
            def result(self, *args, **kwargs):
                class Res:
                    def get_counts(self, index=None): return {"00": 100}
                    results = [type("Exp", (), {"shots": 100})()]
                return Res()

        orig_deserialize = getattr(qiskit_quantuminspire.qi_jobs.QIJob, "deserialize", None)
        orig_provider = getattr(qiskit_quantuminspire.qi_provider, "QIProvider", None)
        try:
            qiskit_quantuminspire.qi_jobs.QIJob.deserialize = staticmethod(lambda *args, **kwargs: FakeQIJob())
            class FakeQIProvider:
                pass
            qiskit_quantuminspire.qi_provider.QIProvider = FakeQIProvider
            manifest = {"job_id": "qi-1", "entries": [{"label": "e1", "kind": "control"}], "provider_handle_path": "/tmp/dummy"}
            res_p = QuantumInspireBatchPoller().transform(MockContext(), MockFlowFile(content=json.dumps(manifest)))
            assert res_p.relationship == "success", res_p.attributes
        finally:
            if orig_deserialize:
                qiskit_quantuminspire.qi_jobs.QIJob.deserialize = orig_deserialize
            if orig_provider:
                qiskit_quantuminspire.qi_provider.QIProvider = orig_provider
