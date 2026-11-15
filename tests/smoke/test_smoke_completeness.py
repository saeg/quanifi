"""Smoke test module validating completeness of processor coverage across all subsystems."""

import pytest

from tests.smoke.smoke_helpers import (
    ALL_PROCESSORS,
    CIRQ_PROCESSORS,
    COMMON_PROCESSORS,
    EXERCISED_PROCESSORS,
    HARDWARE_PROCESSORS,
    PENNYLANE_PROCESSORS,
    PYQUIL_PROCESSORS,
    QISKIT_PROCESSORS,
    QRISP_PROCESSORS,
)
import add_all_components_group as tool

pytestmark = pytest.mark.smoke_comprehensive


def test_every_single_processor_is_accounted_for():
    """Verify that every single processor discovered in the repository is partitioned into a subsystem suite."""
    discovered = {p["name"] for p in tool.categorise(tool.discover())}
    assert len(discovered) == 122, f"Expected 122 processors on saeg/main, found {len(discovered)}"

    uncovered = discovered - ALL_PROCESSORS
    assert not uncovered, f"Processors missing from smoke suites: {uncovered}"

    extra = ALL_PROCESSORS - discovered
    assert not extra, f"Processors in smoke suites not discovered in repo: {extra}"

    # Verify disjoint partitions
    subsystems = [
        ("qiskit", QISKIT_PROCESSORS),
        ("qrisp", QRISP_PROCESSORS),
        ("cirq", CIRQ_PROCESSORS),
        ("pennylane", PENNYLANE_PROCESSORS),
        ("pyquil", PYQUIL_PROCESSORS),
        ("common", COMMON_PROCESSORS),
        ("hardware", HARDWARE_PROCESSORS),
    ]
    seen = set()
    for name, group in subsystems:
        overlap = seen & group
        assert not overlap, f"Subsystem {name} overlaps with other subsystems on: {overlap}"
        seen |= group

    assert seen == discovered, "Union of subsystems must equal discovered processors"

