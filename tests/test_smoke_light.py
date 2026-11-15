"""Backward compatibility wrapper redirecting to tests.smoke.test_smoke_light."""

from tests.smoke.test_smoke_light import (
    TestKeyCircuitBuilders,
    TestKeyHamiltoniansAndEncoders,
    TestKeyReporting,
    TestKeySimulators,
    TestKeySolvers,
    TestKeyTestingInfrastructure,
    pytestmark,
)

__all__ = [
    "TestKeyCircuitBuilders",
    "TestKeyHamiltoniansAndEncoders",
    "TestKeyReporting",
    "TestKeySimulators",
    "TestKeySolvers",
    "TestKeyTestingInfrastructure",
    "pytestmark",
]
