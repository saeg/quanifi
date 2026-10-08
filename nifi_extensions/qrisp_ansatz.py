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

"""Qrisp ansatz-spec helpers shared by QrispAnsatz and QrispVQE.

Kept out of the processor modules because a processor must not import another
processor (see docker/nifi/quanifi_build.py helper_closure).
"""

# Rotation gates applied per qubit, per layer, for each ansatz type.
_ROTATIONS = {
    "efficient_su2": ("ry", "rz"),
    "real_amplitudes": ("ry",),
    "two_local": ("ry",),
}
# Entangling gate per ansatz type (matches the Cirq convention).
_ENTANGLER = {"efficient_su2": "cz", "real_amplitudes": "cz", "two_local": "cx"}


def _entangle_pairs(n, pattern):
    if n < 2:
        return []
    if pattern == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if pattern == "circular":
        return [(i, (i + 1) % n) for i in range(n)]
    return [(i, j) for i in range(n) for j in range(i + 1, n)]  # full


def _num_params_per_layer(atype, n):
    return n * len(_ROTATIONS.get(atype, _ROTATIONS["efficient_su2"]))


def _build_ansatz_function(atype, n, entanglement):
    """Return ``(ansatz_function, num_params_per_layer)`` for Qrisp's VQEProblem.

    The ansatz is a *callable* ``ansatz_function(qv, theta)`` applying one layer
    (Qrisp's VQEProblem repeats it ``depth`` times). This is why the cross-processor
    carrier is a spec rather than a serialised circuit — Qrisp's ansatz is code.
    Quantum gates are imported lazily so the module stays importable without qrisp.
    """
    from qrisp import ry, rz, cx, cz

    rotations = _ROTATIONS.get(atype, _ROTATIONS["efficient_su2"])
    ent = {"cz": cz, "cx": cx}[_ENTANGLER.get(atype, "cz")]
    rot_gate = {"ry": ry, "rz": rz}
    pairs = _entangle_pairs(n, entanglement)

    def ansatz_function(qv, theta):
        k = 0
        for i in range(n):
            for r in rotations:
                rot_gate[r](theta[k], qv[i])
                k += 1
        for a, b in pairs:
            ent(qv[a], qv[b])

    return ansatz_function, n * len(rotations)


def _describe(atype, n, entanglement):
    rot = "+".join(r.upper() for r in _ROTATIONS.get(atype, _ROTATIONS["efficient_su2"]))
    ent = _ENTANGLER.get(atype, "cz").upper()
    return "{}: per layer {} on each of {} qubits, then {} entanglers ({})".format(
        atype, rot, n, ent, entanglement)
