"""Unit tests for the standalone post-hoc GLL zero-mean action tool."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from znmf_actuation import EmbeddedPolicy, face_gll_weights, zero_mean_action


def _record(agents, action):
    return SimpleNamespace(agents=pd.DataFrame(agents), act=np.asarray(action, dtype=float))


def test_face_aware_weights_and_constant_field_are_zero_mean():
    # Face 4 is x-normal, so its tangential indices are iy and iz.  Here ix
    # intentionally differs from iy to catch an accidental ix/iz treatment.
    agents = {
        "iface": [1, 1, 4, 4],
        "ix": [1, 2, 3, 3],
        "iy": [1, 1, 1, 2],
        "iz": [1, 1, 1, 1],
        "ipol": [1, 1, 1, 1],
    }
    rec = _record(agents, [[3.0, 3.0, 3.0, 3.0], [-2.0, -2.0, -2.0, -2.0]])
    weights, _ = face_gll_weights(rec.agents, gll_nodes=3)
    assert weights[2] == weights[0]
    assert weights[3] == weights[1]

    result = zero_mean_action(rec, support="all", gll_nodes=3)
    assert np.allclose(result.removed_mean, [3.0, -2.0])
    assert np.allclose(result.act_zero_mean, 0.0)
    assert np.allclose(result.residual_mean, 0.0)


def _policy():
    return EmbeddedPolicy(
        path=Path("actor.pol"), dimensions=(2, 2, 1), activations=(1, 2),
        squash=True, action_lower=-1.0, action_upper=1.0,
        observation_scale=0.064, action_scale_in_file=0.064,
        observation_permutation=np.array([2, 1]),
        weights=[np.eye(2), np.array([[2.0, -1.0]])],
        biases=[np.zeros(2), np.zeros(1)],
    )


def test_policy_surface_evaluation_with_permutation_and_scaling():
    # Reversing inputs and applying ReLU gives [0, 2] and [4, 0].
    # The last layer therefore evaluates tanh(-2) and tanh(8).
    observations = np.array([[2.0, -3.0], [-4.0, 4.0]])
    expected = np.tanh([-2.0, 8.0]) * 0.3
    for precision in (4, 8):
        action = _policy().evaluate_normalised(observations, 0.3, precision)
        np.testing.assert_allclose(action, expected, atol=1e-7, rtol=0.0)


def test_policy_surface_rejects_inconsistent_layer_counts():
    policy = _policy()
    policy.biases.pop()
    with np.testing.assert_raises_regex(ValueError, "inconsistent policy layer counts"):
        policy.evaluate_normalised(np.zeros((1, 2)), 1.0, 8)


def test_active_support_is_zero_mean_and_leaves_inactive_values_raw():
    agents = {
        "iface": [1, 1, 1, 1],
        "ix": [1, 2, 1, 2],
        "iy": [1, 1, 2, 2],
        "iz": [1, 1, 1, 1],
        "ipol": [0, 1, 1, 0],
    }
    raw = np.array([[10.0, 1.0, 5.0, 20.0], [7.0, -3.0, 2.0, 8.0]])
    rec = _record(agents, raw)
    result = zero_mean_action(rec, support="active", gll_nodes=2)

    assert np.array_equal(result.act_raw, raw)
    assert np.array_equal(result.act_zero_mean[:, ~result.support], raw[:, ~result.support])
    assert np.allclose(result.residual_mean, 0.0)
