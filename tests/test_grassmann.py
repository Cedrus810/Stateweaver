import numpy as np
import pytest

from sroqm.grassmann import GrassmannLogError, grassmann_exp, grassmann_log, lowdin_pair


def proj(Y):
    return Y @ Y.T


@pytest.fixture
def point_and_tangent():
    rng = np.random.default_rng(0)
    Y0, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    G = rng.normal(size=(12, 4))
    G -= Y0 @ (Y0.T @ G)
    return Y0, 0.1 * G


def test_lowdin_pair_squares_and_inverts():
    rng = np.random.default_rng(2)
    B = rng.normal(size=(6, 6))
    S = B @ B.T + 6 * np.eye(6)
    half, invhalf = lowdin_pair(S)
    assert np.allclose(half @ half, S)
    assert np.allclose(half @ invhalf, np.eye(6))


def test_lowdin_pair_rejects_singular_overlap():
    with pytest.raises(ValueError, match="positive definite"):
        lowdin_pair(np.ones((3, 3)))


def test_exp_of_zero_returns_the_base_point(point_and_tangent):
    Y0, _ = point_and_tangent
    assert np.allclose(proj(grassmann_exp(Y0, np.zeros_like(Y0))), proj(Y0))


def test_log_inverts_exp(point_and_tangent):
    Y0, G = point_and_tangent
    Y1 = grassmann_exp(Y0, G)
    back = grassmann_exp(Y0, grassmann_log(Y0, Y1))
    assert np.allclose(proj(back), proj(Y1), atol=1e-12)


def test_log_ignores_rotations_of_the_target(point_and_tangent):
    Y0, G = point_and_tangent
    Y1 = grassmann_exp(Y0, G)
    W, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(4, 4)))
    assert np.allclose(grassmann_log(Y0, Y1 @ W), grassmann_log(Y0, Y1), atol=1e-12)


def test_linear_extrapolation_along_a_geodesic_is_exact(point_and_tangent):
    Y0, G = point_and_tangent
    Y = [grassmann_exp(Y0, t * G) for t in range(3)]
    pred = grassmann_exp(Y[1], -grassmann_log(Y[1], Y[0]))
    assert np.allclose(proj(pred), proj(Y[2]), atol=1e-12)


def test_log_raises_on_an_orthogonal_direction():
    Y0 = np.eye(6)[:, [0, 1]]
    Y1 = np.eye(6)[:, [0, 2]]
    with pytest.raises(GrassmannLogError):
        grassmann_log(Y0, Y1)


def test_empty_subspace_is_a_no_op():
    Y0 = np.zeros((5, 0))
    assert grassmann_log(Y0, Y0).shape == (5, 0)
    assert grassmann_exp(Y0, Y0).shape == (5, 0)
