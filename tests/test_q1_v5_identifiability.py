import numpy as np
import pytest

from q1_v5.identifiability import efficient_information


def test_information_is_residualized_against_nuisance():
    nuisance = np.linspace(-1, 1, 40)
    independent = np.sin(np.arange(40))
    jac = np.column_stack((nuisance, 2 * nuisance + independent))
    result = efficient_information(jac, 1, sigma=.1)
    assert result["nuisance_rank"] == 1
    assert result["efficient_information"] == pytest.approx(
        np.linalg.norm(independent - nuisance * (nuisance @ independent) / (nuisance @ nuisance)) ** 2 / .01
    )


def test_exact_collinearity_has_zero_information_without_inverse():
    x = np.linspace(-2, 3, 20)
    result = efficient_information(np.column_stack((x, 3 * x)), 1, sigma=.2)
    assert result["efficient_information"] == pytest.approx(0, abs=1e-24)
    assert np.isinf(result["crlb_variance"])


def test_rejects_invalid_jacobian_and_noise():
    with pytest.raises(ValueError):
        efficient_information(np.array([[np.nan]]), 0, sigma=1)
    with pytest.raises(ValueError):
        efficient_information(np.ones((2, 1)), 0, sigma=0)
