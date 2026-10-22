"""Small independent synthetic oracle checks for CLSM inference and evolution."""
import numpy as np
import torch

from rwt.clsm.evolution import LawEvolution


class IndependentToyWorld:
    """Evaluator-only force law implemented without importing CLSM modules."""

    def __init__(self, noise_std: float, effect_scale: float = 1.0):
        self.noise_std = noise_std
        self.effect_scale = effect_scale

    @staticmethod
    def _softplus(value):
        return np.logaddexp(0.0, value)

    def mean(self, obs, action, code, theta):
        bits = np.array([(code >> j) & 1 for j in range(3)], dtype=np.float64)
        active = bits * np.asarray(theta)
        g0 = np.exp(-bits[0] * (.12 + self._softplus(active[0])))
        g1 = np.exp(-bits[1] * (.12 + self._softplus(active[1])))
        damping = np.exp(bits[2] * (.12 + self._softplus(active[2])))
        f0, f1 = self.effect_scale * g0 * action[0], self.effect_scale * g1 * action[1]
        # The product term is an explicit, evaluator-only compositional effect.
        delta = np.array([f0 + f1 + .4*f0*f1,
                          -.2*damping*obs[1] + .15*f0 - .1*f1])
        return np.asarray(obs) + delta

    def step(self, obs, action, code, theta, rng):
        return self.mean(obs, action, code, theta) + rng.normal(0, self.noise_std, size=2)


def _durations(rng, kappa, shape, count):
    durations = []
    for _ in range(count):
        age = 0
        while age < 200:
            h = 1 - np.exp(-(((age+1)/kappa)**shape - (age/kappa)**shape))
            age += 1
            if rng.random() < h:
                break
        durations.append(age)
    return np.asarray(durations, dtype=np.int64)


def _fit_duration(durations):
    ages, events = [], []
    for duration in durations:
        ages.extend(range(int(duration)))
        events.extend([0.] * (int(duration)-1) + [1.])
    ages = torch.tensor(ages, dtype=torch.long)
    events = torch.tensor(events, dtype=torch.float64)
    model = LawEvolution()
    optimizer = torch.optim.Adam([model.log_kappa0, model.nu_raw], lr=.03)
    code = torch.zeros_like(ages)
    for _ in range(250):
        hazard = model.hazard(code, ages).clamp(1e-10, 1-1e-10)
        loss = -(events * hazard.log() + (1-events) * torch.log1p(-hazard)).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        return float(model.hazard(torch.tensor([0]), torch.tensor([0]))[0])


def test_independent_oracle_composition_and_posterior_separation():
    rng = np.random.default_rng(153)
    oracle = IndependentToyWorld(noise_std=.02)
    obs = np.array([.1, .4])
    action_pairs = [np.array([1., 0.]), np.array([0., 1.]), np.array([1., 1.])] * 8
    theta = np.array([-2., -2., 0.])
    truth = np.stack([oracle.step(obs, a, 3, theta, rng) for a in action_pairs])
    log_likelihood = []
    for code in range(8):
        means = np.stack([oracle.mean(obs, a, code, theta) for a in action_pairs])
        residual = truth - means
        log_likelihood.append(-.5*np.square(residual / oracle.noise_std).sum())
    posterior = np.exp(np.asarray(log_likelihood) - np.logaddexp.reduce(log_likelihood))
    assert int(np.argmax(posterior)) == 3
    assert posterior[3] > .99

    weak = IndependentToyWorld(noise_std=.2, effect_scale=.01)
    weak_observations = np.stack([weak.step(obs, a, 3, theta, rng) for a in action_pairs[:3]])
    weak_scores = []
    for code in range(8):
        means = np.stack([weak.mean(obs, a, code, theta) for a in action_pairs[:3]])
        residual = weak_observations - means
        weak_scores.append(-.5*np.square(residual / weak.noise_std).sum())
    weak_posterior = np.exp(np.asarray(weak_scores) - np.logaddexp.reduce(weak_scores))
    assert weak_posterior.max() < .5

    mu00 = oracle.mean(obs, np.array([1., 1.]), 0, theta)
    mu10 = oracle.mean(obs, np.array([1., 1.]), 1, theta)
    mu01 = oracle.mean(obs, np.array([1., 1.]), 2, theta)
    mu11 = oracle.mean(obs, np.array([1., 1.]), 3, theta)
    interaction = mu11 - mu10 - mu01 + mu00
    assert np.linalg.norm(interaction) > .01


def test_synthetic_fast_and_slow_dwell_fit_distinct_hazards():
    rng = np.random.default_rng(9871)
    fast = _durations(rng, kappa=7., shape=.8, count=500)
    slow = _durations(rng, kappa=35., shape=1.2, count=500)
    fast_hazard = _fit_duration(fast)
    slow_hazard = _fit_duration(slow)
    assert fast_hazard > 4 * slow_hazard


def test_stationary_censored_world_fits_low_switch_hazard():
    # A stationary sequence contributes survival evidence at every observed
    # age. Fit the registered discrete-Weibull likelihood without switch events.
    evolution = LawEvolution()
    ages = torch.arange(100, dtype=torch.long)
    codes = torch.zeros_like(ages)
    initial = float((-torch.log1p(-evolution.hazard(codes, ages))).mean().detach())
    optimizer = torch.optim.Adam([evolution.log_kappa0, evolution.nu_raw], lr=.03)
    for _ in range(250):
        hazard = evolution.hazard(codes, ages).clamp(1e-10, 1 - 1e-10)
        survival_nll = -torch.log1p(-hazard).mean()
        optimizer.zero_grad()
        survival_nll.backward()
        optimizer.step()
    fitted = float((-torch.log1p(-evolution.hazard(codes, ages))).mean().detach())
    assert fitted < initial
    assert fitted < .01
