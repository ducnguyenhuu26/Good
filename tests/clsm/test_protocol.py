import numpy as np
import json
import pytest

from experiments.rwt.clsm.build_datasets import PUBLIC_KEYS, _sha256
from experiments.rwt.clsm.collectors import NominalControllerMixture
from experiments.rwt.clsm.public_data import load_public_split
from experiments.rwt.clsm.world_schedule import allowed_codes, make_schedule


def test_comp_split_protection():
    assert set(allowed_codes("comp","train")) == {0,1,2,4}
    assert set(allowed_codes("comp","validation")) == {0,1,2,4}
    assert set(allowed_codes("comp","test")) == set(range(8))
    assert {law.code for law in make_schedule("comp","train",1000,41)} <= {0,1,2,4}


def test_no_oracle_public_keys():
    assert not set(PUBLIC_KEYS) & {"code","theta","age","switch","reset","motor_1","motor_2","damping"}


def test_schedule_reproducible_and_reset_age_semantics():
    left = make_schedule("evol","train",100,41)
    right = make_schedule("evol","train",100,41)
    assert left == right
    assert all(0 <= law.code <= 7 for law in left)
    assert all(.5 <= law.motor_1 <= 1 for law in left)


def test_generator_mem_independent_vs_correlated_theta():
    pairs = {condition: [] for condition in ("mem_independent", "mem_correlated")}
    for seed in range(100, 150):
        independent = make_schedule("mem", "test", 500, seed, "mem_independent")
        correlated = make_schedule("mem", "test", 500, seed, "mem_correlated")
        # Separate RNG streams make the structural and reset event sequences
        # identical; only the entry-strength law differs between conditions.
        assert [(law.code, law.reset) for law in independent] == [
            (law.code, law.reset) for law in correlated]
        for condition, schedule in (("mem_independent", independent),
                                    ("mem_correlated", correlated)):
            previous_entry = {}
            for t, law in enumerate(schedule):
                if law.code == 0:
                    assert not law.reset
                if t and law.code != schedule[t - 1].code and law.code:
                    active = next(j for j in range(3) if (law.code >> j) & 1)
                    value = (law.motor_1, law.motor_2, law.damping)[active]
                    if law.code in previous_entry:
                        pairs[condition].append((active, previous_entry[law.code], value))
                    previous_entry[law.code] = value
    results = {}
    for condition, values in pairs.items():
        array = np.asarray(values)
        assert len(array) > 400
        active = array[:, 0].astype(int)
        center = np.where(active == 2, 1.65, .675)
        width = np.where(active == 2, .7, .35)
        old = (array[:, 1] - center) / width
        new = (array[:, 2] - center) / width
        results[condition] = (float(np.corrcoef(old, new)[0, 1]),
                              float(np.mean((new - old) ** 2)),
                              float(np.mean(new ** 2)))
    independent_corr, independent_cache_mse, independent_prior_mse = results["mem_independent"]
    correlated_corr, correlated_cache_mse, correlated_prior_mse = results["mem_correlated"]
    assert abs(independent_corr) < .15 and independent_cache_mse > independent_prior_mse
    assert correlated_corr > .75 and correlated_cache_mse < correlated_prior_mse / 2


def test_training_loader_rejects_oracle_column_even_with_valid_hash(tmp_path):
    arrays = {key: np.zeros(1) for key in PUBLIC_KEYS}
    arrays.update(obs=np.zeros((1, 3)), action=np.zeros((1, 2)), next_obs=np.zeros((1, 3)),
                  world_id=np.zeros(1, dtype=np.int64), world_step=np.zeros(1, dtype=np.int64),
                  code=np.zeros(1, dtype=np.int64))
    path = tmp_path / "train.npz"
    np.savez_compressed(path, **arrays)
    entry = {"public_file": str(path), "public_sha256": _sha256(path),
             "transitions": 1, "world_ids": [0]}
    (tmp_path / "manifest.json").write_text(json.dumps({"splits": {"train": entry}}))
    with pytest.raises(ValueError, match="nonpublic"):
        load_public_split(tmp_path, "train")


def test_full_collector_seeded_and_public_observation_only():
    class PublicPolicy:
        def predict(self, observation, deterministic=True):
            assert observation.shape == (3,) and deterministic
            return np.array([observation[0], -observation[0]]), None
    args = (-np.ones(2), np.ones(2), PublicPolicy(), 41)
    first, second = NominalControllerMixture(*args), NominalControllerMixture(*args)
    observations = [np.array([.2, .1, -.1])] * 30
    left = np.stack([first.act(obs) for obs in observations])
    right = np.stack([second.act(obs) for obs in observations])
    assert np.array_equal(left, right)
    assert np.max(np.abs(left)) <= 1
    assert np.std(left) > 0
