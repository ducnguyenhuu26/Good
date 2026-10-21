from experiments.rwt.clsm.summarize import hierarchical_interval


def test_hierarchical_interval_requires_independent_seeds_and_worlds():
    rows = [{"train_seed": 41, "world_seed": world, "effect": 1.}
            for world in range(20)]
    assert hierarchical_interval(rows) is None
    rows = [{"train_seed": seed, "world_seed": world, "effect": 1.}
            for seed in range(41, 46) for world in range(801, 821)]
    assert hierarchical_interval(rows, draws=100) == (1., 1.)
