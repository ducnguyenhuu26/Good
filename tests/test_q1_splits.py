from q1.datasets import make_split_manifest, sample_environment


def test_split_ids_are_disjoint_and_parameter_regions_are_disjoint():
    manifest = make_split_manifest(3)
    groups = list(manifest["environment_ids"].values())
    assert len(set(sum(groups, []))) == sum(map(len, groups))
    seed_groups = list(manifest["environment_seeds"].values())
    assert len(set(sum(seed_groups, []))) == sum(map(len, seed_groups))
    assert manifest["regions"]["train"] != manifest["regions"]["test_background_ood"]
    a = sample_environment(manifest, "train", 0)
    b = sample_environment(manifest, "test_background_ood", 0)
    assert a.d != b.d
