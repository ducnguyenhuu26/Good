import numpy as np

from experiments.rwt.clsm.build_datasets import PUBLIC_KEYS
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
