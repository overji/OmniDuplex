from concurrent.futures import ThreadPoolExecutor
import pytest
from train.src.models.loading import ensure_alias


def test_ranks_can_create_the_same_model_alias_concurrently(tmp_path):
    target=tmp_path/'weights';target.mkdir();alias=tmp_path/'cache/model'
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(p==alias for p in pool.map(lambda _:ensure_alias(alias,target),range(32)))
    assert alias.is_symlink() and alias.resolve()==target


def test_existing_alias_to_other_weights_is_never_overwritten(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir();alias=tmp_path/'model';alias.symlink_to(a)
    with pytest.raises(ValueError,match='different checkpoint'):ensure_alias(alias,b)
    assert alias.resolve()==a
