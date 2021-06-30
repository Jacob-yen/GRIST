from scripts.study_case.ID_4.torch_geometric import is_debug_enabled, debug, set_debug


def test_debug():
    assert is_debug_enabled() is False
    set_debug(True)
    assert is_debug_enabled() is True
    set_debug(False)
    assert is_debug_enabled() is False

    assert is_debug_enabled() is False
    with set_debug(True):
        assert is_debug_enabled() is True
    assert is_debug_enabled() is False

    assert is_debug_enabled() is False
    set_debug(True)
    assert is_debug_enabled() is True
    with set_debug(False):
        assert is_debug_enabled() is False
    assert is_debug_enabled() is True
    set_debug(False)
    assert is_debug_enabled() is False

    assert is_debug_enabled() is False
    with debug():
        assert is_debug_enabled() is True
    assert is_debug_enabled() is False
