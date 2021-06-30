import pytest
import shutil

import scripts.study_case.ID_5.matchzoo as mz
from scripts.study_case.ID_5.matchzoo.engine.base_preprocessor import BasePreprocessor


@pytest.fixture
def base_preprocessor():
    BasePreprocessor.__abstractmethods__ = set()
    base_processor = BasePreprocessor()
    return base_processor


def test_save_load(base_preprocessor):
    dirpath = '.tmpdir'
    base_preprocessor.save(dirpath)
    assert mz.load_preprocessor(dirpath)
    shutil.rmtree(dirpath)
