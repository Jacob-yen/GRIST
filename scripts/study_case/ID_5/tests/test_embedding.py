import pytest

import scripts.study_case.ID_5.matchzoo as mz


@pytest.fixture
def term_index():
    return {'G': 1, 'C': 2, 'D': 3, 'A': 4, '_PAD': 0}


def test_embedding(term_index):
    embed = mz.embedding.load_from_file(mz.datasets.embeddings.EMBED_RANK)
    matrix = embed.build_matrix(term_index)
    assert matrix.shape == (len(term_index), 50)
    embed = mz.embedding.load_from_file(mz.datasets.embeddings.EMBED_10_GLOVE,
                                        mode='glove')
    matrix = embed.build_matrix(term_index)
    assert matrix.shape == (len(term_index), 10)
