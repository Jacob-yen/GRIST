import pytest

import scripts.study_case.ID_5.matchzoo as mz
from scripts.study_case.ID_5.matchzoo import preprocessors
from scripts.study_case.ID_5.matchzoo.dataloader import callbacks
from scripts.study_case.ID_5.matchzoo.dataloader import Dataset, DataLoader
from scripts.study_case.ID_5.matchzoo.datasets import embeddings
from scripts.study_case.ID_5.matchzoo.embedding import load_from_file


@pytest.fixture(scope='module')
def train_raw():
    return mz.datasets.toy.load_data('test', task='ranking')[:5]


def test_basic_padding(train_raw):
    preprocessor = preprocessors.BasicPreprocessor()
    data_preprocessed = preprocessor.fit_transform(train_raw, verbose=0)
    dataset = Dataset(data_preprocessed, mode='point')

    pre_fixed_padding = callbacks.BasicPadding(
        fixed_length_left=5, fixed_length_right=5, pad_word_mode='pre', with_ngram=False)
    dataloader = DataLoader(
        dataset, batch_size=5, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 5)
        assert batch[0]['text_right'].shape == (5, 5)

    post_padding = callbacks.BasicPadding(pad_word_mode='post', with_ngram=False)
    dataloader = DataLoader(dataset, batch_size=5, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].numpy())
        max_right_len = max(batch[0]['length_right'].numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len)
        assert batch[0]['text_right'].shape == (5, max_right_len)


def test_drmm_padding(train_raw):
    preprocessor = preprocessors.BasicPreprocessor()
    data_preprocessed = preprocessor.fit_transform(train_raw, verbose=0)

    embedding_matrix = load_from_file(embeddings.EMBED_10_GLOVE, mode='glove')
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = embedding_matrix.build_matrix(term_index)
    histgram_callback = callbacks.Histogram(
        embedding_matrix=embedding_matrix, bin_size=30, hist_mode='LCH')
    dataset = Dataset(
        data_preprocessed, mode='point', callbacks=[histgram_callback])

    pre_fixed_padding = callbacks.DRMMPadding(
        fixed_length_left=5, fixed_length_right=5, pad_mode='pre')
    dataloader = DataLoader(
        dataset, batch_size=5, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 5)
        assert batch[0]['text_right'].shape == (5, 5)
        assert batch[0]['match_histogram'].shape == (5, 5, 30)

    post_padding = callbacks.DRMMPadding(pad_mode='post')
    dataloader = DataLoader(dataset, batch_size=5, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].numpy())
        max_right_len = max(batch[0]['length_right'].numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len)
        assert batch[0]['text_right'].shape == (5, max_right_len)
        assert batch[0]['match_histogram'].shape == (5, max_left_len, 30)


def test_bert_padding(train_raw):
    preprocessor = preprocessors.BertPreprocessor()
    data_preprocessed = preprocessor.transform(train_raw, verbose=0)
    dataset = Dataset(data_preprocessed, mode='point')

    pre_fixed_padding = callbacks.BertPadding(
        fixed_length_left=5, fixed_length_right=5, pad_mode='pre')
    dataloader = DataLoader(
        dataset, batch_size=5, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 6)
        assert batch[0]['text_right'].shape == (5, 7)

    post_padding = callbacks.BertPadding(pad_mode='post')
    dataloader = DataLoader(dataset, batch_size=5, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].numpy())
        max_right_len = max(batch[0]['length_right'].numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len + 1)
        assert batch[0]['text_right'].shape == (5, max_right_len + 2)
