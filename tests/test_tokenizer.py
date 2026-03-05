"""Tests for the BRICS-aware BPE tokenizer (SafeTokenizer)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

import safe_oxidizer
from safe_oxidizer import SafeTokenizer


TRAIN_SMILES = [
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "CCCOCC",
    "O=C(CN1CC[NH2+]CC1)N1CCCCC1",
    "CCCOCCC(=O)c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "CC(CS)C(=O)N1CCCC1C(=O)O",
] * 100  # 1000 molecules for training


class TestTraining:
    def test_train_basic(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES), vocab_size=300)
        assert tok.vocab_size == 300

    def test_train_small_vocab(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES), vocab_size=261)
        # 256 base + 0 merges + 5 special = 261 minimum
        assert tok.vocab_size == 261

    def test_train_produces_merges(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES), vocab_size=300)
        # 300 - 256 - 5 = 39 merges
        assert tok.vocab_size == 300


class TestEncodeDecode:
    @pytest.fixture
    def trained_tok(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES), vocab_size=350)
        return tok

    def test_roundtrip_simple(self, trained_tok):
        smi = "CCCOCC"
        ids = trained_tok.encode(smi)
        decoded = trained_tok.decode(ids)
        # decoded should be the SAFE encoding of the input
        safe_encoded = safe_oxidizer.safe_encode(smi)
        assert decoded == safe_encoded

    def test_roundtrip_celecoxib(self, trained_tok):
        smi = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
        ids = trained_tok.encode(smi)
        decoded = trained_tok.decode(ids)
        safe_encoded = safe_oxidizer.safe_encode(smi)
        assert decoded == safe_encoded

    def test_encode_produces_integers(self, trained_tok):
        ids = trained_tok.encode("CCCOCC")
        assert all(isinstance(i, int) for i in ids)
        assert all(0 <= i < trained_tok.vocab_size for i in ids)

    def test_dot_tokens_present(self, trained_tok):
        smi = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
        ids = trained_tok.encode(smi)
        dot_count = sum(1 for i in ids if i == trained_tok.dot_token_id)
        # celecoxib has 4 fragments -> 3 dots
        assert dot_count == 3

    def test_no_dot_within_fragments(self, trained_tok):
        # Encode ethanol (no BRICS bonds) -> no dot tokens
        ids = trained_tok.encode("CCO")
        assert trained_tok.dot_token_id not in ids

    def test_fallback_invalid_smiles(self, trained_tok):
        # Invalid SMILES should still encode (byte-level fallback)
        ids = trained_tok.encode("not_valid")
        assert len(ids) > 0
        decoded = trained_tok.decode(ids)
        assert decoded == "not_valid"


class TestBatch:
    @pytest.fixture
    def trained_tok(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES), vocab_size=350)
        return tok

    def test_batch_returns_numpy(self, trained_tok):
        smiles = ["CCCOCC", "CCO"]
        batch = trained_tok.batch_encode(smiles)
        assert isinstance(batch, np.ndarray)
        assert batch.dtype == np.uint32
        assert batch.ndim == 2
        assert batch.shape[0] == 2

    def test_batch_pads_to_longest(self, trained_tok):
        smiles = ["CCCOCC", "CCO", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"]
        batch = trained_tok.batch_encode(smiles)
        individual = [trained_tok.encode(s) for s in smiles]
        max_len = max(len(ids) for ids in individual)
        assert batch.shape == (3, max_len)
        # Check content matches (non-padded part)
        for i, ids in enumerate(individual):
            np.testing.assert_array_equal(batch[i, :len(ids)], ids)
            # Padding should be pad_token_id
            if len(ids) < max_len:
                np.testing.assert_array_equal(
                    batch[i, len(ids):],
                    np.full(max_len - len(ids), trained_tok.pad_token_id, dtype=np.uint32),
                )

    def test_batch_max_length(self, trained_tok):
        smiles = ["CCCOCC", "CCO"]
        batch = trained_tok.batch_encode(smiles, max_length=10)
        assert batch.shape == (2, 10)
        assert batch.dtype == np.uint32

    def test_batch_max_length_truncates(self, trained_tok):
        smiles = ["Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"]
        ids = trained_tok.encode(smiles[0])
        short_len = 5
        assert len(ids) > short_len  # precondition
        batch = trained_tok.batch_encode(smiles, max_length=short_len)
        assert batch.shape == (1, short_len)
        np.testing.assert_array_equal(batch[0], ids[:short_len])

    def test_batch_padding_value(self, trained_tok):
        smiles = ["CCO"]
        batch = trained_tok.batch_encode(smiles, max_length=100)
        ids = trained_tok.encode("CCO")
        # Everything after the sequence should be pad_token_id
        assert np.all(batch[0, len(ids):] == trained_tok.pad_token_id)

    def test_batch_empty(self, trained_tok):
        batch = trained_tok.batch_encode([])
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (0, 0)
        assert trained_tok.batch_decode([]) == []

    def test_batch_decode(self, trained_tok):
        smiles = ["CCCOCC", "CCO"]
        ids_list = [trained_tok.encode(s) for s in smiles]
        decoded = trained_tok.batch_decode(ids_list)
        for smi, dec in zip(smiles, decoded):
            safe_enc = safe_oxidizer.safe_encode(smi)
            assert dec == safe_enc


class TestShuffle:
    @pytest.fixture
    def trained_tok(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES), vocab_size=350)
        return tok

    def test_shuffle_returns_valid_tokens(self, trained_tok):
        smiles = ["CCCOCC", "CCO"]
        batch = trained_tok.batch_encode(smiles, shuffle=True)
        assert isinstance(batch, np.ndarray)
        assert batch.dtype == np.uint32
        assert batch.shape[0] == 2

    def test_shuffle_same_fragments(self, trained_tok):
        smi = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
        normal_ids = trained_tok.encode(smi)
        normal_decoded = trained_tok.decode(normal_ids)
        normal_frags = set(normal_decoded.split("."))

        shuffled_ids = trained_tok.batch_encode([smi], shuffle=True)[0]
        # Strip padding
        pad = trained_tok.pad_token_id
        shuffled_ids = [int(x) for x in shuffled_ids if x != pad]
        shuffled_decoded = trained_tok.decode(shuffled_ids)
        shuffled_frags = set(shuffled_decoded.split("."))

        assert normal_frags == shuffled_frags

    def test_shuffle_produces_different_orderings(self, trained_tok):
        smi = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
        orderings = set()
        for _ in range(20):
            ids = trained_tok.batch_encode([smi], shuffle=True)[0]
            pad = trained_tok.pad_token_id
            ids_tuple = tuple(int(x) for x in ids if x != pad)
            orderings.add(ids_tuple)
        # With 4 fragments (4! = 24 permutations), 20 tries should yield >1 ordering
        assert len(orderings) > 1


class TestSaveLoad:
    @pytest.fixture
    def trained_tok(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES), vocab_size=350)
        return tok

    def test_save_load_roundtrip(self, trained_tok):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        trained_tok.save(path)
        loaded = SafeTokenizer.load(path)

        assert loaded.vocab_size == trained_tok.vocab_size
        assert loaded.dot_token_id == trained_tok.dot_token_id

        # Same encoding results
        test_smiles = ["CCCOCC", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"]
        for smi in test_smiles:
            assert loaded.encode(smi) == trained_tok.encode(smi)

        Path(path).unlink()


class TestSpecialTokens:
    def test_special_token_ids_unique(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES[:10]), vocab_size=270)
        ids = {tok.dot_token_id, tok.pad_token_id, tok.bos_token_id, tok.eos_token_id, tok.unk_token_id}
        assert len(ids) == 5

    def test_special_tokens_above_merges(self):
        tok = SafeTokenizer()
        tok.train_from_iterator(iter(TRAIN_SMILES[:10]), vocab_size=270)
        min_special = min(tok.dot_token_id, tok.pad_token_id, tok.bos_token_id, tok.eos_token_id, tok.unk_token_id)
        # All special tokens should be >= 256 + num_merges
        num_merges = tok.vocab_size - 256 - 5
        assert min_special >= 256 + num_merges
