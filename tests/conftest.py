"""Offline test fixtures: a stub word-level tokenizer and a tiny GPT-2. Nothing is downloaded."""
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

WORDS = ("the staff was rude helpful price too high broken arrived store crowded queue clean "
         "great and very is a it comment category").split()


@pytest.fixture
def stub_tokenizer():
    vocab = {"<unk>": 0, "<eos>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(WORDS + [":", "."])})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


@pytest.fixture
def toy_model(stub_tokenizer):
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(stub_tokenizer), n_embd=32, n_layer=2, n_head=2, n_positions=64,
                     bos_token_id=1, eos_token_id=1)
    return GPT2LMHeadModel(cfg)
