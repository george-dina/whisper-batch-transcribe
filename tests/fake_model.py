"""A WhisperModel whose neural network is replaced by a deterministic stub.

Everything else (feature extraction, the batched pipeline, segment splitting,
tokenizer logic) is real faster-whisper code. Each test clip is a pure tone;
the stub "recognises" a clip by the dominant mel bin of its features and
outputs the word registered for that tone. That makes it obvious if a batched
transcript ever ends up attached to the wrong file.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES
from faster_whisper.utils import get_logger
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

SAMPLE_RATE = 16000
WORDS = [f"clip{i}" for i in range(80)]


def build_tokenizer() -> Tokenizer:
    specials = ["<|endoftext|>", "<|startoftranscript|>"]
    specials += [f"<|{code}|>" for code in _LANGUAGE_CODES]
    specials += ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"]
    specials += [f"<|{i * 0.02:.2f}|>" for i in range(1501)]
    vocab = {tok: i for i, tok in enumerate(WORDS + ["[UNK]"] + specials)}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return tok


def tone(freq: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class StubNetwork:
    """Stands in for ctranslate2.models.Whisper."""

    is_multilingual = True
    n_mels = 80
    device = "cpu"
    device_index = [0]

    def __init__(self, bin_to_word: dict[int, int], bad_bins: set[int]):
        self.bin_to_word = bin_to_word
        self.bad_bins = bad_bins
        self.calls: list[int] = []  # batch size of every generate() call

    def encode(self, features, to_cpu=False):
        return np.array(features)

    def generate(self, encoder_output, prompts, **kwargs):
        self.calls.append(len(prompts))
        results = []
        for feats in encoder_output:
            b = dominant_bin(feats)
            word = self.bin_to_word.get(b, 0)
            # "bad" clips decode with low confidence, but only when batched
            score = -5.0 if (b in self.bad_bins and len(prompts) > 1) else -0.1
            results.append(SimpleNamespace(sequences_ids=[[word]], scores=[score], no_speech_prob=0.0))
        return results

    def detect_language(self, encoder_output):
        return [[("<|ro|>", 1.0)] for _ in range(len(encoder_output))]


def dominant_bin(feats: np.ndarray) -> int:
    return int(np.argmax(np.asarray(feats)[:, :50].mean(axis=1)))


def make_model(freqs: list[float], bad_freqs: set[float] = frozenset()) -> tuple[WhisperModel, dict[float, str]]:
    """Return a stub model plus the word each tone frequency will be transcribed as."""
    fe = FeatureExtractor()
    bins = {f: dominant_bin(fe(tone(f, 2.0))) for f in freqs}
    assert len(set(bins.values())) == len(freqs), "test tones must land in distinct mel bins"
    bin_to_word = {b: i for i, b in enumerate(bins.values())}

    m = WhisperModel.__new__(WhisperModel)
    m.logger = get_logger()
    m.model = StubNetwork(bin_to_word, {bins[f] for f in bad_freqs})
    m.hf_tokenizer = build_tokenizer()
    m.feat_kwargs = {}
    m.feature_extractor = fe
    m.input_stride = 2
    m.num_samples_per_token = fe.hop_length * m.input_stride
    m.frames_per_second = fe.sampling_rate // fe.hop_length
    m.tokens_per_second = fe.sampling_rate // m.num_samples_per_token
    m.time_precision = 0.02
    m.max_length = 448
    return m, {f: WORDS[bin_to_word[bins[f]]] for f in freqs}
