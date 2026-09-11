"""Configuration. Edit the defaults here — this file *is* the config.

Every config object is a flat dataclass of scalars and strings. That is a
deliberate constraint, not an aesthetic one: the eventual native library exposes
a C config struct, and anything that cannot be expressed as a scalar or string
here would not survive the port.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_PERSONA = (
    "A composed middle-aged man with a warm, resonant baritone and a steady, "
    "unhurried delivery."
)
DEFAULT_DELIVERY = "Speak naturally and clearly, at a conversational pace."

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful in-game character. Answer in plain spoken prose: no "
    "markdown, no lists, no code blocks, no emoji. Keep answers to a few "
    "sentences unless asked for more."
)


@dataclass(slots=True)
class LlmConfig:
    """llama.cpp server, OpenAI-compatible /v1/chat/completions."""

    base_url: str = "http://127.0.0.1:8080"
    model: str = "local"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.95
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 5.0
    # Qwen3 emits <think> blocks unless told not to; textprep strips them either
    # way, but disabling saves tokens and latency.
    disable_thinking: bool = True


@dataclass(slots=True)
class SegmentationConfig:
    """Knobs forwarded to stream2sentence.

    Names mirror the library's parameters exactly so they can be passed through
    without translation. Unknown names are dropped with a warning by
    ``segmentation.py`` rather than raising, because these parameters have
    drifted between stream2sentence releases.
    """

    # Speaking on the first clause is the single biggest lever on
    # time-to-first-audio, and it is off anyway — turning it on costs voice
    # identity. A clause is a handful of characters, and VoiceDesign draws its
    # speaker per generation with less context the shorter the text, so the
    # opening fragment came out audibly as a different person and everything
    # after it settled. Judged by ear; consistency won.
    #
    # The two lengths below only shape the quick-yield fragment, so they are
    # inert while it is off. Left at their tuned values for anyone turning it
    # back on to trade identity for latency.
    quick_yield_single_sentence_fragment: bool = False
    minimum_first_fragment_length: int = 12
    minimum_sentence_length: int = 24
    force_first_fragment_after_words: int = 8
    # Guards. never_split_numbers keeps "3.14" intact; the nltk consensus
    # tokenizer is what keeps "Dr." from ending a sentence.
    never_split_numbers: bool = True
    # Not a stream2sentence parameter: applied by segmentation.py after it.
    #
    # Every segment becomes its own TTS generation, and VoiceDesign re-samples
    # the speaker on each one, so more segments means more voice drift. Judged by
    # ear: six 9-21 char segments drift clearly, three 56-65 char segments much
    # less, the same text in one call not at all. Coalescing short segments up to
    # this many characters buys most of that back, and it applies to the *first*
    # segment too — a short opener is exactly the one that comes out as someone
    # else. That is the whole time-to-first-audio cost: roughly this many
    # characters of LLM output before speech starts. 0 disables it.
    coalesce_min_chars: int = 60
    tokenizer: str = "nltk+rule-based"
    language: str = "en"
    cleanup_text_links: bool = True
    cleanup_text_emojis: bool = True
    filter_first_non_alnum_characters: bool = True


@dataclass(slots=True)
class TtsConfig:
    backend: str = "qwen3_torch"  # qwen3_torch | fake
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    # Relative to the checkout root, not the working directory — models/ is
    # shared with the cpp/ port and sits above py/. See paths.py.
    local_dir: str = "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    language: str = "English"
    # Codec frames per yielded chunk. 8 ~= 667 ms of audio; 2-4 lowers TTFA at
    # some throughput cost. Swept by bench/latency.py.
    chunk_size: int = 8
    seed: int = 1234
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    # MUST stay "sdpa" on RTX 50-series: flash-attn 2.x does not build for
    # sm_120 (nvcc segfaults). See README "GPU setup".
    attn_implementation: str = "sdpa"
    sample_rate: int = 24000
    prebuffer_chunks: int = 0


@dataclass(slots=True)
class AudioConfig:
    device: int = -1  # -1 -> system default
    ring_seconds: float = 2.0
    blocksize: int = 512
    enabled: bool = True


@dataclass(slots=True)
class StyleConfig:
    """The out-of-band voice channel.

    ``persona`` is held byte-identical across utterances so the speaker identity
    stays put; only ``delivery`` moves. Neither ever reaches the LLM.
    """

    persona: str = DEFAULT_PERSONA
    delivery: str = DEFAULT_DELIVERY
    # Named voices live in a JSON file, not here: a character is user data that
    # accumulates, and this file has to stay portable to a C config struct.
    # Relative paths anchor at the checkout root (see paths.py), so the library
    # is the same file whether uv ran from py/ or from the root.
    voices_path: str = "voices.json"
    # Name of a saved voice to start with. Empty uses persona/seed above.
    voice: str = ""


@dataclass(slots=True)
class BudgetConfig:
    vram_cap_gb: float = 10.0
    enforce: bool = True


@dataclass(slots=True)
class AppConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
