"""Interactive REPL: type text, hear the answer streaming.

Commands:
    /style <text>       change delivery (emotion, volume, pacing) — out-of-band channel
    /persona <text>     change who is speaking (redraws the voice by design)
    /seed [n]           show or set the sampling seed; with persona, this *is* the voice
    /instruct           show the composed instruct string sent to the TTS
    /audition [n]       speak one line across n seeds so you can pick a voice by ear
    /voices             list saved voices
    /voice <name>       switch to a saved voice
    /voice save <name>  save the current persona + seed under a name
    /voice rm <name>    delete a saved voice
    /help, /quit

Designing a character: /persona ... then /audition, /seed the one you liked,
then /voice save <name>. A voice is just persona + seed, so it costs ~300 bytes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from .app import App, apply_voice
from .config import AppConfig
from .pipeline import Pipeline
from .voices import Voice

# Long enough to hear a voice's character, short enough to sit through N of them.
_AUDITION_LINE = (
    "I have been waiting here since the rain stopped, and I am not leaving "
    "until we have settled this properly."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmvoice", description=__doc__.split("\n")[0])
    p.add_argument(
        "--tts-backend",
        choices=["qwen3_torch", "fake"],
        help="override tts.backend; 'fake' exercises the full pipeline with no GPU",
    )
    p.add_argument(
        "--fake-llm",
        action="store_true",
        help="canned LLM replies: no llama-server needed",
    )
    p.add_argument("--no-audio", action="store_true", help="do not open an output device")
    p.add_argument("--voice", help="start with a saved voice (see /voices)")
    p.add_argument("-p", "--prompt", action="append", help="run a prompt and exit (repeatable)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # INFO, not DEBUG: stream2sentence logs per-segment timings to the *root*
    # logger, which cannot be quieted separately from our own output.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Defaults live in config.py; edit them there.
    cfg = AppConfig()
    if args.tts_backend:
        cfg.tts.backend = args.tts_backend
    if args.no_audio:
        cfg.audio.enabled = False
    if args.voice:
        cfg.style.voice = args.voice

    try:
        with App(cfg, fake_llm=args.fake_llm) as app:
            _announce(app)
            if args.prompt:
                for prompt in args.prompt:
                    _speak(app.pipeline, prompt)
                return 0
            return _repl(app)
    # KeyError/ValueError: an unknown --voice or a malformed voices.json, both of
    # which are user input and deserve a message rather than a traceback.
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"error: {_message(exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _announce(app: App) -> None:
    print(f"llmvoice — tts={app.cfg.tts.backend} sample_rate={app.tts.sample_rate}")
    voice = app.cfg.style.voice
    print(f"voice: {voice or '(unsaved)'} seed={app.style.seed} — {len(app.voices)} saved")
    print(f"instruct: {app.style.snapshot().instruct}")
    for warning in app.style.warnings():
        print(f"  note: {warning}")
    print("Type a prompt, or /help. Ctrl+C interrupts speech.\n")


def _speak(pipeline: Pipeline, prompt: str) -> None:
    pipeline.on_segment = lambda segment: print(f"  │ {segment}")
    try:
        m = pipeline.speak(prompt)
    except KeyboardInterrupt:
        pipeline.cancel()
        print("\n  (interrupted)")
        return
    finally:
        pipeline.on_segment = None
    print(f"  {m.summary()}  underruns={pipeline.metrics.underruns}")


def _repl(app: App) -> int:
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if _command(app, line):
                return 0
            continue
        _speak(app.pipeline, line)


def _command(app: App, line: str) -> bool:
    """Handle a slash command. Returns True to exit."""
    cmd, _, rest = line.partition(" ")
    rest = rest.strip()
    match cmd:
        case "/quit" | "/exit":
            return True
        case "/help":
            print(__doc__)
        case "/style":
            if not rest:
                print(f"  delivery: {app.style.delivery}")
            else:
                app.style.set_delivery(rest)
                print(f"  delivery -> {rest}")
                for warning in app.style.warnings():
                    print(f"  note: {warning}")
        case "/persona":
            if not rest:
                print(f"  persona: {app.style.persona}")
            else:
                app.style.set_persona(rest)
                print("  persona changed — expect the voice identity to shift")
        case "/instruct":
            print(f"  {app.style.snapshot().instruct}")
        case "/seed":
            _seed(app, rest)
        case "/audition":
            _audition(app, rest)
        case "/voices":
            _list_voices(app)
        case "/voice":
            _voice(app, rest)
        case _:
            print(f"  unknown command {cmd}; try /help")
    return False


def _message(exc: Exception) -> str:
    """KeyError stringifies as repr(arg), which would show the message in quotes."""
    return str(exc.args[0]) if exc.args else str(exc)


def _seed(app: App, rest: str) -> None:
    if not rest:
        print(f"  seed: {app.style.seed}")
        return
    try:
        app.style.set_seed(int(rest))
    except ValueError:
        print(f"  seed must be an integer, got {rest!r}")
        return
    print(f"  seed -> {rest}")


def _audition(app: App, rest: str) -> None:
    """Speak one fixed line across several seeds.

    Same words every time and one generation each, so the only thing that differs
    between takes is the seed — which is what you are trying to judge.
    """
    count = 5
    if rest:
        try:
            count = int(rest)
        except ValueError:
            print(f"  /audition takes a count, got {rest!r}")
            return
        if count < 1:
            print("  /audition needs a count of at least 1")
            return
    style = app.style.snapshot()
    print(f"  auditioning {count} seeds for the current persona:")
    for seed in range(style.seed, style.seed + count):
        print(f"  │ seed {seed}")
        try:
            app.pipeline.say(_AUDITION_LINE, replace(style, seed=seed))
        except KeyboardInterrupt:
            app.pipeline.cancel()
            print("\n  (interrupted)")
            return
    print("  pick one with /seed <n>, then /voice save <name>")


def _list_voices(app: App) -> None:
    if not len(app.voices):
        print(f"  no voices saved in {app.voices.path}; make one with /voice save <name>")
        return
    active = app.style.snapshot()
    for voice in app.voices.all():
        here = " *" if (voice.persona == active.persona and voice.seed == active.seed) else "  "
        print(f" {here} {voice.name}  seed={voice.seed}  {voice.note}".rstrip())
        print(f"      {voice.persona}")


def _voice(app: App, rest: str) -> None:
    sub, _, name = rest.partition(" ")
    name = name.strip()
    match sub:
        case "":
            print("  /voice <name> | /voice save <name> | /voice rm <name>")
        case "save":
            if not name:
                print("  /voice save needs a name")
                return
            style = app.style.snapshot()
            existed = name in app.voices
            app.voices.add(Voice(name=name, persona=style.persona, seed=style.seed))
            app.voices.save()
            verb = "updated" if existed else "saved"
            print(f"  {verb} {name!r} (seed={style.seed}) in {app.voices.path}")
        case "rm" | "delete":
            if not name:
                print("  /voice rm needs a name")
                return
            try:
                app.voices.remove(name)
            except KeyError as exc:
                print(f"  {_message(exc)}")
                return
            app.voices.save()
            print(f"  removed {name!r}")
        case _:
            # No subcommand: the whole argument is a voice name to activate.
            try:
                voice = app.voices.get(rest.strip())
            except KeyError as exc:
                print(f"  {_message(exc)}")
                return
            apply_voice(app.style, voice)
            print(f"  voice -> {voice.name} (seed={voice.seed})")


if __name__ == "__main__":
    raise SystemExit(main())
