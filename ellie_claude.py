#!/usr/bin/env python3
# Copyright (c) 2026 Shreyash Padhi. All rights reserved.
# Licensed under the Ellie Source Available License v1.0 — see LICENSE file.
# Contact: shreyashpadhi101@gmail.com for commercial licensing.
"""
ellie_claude.py — Windowed Claude assistant with voice input (PTY edition)

Embeds a proper ConPTY terminal running `claude` inside a tkinter window.
Hold F9 to speak; release to transcribe and send text to Claude.
Say "Hey Ellie" to activate hands-free; stays in active mode for follow-ups.
Registers itself to run at Windows startup automatically.

Dependencies:
    pip install faster-whisper sounddevice numpy keyboard pywinpty pyte
"""

import collections
import ctypes
import os
import re
import shutil
import site
import sys
import threading
import time
import tkinter as tk
import winreg
from tkinter import font as tkfont

# ── CUDA DLL paths ────────────────────────────────────────────────────────────
# Must run before importing GPU-backed packages so NVIDIA DLLs are on PATH.
for _sp in site.getsitepackages():
    for _pkg in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin"):
        _dll_path = os.path.join(_sp, _pkg.replace("/", os.sep))
        if os.path.isdir(_dll_path) and _dll_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _dll_path + os.pathsep + os.environ.get("PATH", "")

import keyboard  # noqa: E402
import numpy as np  # noqa: E402
import pyte  # noqa: E402
import sounddevice as sd  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from winpty import PtyProcess  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
HOTKEY = "f9"
WAKE_MODEL_SIZE = "tiny.en"
WAKE_WORD_USER = "hey ellie"
WAKE_WORD = "ellie"
WAKE_WINDOW_S = 2.0
WAKE_STEP_S = 0.5
APP_NAME = "ellie_claude"
FONT_SIZE = 11
REDRAW_MS = 50  # ~20 fps

SILENCE_RMS = 0.008
SILENCE_S = 1.5

STOP_HOTKEY = "escape"
STOP_PHRASE = "stop listening"
PTT_HOTKEY = "f10"

RESPONSE_COOLDOWN_S = 3.5  # pause before listening again after a command is sent

# ── Catppuccin Mocha palette ──────────────────────────────────────────────────
TERM_BG = "#0c0c0c"
TERM_FG = "#cdd6f4"

PYTE_COLOR_MAP: dict[str, str] = {
    "default": TERM_FG,
    "black": "#45475a",
    "red": "#f38ba8",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "blue": "#89b4fa",
    "magenta": "#cba6f7",
    "cyan": "#89dceb",
    "white": "#cdd6f4",
    "brightblack": "#585b70",
    "brightred": "#f38ba8",
    "brightgreen": "#a6e3a1",
    "brightyellow": "#f9e2af",
    "brightblue": "#89b4fa",
    "brightmagenta": "#cba6f7",
    "brightcyan": "#89dceb",
    "brightwhite": "#ffffff",
}

SPECIAL_KEY_MAP: dict[str, str] = {
    "Return": "\r",
    "KP_Enter": "\r",
    "BackSpace": "\x7f",
    "Tab": "\t",
    "Escape": "\x1b",
    "Up": "\x1b[A",
    "Down": "\x1b[B",
    "Right": "\x1b[C",
    "Left": "\x1b[D",
    "Home": "\x1b[H",
    "End": "\x1b[F",
    "Delete": "\x1b[3~",
    "Insert": "\x1b[2~",
    "Prior": "\x1b[5~",
    "Next": "\x1b[6~",
    "F1": "\x1bOP",
    "F2": "\x1bOQ",
    "F3": "\x1bOR",
    "F4": "\x1bOS",
    "F5": "\x1b[15~",
    "F6": "\x1b[17~",
    "F7": "\x1b[18~",
    "F8": "\x1b[19~",
    # F9 intentionally omitted — it is the voice hotkey
    "F10": "\x1b[21~",
    "F11": "\x1b[23~",
    "F12": "\x1b[24~",
}

VOICE_OPTION_MAP: dict[str, str] = {
    "yes": "y\r",
    "yeah": "y\r",
    "yep": "y\r",
    "yup": "y\r",
    "no": "n\r",
    "nope": "n\r",
    "nah": "n\r",
    "one": "1\r",
    "1": "1\r",
    "two": "2\r",
    "2": "2\r",
    "three": "3\r",
    "3": "3\r",
    "four": "4\r",
    "4": "4\r",
    "five": "5\r",
    "5": "5\r",
    "six": "6\r",
    "6": "6\r",
    "seven": "7\r",
    "7": "7\r",
    "eight": "8\r",
    "8": "8\r",
    "nine": "9\r",
    "9": "9\r",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "enter": "\r",
    "return": "\r",
    "escape": "\x1b",
    "cancel": "\x1b",
    "back": "\x1b",
}

# ── Gibberish / hallucination filter ─────────────────────────────────────────

# Whisper commonly outputs these when it hears silence or background noise.
_WHISPER_HALLUCINATIONS: frozenset[str] = frozenset(
    {
        "thank you",
        "thanks for watching",
        "thanks for watching!",
        "thank you so much",
        "thank you very much",
        "please subscribe",
        "like and subscribe",
        "you",
        "bye",
        "bye bye",
        "goodbye",
        "okay",
        "ok",
        "um",
        "uh",
        "hmm",
        "hm",
        "ah",
        "oh",
    }
)


def _is_gibberish(text: str) -> bool:
    """Return True if the transcription looks like noise, silence, or a hallucination."""
    clean = text.strip().lower()

    # Empty or a known whisper hallucination phrase
    if not clean or clean.rstrip(".,!? ") in _WHISPER_HALLUCINATIONS:
        return True

    # Must have at least 2 alphabetic characters
    alpha = [c for c in clean if c.isalpha()]
    if len(alpha) < 2:
        return True

    # Alphabetic ratio too low — mostly symbols/numbers with no real words
    if len(alpha) / len(clean) < 0.4:
        return True

    # Repetitive: any single word repeated more than 3 times (e.g. "ha ha ha ha ha")
    words = clean.split()
    if words and max(words.count(w) for w in set(words)) > 3:
        return True

    # Repetitive character run: same char 5+ times in a row (e.g. "aaaaaa")
    if re.search(r"(.)\1{4,}", clean):
        return True

    return False


# ── 256-color helper ──────────────────────────────────────────────────────────


def _256_to_hex(n: int) -> str:
    if n < 16:
        palette = [
            "#000000",
            "#800000",
            "#008000",
            "#808000",
            "#000080",
            "#800080",
            "#008080",
            "#c0c0c0",
            "#808080",
            "#ff0000",
            "#00ff00",
            "#ffff00",
            "#0000ff",
            "#ff00ff",
            "#00ffff",
            "#ffffff",
        ]
        return palette[n] if n < len(palette) else "#ffffff"
    if n < 232:
        n -= 16
        b = n % 6
        g = (n // 6) % 6
        r = n // 36

        def _c(x: int) -> int:
            return 0 if x == 0 else 55 + x * 40

        return f"#{_c(r):02x}{_c(g):02x}{_c(b):02x}"
    v = 8 + (n - 232) * 10
    return f"#{v:02x}{v:02x}{v:02x}"


def _resolve_color(color, is_bg: bool = False) -> str:
    if color == "default":
        return TERM_BG if is_bg else TERM_FG
    if isinstance(color, int):
        return _256_to_hex(color)
    return PYTE_COLOR_MAP.get(str(color).lower(), TERM_BG if is_bg else TERM_FG)


# ── Audio feedback tones ──────────────────────────────────────────────────────


def _play_tone(freqs: list[float], durations: list[float], vol: float = 0.35, gap_s: float = 0.04):
    """Synthesise and play a sequence of sine-wave tones asynchronously."""

    def _run():
        chunks = []
        fade_n = int(0.008 * SAMPLE_RATE)
        for i, (freq, dur) in enumerate(zip(freqs, durations, strict=False)):
            n = int(SAMPLE_RATE * dur)
            t = np.linspace(0, dur, n, endpoint=False)
            wave = vol * np.sin(2 * np.pi * freq * t).astype(np.float32)
            fade = min(fade_n, n // 4)
            wave[:fade] *= np.linspace(0, 1, fade)
            wave[-fade:] *= np.linspace(1, 0, fade)
            chunks.append(wave)
            if i < len(freqs) - 1:
                chunks.append(np.zeros(int(SAMPLE_RATE * gap_s), dtype=np.float32))
        try:
            sd.play(np.concatenate(chunks), SAMPLE_RATE)
            sd.wait()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def _tone_wake():
    _play_tone([660, 880], [0.10, 0.14])


def _tone_rec_start():
    _play_tone([330], [0.07], vol=0.25)


def _tone_rec_stop():
    _play_tone([880, 660], [0.08, 0.10])


def _tone_sent():
    _play_tone([1047, 1319], [0.10, 0.16])


def _tone_active():
    _play_tone([528], [0.08], vol=0.20)


def _tone_error():
    _play_tone([220], [0.20], vol=0.30)


# ── TerminalWidget ────────────────────────────────────────────────────────────


class TerminalWidget(tk.Frame):
    """Tkinter widget that wraps a ConPTY session via pywinpty + pyte."""

    def __init__(self, parent, command: list[str], **kwargs):
        super().__init__(parent, bg=TERM_BG, **kwargs)

        self._font = self._detect_font()

        self._cols = 120
        self._rows = 36
        self._screen = pyte.Screen(self._cols, self._rows)
        self._stream = pyte.ByteStream(self._screen)
        self._screen_lock = threading.Lock()
        self._dirty = False
        self._first_render = True
        self._need_full_redraw = False

        self._pty: PtyProcess | None = None
        self._pty_lock = threading.Lock()

        self._tag_cache: dict[tuple, str] = {}
        self._tag_counter = 0

        self._text = tk.Text(
            self,
            font=self._font,
            bg=TERM_BG,
            fg=TERM_FG,
            insertbackground=TERM_FG,
            relief=tk.FLAT,
            borderwidth=0,
            wrap=tk.NONE,
            cursor="xterm",
            state=tk.DISABLED,
        )
        vsb = tk.Scrollbar(self, orient=tk.VERTICAL, command=self._text.yview)
        hsb = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self._text.xview)
        self._text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self._text.pack(fill=tk.BOTH, expand=True)

        self._text.bind("<Key>", self._on_key)
        self._text.focus_set()
        self.bind("<Configure>", self._on_resize)

        threading.Thread(target=self._run_pty, args=(command,), daemon=True).start()
        self.after(REDRAW_MS, self._redraw_loop)

    # ── Font detection ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_font() -> tuple[str, int]:
        available = set(tkfont.families())
        for name in ("Cascadia Code", "Cascadia Mono", "Consolas", "Courier New", "Courier"):
            if name in available:
                return (name, FONT_SIZE)
        return ("Courier New", FONT_SIZE)

    # ── PTY lifecycle ──────────────────────────────────────────────────────

    def _run_pty(self, command: list[str]):
        try:
            env = os.environ.copy()
            env.setdefault("TERM", "xterm-256color")
            env.setdefault("COLORTERM", "truecolor")
            extra_paths = [
                os.path.expanduser(r"~\.local\bin"),
                os.path.join(os.environ.get("APPDATA", ""), "npm"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "npm"),
            ]
            env["PATH"] = (
                os.pathsep.join(p for p in extra_paths if os.path.isdir(p))
                + os.pathsep
                + env.get("PATH", "")
            )
            pty = PtyProcess.spawn(command, dimensions=(self._rows, self._cols), env=env)
        except Exception as exc:
            msg = f"Failed to start terminal: {exc}"
            self.after(0, lambda: self._show_error(msg))
            return

        with self._pty_lock:
            self._pty = pty

        while pty.isalive():
            try:
                data = pty.read(4096)
                if not data:
                    continue
                raw = data.encode("utf-8", errors="replace") if isinstance(data, str) else data
                with self._screen_lock:
                    self._stream.feed(raw)
                self._dirty = True

            except EOFError:
                break
            except Exception:
                break

    def _show_error(self, msg: str):
        self._text.config(state=tk.NORMAL)
        self._text.insert(tk.END, f"\n[{msg}]\n")
        self._text.config(state=tk.DISABLED)

    # ── Rendering ──────────────────────────────────────────────────────────

    def _redraw_loop(self):
        if self._dirty:
            self._dirty = False
            self._redraw()
        self.after(REDRAW_MS, self._redraw_loop)

    def _get_tag(self, fg: str, bg: str, bold: bool, italic: bool, reverse: bool) -> str:
        if reverse:
            fg, bg = bg, fg
        key = (fg, bg, bold, italic)
        if key in self._tag_cache:
            return self._tag_cache[key]
        name = f"t{self._tag_counter}"
        self._tag_counter += 1
        style = " ".join(s for s in ("bold" if bold else "", "italic" if italic else "") if s)
        self._text.tag_configure(
            name,
            foreground=fg,
            background=bg,
            font=(*self._font, style) if style else self._font,
        )
        self._tag_cache[key] = name
        return name

    def _build_segments(self, line: dict, cols: int) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        cur_tag: str | None = None
        cur_chars: list[str] = []

        for x in range(cols):
            ch = line.get(x)
            data = ch.data if ch else " "
            fg = _resolve_color(ch.fg if ch else "default", is_bg=False)
            bg = _resolve_color(ch.bg if ch else "default", is_bg=True)
            bold = ch.bold if ch else False
            italic = getattr(ch, "italics", getattr(ch, "italic", False)) if ch else False
            reverse = ch.reverse if ch else False
            tag = self._get_tag(fg, bg, bold, italic, reverse)
            if tag != cur_tag:
                if cur_chars and cur_tag is not None:
                    segments.append(("".join(cur_chars), cur_tag))
                cur_tag = tag
                cur_chars = [data]
            else:
                cur_chars.append(data)

        if cur_chars and cur_tag is not None:
            segments.append(("".join(cur_chars), cur_tag))
        return segments

    def _redraw(self):
        with self._screen_lock:
            full = self._first_render or self._need_full_redraw
            if full:
                dirty_lines = set(range(self._screen.lines))
            else:
                dirty_lines = set(self._screen.dirty)
            self._screen.dirty.clear()
            buf = {y: dict(self._screen.buffer.get(y, {})) for y in dirty_lines}
            cur_x = self._screen.cursor.x
            cur_y = self._screen.cursor.y
            cols = self._screen.columns
            rows = self._screen.lines

        if not dirty_lines:
            return

        self._text.config(state=tk.NORMAL)

        if full:
            self._text.delete("1.0", tk.END)
            for y in range(rows):
                for text, tag in self._build_segments(buf.get(y, {}), cols):
                    self._text.insert(tk.END, text, tag)
                if y < rows - 1:
                    self._text.insert(tk.END, "\n")
            self._first_render = False
            self._need_full_redraw = False
        else:
            for y in sorted(dirty_lines):
                if y >= rows:
                    continue
                self._text.delete(f"{y + 1}.0", f"{y + 1}.end")
                for text, tag in self._build_segments(buf.get(y, {}), cols):
                    self._text.insert(f"{y + 1}.end", text, tag)

        try:
            idx = f"{cur_y + 1}.{cur_x}"
            self._text.mark_set(tk.INSERT, idx)
            self._text.see(idx)
        except Exception:
            pass

        self._text.config(state=tk.DISABLED)

    # ── Input ──────────────────────────────────────────────────────────────

    def _on_key(self, event: tk.Event):
        with self._pty_lock:
            pty = self._pty
        if pty is None or not pty.isalive():
            return "break"

        ctrl = bool(event.state & 0x4)

        if ctrl and len(event.keysym) == 1 and event.keysym.isalpha():
            code = ord(event.keysym.upper()) - 64
            if 1 <= code <= 26:
                pty.write(chr(code))
            return "break"

        if event.keysym in SPECIAL_KEY_MAP:
            pty.write(SPECIAL_KEY_MAP[event.keysym])
            return "break"

        if event.char:
            pty.write(event.char)
        return "break"

    # ── Resize ─────────────────────────────────────────────────────────────

    def _on_resize(self, event: tk.Event):
        try:
            f = tkfont.Font(family=self._font[0], size=self._font[1])
            cw = f.measure("M")
            ch = f.metrics("linespace")
        except Exception:
            return
        if cw <= 0 or ch <= 0:
            return
        new_cols = max(20, event.width // cw)
        new_rows = max(5, event.height // ch)
        if new_cols == self._cols and new_rows == self._rows:
            return
        self._cols, self._rows = new_cols, new_rows
        with self._screen_lock:
            self._screen.resize(new_rows, new_cols)
        with self._pty_lock:
            if self._pty and self._pty.isalive():
                self._pty.setwinsize(new_rows, new_cols)
        self._need_full_redraw = True
        self._dirty = True

    # ── Public API ─────────────────────────────────────────────────────────

    def send_text(self, text: str):
        with self._pty_lock:
            pty = self._pty
        if pty and pty.isalive():
            pty.write(text + "\r")

    def send_raw(self, seq: str):
        with self._pty_lock:
            pty = self._pty
        if pty and pty.isalive():
            pty.write(seq)

    def terminate(self):
        with self._pty_lock:
            if self._pty:
                try:
                    self._pty.terminate(force=True)
                except Exception:
                    pass


# ── EllieApp ──────────────────────────────────────────────────────────────────


class EllieApp:
    """
    State machine for voice input:

      active        — default; listens for wake word, then records command
      ptt_idle      — push-to-talk mode; waiting for F9 tap
      recording_ptt — recording a PTT command (silence-gated)
      transcribing  — processing audio (shared by both modes)

      Transitions:
        startup                          → active
        active       + F9               → ptt_idle
        active       + ESC / "stop"     → ptt_idle
        ptt_idle     + F9 tap           → recording_ptt
        recording_ptt + silence         → transcribing → ptt_idle
        any          + F10              → active
        transcribing → (active source)  → active  (with cooldown)
        transcribing → (ptt source)     → ptt_idle
    """

    def __init__(self):
        self._ensure_single_instance()
        self.root = tk.Tk()
        self.root.title("Ellie — Claude")
        self.root.geometry("1280x820")
        self.root.configure(bg="#1e1e2e")
        self.root.minsize(800, 500)

        # Voice state — starts in ptt_idle while models load, switches to active when ready
        self._state = "ptt_idle"
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._last_sent_at: float = 0.0

        # Rolling buffer for wake-word analysis
        _wake_cap = int(WAKE_WINDOW_S * SAMPLE_RATE)
        self._wake_buf: collections.deque[np.ndarray] = collections.deque(maxlen=_wake_cap)

        self.model: WhisperModel | None = None
        self._wake_model: WhisperModel | None = None

        self._terminal: TerminalWidget | None = None

        self._build_ui()
        self._register_startup()
        self._start_voice_thread()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        bar = tk.Frame(self.root, bg="#313244", height=44)
        bar.pack(fill=tk.X, side=tk.TOP)
        bar.pack_propagate(False)

        tk.Label(
            bar,
            text="⚡  Ellie  ×  Claude",
            bg="#313244",
            fg="#cdd6f4",
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT, padx=14, pady=10)

        tk.Label(
            bar,
            text=f'say "{WAKE_WORD_USER}"  |  {HOTKEY.upper()} for push-to-talk  |  {PTT_HOTKEY.upper()} for active listening  |  {STOP_HOTKEY.upper()} to stop',
            bg="#313244",
            fg="#45475a",
            font=("Segoe UI", 9),
        ).pack(side=tk.RIGHT, padx=14, pady=10)

        self._status_var = tk.StringVar(value="● Warming up …")
        self._status_lbl = tk.Label(
            bar,
            textvariable=self._status_var,
            bg="#313244",
            fg="#45475a",
            font=("Consolas", 10),
        )
        self._status_lbl.pack(side=tk.RIGHT, padx=8, pady=10)

        _extra_path = os.pathsep.join(
            filter(
                os.path.isdir,
                [
                    os.path.expanduser(r"~\.local\bin"),
                    os.path.join(os.environ.get("APPDATA", ""), "npm"),
                    os.path.join(os.environ.get("LOCALAPPDATA", ""), "npm"),
                ],
            )
        )
        claude_exe = (
            shutil.which("claude")
            or shutil.which("claude", path=_extra_path + os.pathsep + os.environ.get("PATH", ""))
            or os.path.expanduser(r"~\.local\bin\claude.exe")
        )
        self._terminal = TerminalWidget(
            self.root,
            command=["cmd.exe", "/k", claude_exe],
        )
        self._terminal.pack(fill=tk.BOTH, expand=True)

        foot = tk.Frame(self.root, bg="#181825", height=22)
        foot.pack(fill=tk.X, side=tk.BOTTOM)
        foot.pack_propagate(False)

        self._model_lbl = tk.Label(
            foot,
            text="Warming up …",
            bg="#181825",
            fg="#fab387",
            font=("Segoe UI", 8),
        )
        self._model_lbl.pack(side=tk.LEFT, padx=8, pady=3)

        tk.Label(foot, text=APP_NAME, bg="#181825", fg="#313244", font=("Segoe UI", 8)).pack(
            side=tk.RIGHT, padx=8, pady=3
        )

    def _set_status(self, text: str, color: str = "#45475a"):
        self._status_var.set(text)
        self._status_lbl.config(fg=color)

    # ── Model loading + audio bootstrap ───────────────────────────────────

    def _start_voice_thread(self):
        threading.Thread(target=self._load_models_and_start, daemon=True).start()

    def _load_models_and_start(self):
        try:
            import ctranslate2

            use_gpu = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            use_gpu = False

        if use_gpu:
            cmd_model_size = "base.en"
            cmd_device = "cuda"
            cmd_compute = "float16"
        else:
            cmd_model_size = "small.en"
            cmd_device = "cpu"
            cmd_compute = "int8"

        self.root.after(0, lambda: self._model_lbl.config(text="Warming up …"))
        self.model = WhisperModel(cmd_model_size, device=cmd_device, compute_type=cmd_compute)
        self.root.after(
            0, lambda: self._model_lbl.config(text="Almost ready — tuning ears …", fg="#f9e2af")
        )
        self._wake_model = WhisperModel(WAKE_MODEL_SIZE, device="cpu", compute_type="int8")

        threading.Thread(target=self._record_loop, daemon=True).start()
        keyboard.on_press_key(HOTKEY, self._on_f9_press)
        keyboard.on_press_key(PTT_HOTKEY, self._on_f10_press)
        keyboard.on_press_key(STOP_HOTKEY, self._on_stop_key)

        self._enter_active(ready=True)

    # ── Shared audio stream ────────────────────────────────────────────────

    def _record_loop(self):
        try:
            stream_ctx = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            stream_ctx.__enter__()
        except Exception as exc:
            msg = f"No microphone found: {exc}"
            print(f"[audio] {msg}")
            self.root.after(
                0,
                lambda: (
                    self._set_status("⚠ Can't hear you", "#f38ba8"),
                    self._model_lbl.config(
                        text="Connect a microphone to talk to Ellie", fg="#f38ba8"
                    ),
                ),
            )
            return

        try:
            stream = stream_ctx
            while True:
                chunk, _ = stream.read(512)
                chunk = chunk.copy().flatten()
                with self._lock:
                    self._wake_buf.extend(chunk.tolist())
                    if self._state in ("active", "recording_ptt"):
                        self._frames.append(chunk)
        except Exception as exc:
            print(f"[audio] stream error: {exc}")
            self.root.after(
                0,
                lambda: (
                    self._set_status("⚠ Lost hearing", "#f38ba8"),
                    self._model_lbl.config(
                        text="Ellie lost her hearing — reconnect the mic and restart", fg="#f38ba8"
                    ),
                ),
            )
        finally:
            try:
                stream_ctx.__exit__(None, None, None)
            except Exception:
                pass

    # ── Mode switching ─────────────────────────────────────────────────────

    def _enter_active(self, ready: bool = False):
        """Switch to active listening mode."""
        with self._lock:
            self._cancel_event.set()
            self._state = "active"
            self._frames.clear()
        if ready:
            self.root.after(
                0,
                lambda: (
                    self._set_status(f'◉ Say "{WAKE_WORD_USER}" …', "#cba6f7"),
                    self._model_lbl.config(
                        text=f'Ready  |  say "{WAKE_WORD_USER}" or tap F9', fg="#a6e3a1"
                    ),
                ),
            )
        else:
            self.root.after(0, lambda: self._set_status(f'◉ Say "{WAKE_WORD_USER}" …', "#cba6f7"))
        threading.Thread(target=self._active_loop, daemon=True).start()

    def _enter_ptt(self):
        """Switch to push-to-talk mode."""
        with self._lock:
            self._cancel_event.set()
            self._state = "ptt_idle"
            self._frames.clear()
        _tone_error()
        self.root.after(0, lambda: self._set_status("● Tap F9 to speak", "#45475a"))

    # ── F9 — push-to-talk ──────────────────────────────────────────────────

    def _on_f9_press(self, _):
        with self._lock:
            state = self._state
        if state == "active":
            self._enter_ptt()
        elif state == "ptt_idle":
            with self._lock:
                if self._state != "ptt_idle":
                    return
                self._state = "recording_ptt"
                self._frames.clear()
            _tone_rec_start()
            self.root.after(0, lambda: self._set_status("● Listening …", "#f38ba8"))
            threading.Thread(target=self._ptt_loop, daemon=True).start()

    # ── F10 — back to active listening ─────────────────────────────────────

    def _on_f10_press(self, _):
        with self._lock:
            state = self._state
        if state != "active":
            self._enter_active()

    # ── Escape — stop listening ────────────────────────────────────────────

    def _stop_listening(self):
        """Cancel any active recording and switch to push-to-talk mode."""
        with self._lock:
            if self._state in ("ptt_idle", "transcribing"):
                return
            self._cancel_event.set()
            self._state = "ptt_idle"
            self._frames.clear()
        _tone_error()
        self.root.after(0, lambda: self._set_status("● Tap F9 to speak", "#45475a"))

    def _on_stop_key(self, _):
        with self._lock:
            state = self._state
        if state in ("active", "recording_ptt"):
            self._stop_listening()

    # ── Active listening loop (wake-word gated) ────────────────────────────

    def _active_loop(self):
        """Wait for wake word, then record command. Loops until mode changes."""
        while True:
            # Cooldown after last command
            while time.time() < self._last_sent_at + RESPONSE_COOLDOWN_S:
                time.sleep(0.05)
                with self._lock:
                    if self._state != "active":
                        return

            # Phase 1: wait for wake word
            while True:
                time.sleep(WAKE_STEP_S)
                with self._lock:
                    if self._state != "active":
                        return
                    if self._wake_model is None:
                        continue
                    buf_snapshot = np.array(list(self._wake_buf), dtype=np.float32)

                if len(buf_snapshot) < SAMPLE_RATE:
                    continue

                try:
                    segs, _ = self._wake_model.transcribe(
                        buf_snapshot,
                        beam_size=1,
                        language="en",
                        condition_on_previous_text=False,
                        no_speech_threshold=0.7,
                    )
                    heard = " ".join(s.text for s in segs).lower()
                except Exception:
                    continue

                if WAKE_WORD in heard:
                    break

            with self._lock:
                if self._state != "active":
                    return
                self._frames.clear()

            _tone_wake()
            self.root.after(0, lambda: self._set_status("◉ Yes?", "#cba6f7"))

            # Phase 2: record command until silence
            silence_since: float | None = None
            started = time.time()

            while True:
                time.sleep(0.1)
                with self._lock:
                    if self._state != "active":
                        return
                    frames = list(self._frames)

                if not frames or sum(len(f) for f in frames) / SAMPLE_RATE < 0.5:
                    continue

                recent = np.concatenate(frames)[-int(0.3 * SAMPLE_RATE) :]
                rms = float(np.sqrt(np.mean(recent**2)))

                if rms < SILENCE_RMS:
                    if silence_since is None:
                        silence_since = time.time()
                    elif time.time() - silence_since >= SILENCE_S:
                        break
                else:
                    silence_since = None

                if time.time() - started > 30:
                    break

            with self._lock:
                if self._state != "active" or self._cancel_event.is_set():
                    return
                self._state = "transcribing"
                frames = list(self._frames)
                self._frames.clear()

            _tone_rec_stop()
            self.root.after(0, lambda: self._set_status("◎ Understanding …", "#fab387"))
            threading.Thread(
                target=self._transcribe_and_send, args=(frames, "active"), daemon=True
            ).start()
            return  # _transcribe_and_send re-enters active via _enter_active()

    # ── Push-to-talk recording loop ────────────────────────────────────────

    def _ptt_loop(self):
        """Record until silence, then transcribe. Called on F9 tap."""
        silence_since: float | None = None
        started = time.time()

        while True:
            time.sleep(0.1)
            with self._lock:
                if self._state != "recording_ptt":
                    return
                frames = list(self._frames)

            if not frames:
                continue

            if sum(len(f) for f in frames) / SAMPLE_RATE < 0.5:
                continue

            recent = np.concatenate(frames)[-int(0.3 * SAMPLE_RATE) :]
            rms = float(np.sqrt(np.mean(recent**2)))

            if rms < SILENCE_RMS:
                if silence_since is None:
                    silence_since = time.time()
                elif time.time() - silence_since >= SILENCE_S:
                    break
            else:
                silence_since = None

            if time.time() - started > 30:
                break

        with self._lock:
            if self._state != "recording_ptt" or self._cancel_event.is_set():
                return
            self._state = "transcribing"
            frames = list(self._frames)
            self._frames.clear()

        _tone_rec_stop()
        self.root.after(0, lambda: self._set_status("◎ Understanding …", "#fab387"))
        threading.Thread(
            target=self._transcribe_and_send, args=(frames, "ptt"), daemon=True
        ).start()

    # ── Transcription + send ───────────────────────────────────────────────

    def _transcribe_and_send(self, frames: list[np.ndarray], return_to: str):
        """Transcribe audio and send to terminal. return_to is 'active' or 'ptt'."""
        text: str | None = None

        def _on_timeout():
            with self._lock:
                if self._state == "transcribing":
                    self._state = "ptt_idle"
            self.root.after(0, lambda: self._set_status("● Tap F9 to speak", "#45475a"))

        _timer = threading.Timer(30.0, _on_timeout)
        _timer.daemon = True
        _timer.start()

        own_state = False
        try:
            if self._cancel_event.is_set():
                return
            if not frames or self.model is None:
                return

            audio = np.concatenate(frames).flatten()
            segments, _ = self.model.transcribe(audio, beam_size=5, language="en")

            if self._cancel_event.is_set():
                return

            text = " ".join(s.text.strip() for s in segments).strip()
            if not text:
                return

            if _is_gibberish(text):
                print(f"[transcribe] dropped gibberish: {text!r}")
                return

        except Exception as exc:
            print(f"[transcribe] error: {exc}")
            text = None
        finally:
            _timer.cancel()
            with self._lock:
                cancelled = self._cancel_event.is_set()
                self._cancel_event.clear()
                if self._state == "transcribing":
                    self._state = "ptt_idle"
                    own_state = True

        if cancelled or not own_state:
            return

        if text:
            # Stop-listening voice command — substring match for better recognition
            _norm = re.sub(r"[^a-z ]", "", text.strip().lower())
            if STOP_PHRASE in _norm:
                _tone_error()
                self.root.after(0, lambda: self._set_status("● Tap F9 to speak", "#45475a"))
                return

            _clean = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text.lower())
            _words = _clean.split()
            option_seq = VOICE_OPTION_MAP.get(_clean) or (
                VOICE_OPTION_MAP.get(re.sub(r"[^a-z0-9]", "", _words[0]))
                if len(_words) <= 3 and _words
                else None
            )
            if option_seq is not None:
                label = text.strip()
                self.root.after(0, lambda: self._set_status(f"✓ {label}", "#a6e3a1"))
                if self._terminal:
                    self._terminal.send_raw(option_seq)
            else:
                preview = text[:50] + ("…" if len(text) > 50 else "")
                self.root.after(0, lambda: self._set_status(f"Got it: {preview}", "#89b4fa"))
                if self._terminal:
                    self._terminal.send_text(text)

            _tone_sent()
            self._last_sent_at = time.time()
            if return_to == "active":
                self._enter_active()
            else:
                self.root.after(0, lambda: self._set_status("● Tap F9 to speak", "#45475a"))
        else:
            _tone_error()
            if return_to == "active":
                self._enter_active()
            else:
                self.root.after(0, lambda: self._set_status("● Tap F9 to speak", "#45475a"))

    # ── Startup registration ───────────────────────────────────────────────

    def _register_startup(self):
        script = os.path.abspath(__file__)
        pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        if not os.path.exists(pythonw):
            pythonw = sys.executable
        cmd = f'"{pythonw}" "{script}"'
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE,
            )
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)
            winreg.CloseKey(key)
        except Exception as exc:
            print(f"[startup] Could not register: {exc}")

    # ── Single-instance guard ──────────────────────────────────────────────

    def _ensure_single_instance(self):
        """Exit immediately if another instance of Ellie is already running."""
        self._mutex = ctypes.windll.kernel32.CreateMutexW(None, True, f"Global\\{APP_NAME}Mutex")
        if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            sys.exit(0)

    # ── Cleanup ────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._terminal:
            self._terminal.terminate()
        self.root.destroy()


if __name__ == "__main__":
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass
    EllieApp()
