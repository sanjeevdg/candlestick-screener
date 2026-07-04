#!/usr/bin/env python3
"""
srt_to_audio.py
Convert an SRT subtitle file to a single audio file using supertonic TTS,
preserving original timings and gaps between subtitles.
"""

import re
import numpy as np
from pathlib import Path
from supertonic import TTS

SAMPLE_RATE = 44100  # supertonic outputs at 44100 Hz


def parse_srt_time(time_str):
    """Parse SRT timestamp 'HH:MM:SS,mmm' to seconds (float)."""
    match = re.match(r"(\d{2}):(\d{2}):{1}(\d{2})[,.](\d{3})", time_str.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: '{time_str}'")
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(filepath):
    """Parse an SRT file and return a list of subtitle entries."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    # Normalize line endings and split into blocks
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", content.strip())

    entries = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Line 0: index number, Line 1: time range, Lines 2+: text
        index = lines[0].strip()
        time_line = lines[1].strip()
        text = "\n".join(lines[2:]).strip()

        # Parse time range
        time_match = re.match(
            r"(.+?)\s*-->\s*(.+)", time_line
        )
        if not time_match:
            continue

        start = parse_srt_time(time_match.group(1))
        end = parse_srt_time(time_match.group(2))

        if text:
            entries.append({
                "index": index,
                "start": start,
                "end": end,
                "text": text,
            })

    return entries


def seconds_to_samples(seconds):
    """Convert seconds to sample count."""
    return int(round(seconds * SAMPLE_RATE))


def assemble_audio(subtitles, wavs):
    """
    Place each synthesized WAV into a silent audio buffer at the correct
    timestamp. Audio that exceeds the subtitle's end time is truncated.
    """
    if not subtitles:
        return np.zeros((1, 0), dtype=np.float32)

    # Total duration = end time of last subtitle
    total_duration = max(sub["end"] for sub in subtitles)
    total_samples = seconds_to_samples(total_duration)

    # Silent audio buffer: shape (1, total_samples)
    audio_buffer = np.zeros((1, total_samples), dtype=np.float32)

    for sub, wav in zip(subtitles, wavs):
        start_sample = seconds_to_samples(sub["start"])
        end_sample = seconds_to_samples(sub["end"])
        max_length = end_sample - start_sample

        # Truncate if synthesized audio exceeds the subtitle window
        audio_segment = wav[:, :max_length] if wav.shape[1] > max_length else wav

        # Place into buffer
        seg_len = audio_segment.shape[1]
        audio_buffer[0, start_sample : start_sample + seg_len] = audio_segment[0]

    return audio_buffer


def srt_to_audio(
    srt_path,
    output_path="output.wav",
    lang="en",
    voice_name="M1",
    speed=1.0,
    total_steps=8,
    hf_token=None,
):
    """
    Convert an SRT file to a single audio file.

    Parameters
    ----------
    srt_path : str
        Path to the input SRT file.
    output_path : str
        Path for the output WAV file.
    lang : str
        Language code for TTS (e.g., "en", "ko", "na").
    voice_name : str
        Voice style name for supertonic.
    speed : float
        Speech speed multiplier (0.7–2.0).
    total_steps : int
        Synthesis quality (5–12).
    hf_token : str, optional
        Hugging Face token for model download.
    """
    srt_path = Path(srt_path)
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    print(f"📄 Parsing SRT: {srt_path}")
    subtitles = parse_srt(srt_path)
    print(f"   Found {len(subtitles)} subtitle entries")

    if not subtitles:
        print("⚠️  No subtitles found. Exiting.")
        return

    # Initialize TTS
    print("🔊 Initializing TTS engine...")
    tts_kwargs = {"auto_download": True}
    if hf_token:
        tts_kwargs["hf_token"] = hf_token

    tts = TTS(**tts_kwargs)
    style = tts.get_voice_style(voice_name=voice_name)

    # Synthesize each subtitle
    wavs = []
    for i, sub in enumerate(subtitles):
        print(
            f"   [{i+1}/{len(subtitles)}] "
            f"{sub['start']:.2f}s → {sub['end']:.2f}s | "
            f"{sub['text'][:60]}{'...' if len(sub['text']) > 60 else ''}"
        )

        wav, duration = tts.synthesize(
            text=sub["text"],
            lang=lang,
            voice_style=style,
            total_steps=total_steps,
            speed=speed,
        )
        wavs.append(wav)

    # Assemble final audio with correct timing
    print("🎵 Assembling final audio...")
    final_audio = assemble_audio(subtitles, wavs)

    # Save
    tts.save_audio(final_audio, output_path)
    total_dur = final_audio.shape[1] / SAMPLE_RATE
    print(f"✅ Saved {total_dur:.2f}s of audio → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SRT subtitles to timed audio using supertonic TTS"
    )
    parser.add_argument("srt", help="Path to input SRT file")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--voice", default="M1", help="Voice style name (default: M1)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed (0.7–2.0)")
    parser.add_argument("--steps", type=int, default=8, help="Quality steps (5–12)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token")

    args = parser.parse_args()

    srt_to_audio(
        srt_path=args.srt,
        output_path=args.output,
        lang=args.lang,
        voice_name=args.voice,
        speed=args.speed,
        total_steps=args.steps,
        hf_token=args.hf_token,
    )

