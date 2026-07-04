#!/usr/bin/env python3
"""
srt_to_audio.py

Convert an SRT subtitle file into a single audio file using Supertonic TTS,
preserving the original timing gaps between subtitle entries.

Usage:
    python srt_to_audio.py subtitles.srt output.wav [options]

Requirements:
    pip install supertonic srt numpy
"""

import argparse
import numpy as np
#import srt
from datetime import timedelta
from pathlib import Path

from supertonic import TTS


def seconds_from_timedelta(td: timedelta) -> float:
    """Convert a timedelta to total seconds as a float."""
    return td.total_seconds()


def generate_silence(duration_seconds: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a silence (zeros) array of the given duration."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros((1, num_samples), dtype=np.float32)


def stretch_audio_to_duration(wav: np.ndarray, target_duration: float, sample_rate: int = 44100) -> np.ndarray:
    """
    If the generated audio is longer than the subtitle's display duration,
    truncate it. If shorter, it will end before the next subtitle starts
    (the gap/silence logic handles the rest).
    """
    target_samples = int(target_duration * sample_rate)
    if wav.shape[1] > target_samples:
        return wav[:, :target_samples]
    return wav


def srt_to_audio(
    srt_path: str,
    output_path: str,
    lang: str = "en",
    voice_name: str = "M1",
    total_steps: int = 8,
    speed: float = 1.05,
    gap_padding: float = 0.0,
    truncate_to_subtitle_duration: bool = False,
):
    """
    Convert an SRT file to a single audio file.

    Args:
        srt_path:          Path to the input .srt file.
        output_path:       Path to the output .wav file.
        lang:              Language code for TTS (e.g., "en", "ko", "na").
        voice_name:        Voice style name for Supertonic TTS.
        total_steps:       TTS quality (5=low to 12=high).
        speed:             TTS speed multiplier (0.7=slow to 2.0=fast).
        gap_padding:       Extra silence (seconds) appended after each subtitle's audio.
        truncate_to_subtitle_duration:
                           If True, truncate generated audio to not exceed the subtitle's
                           on-screen duration.
    """
    SAMPLE_RATE = 44100

    # --- Parse SRT ---
    srt_text = Path(srt_path).read_text(encoding="utf-8")
    subtitles = list(srt.parse(srt_text))

    if not subtitles:
        raise ValueError(f"No subtitles found in '{srt_path}'.")

    print(f"Parsed {len(subtitles)} subtitle entries from '{srt_path}'.")

    # --- Initialize TTS ---
    print("Initializing Supertonic TTS (first run downloads the model)...")
    tts = TTS(auto_download=True)
    voice_style = tts.get_voice_style(voice_name=voice_name)
    print(f"Using voice style: '{voice_name}', lang: '{lang}', steps: {total_steps}, speed: {speed}")

    # --- Build audio segments ---
    audio_segments = []

    # Silence from t=0 to the start of the first subtitle
    first_start = seconds_from_timedelta(subtitles[0].start)
    if first_start > 0:
        audio_segments.append(generate_silence(first_start, SAMPLE_RATE))

    for i, sub in enumerate(subtitles):
        start_sec = seconds_from_timedelta(sub.start)
        end_sec = seconds_from_timedelta(sub.end)
        subtitle_duration = end_sec - start_sec
        text = sub.content.strip().replace("\n", " ")

        if not text:
            # Empty subtitle — just add silence for its duration
            audio_segments.append(generate_silence(subtitle_duration, SAMPLE_RATE))
            continue

        print(f"  [{i+1}/{len(subtitles)}] {start_sec:.2f}s – {end_sec:.2f}s | \"{text[:60]}{'...' if len(text) > 60 else ''}\"")

        # Synthesize speech
        wav, duration = tts.synthesize(
            text=text,
            lang=lang,
            voice_style=voice_style,
            total_steps=total_steps,
            speed=speed,
        )

        # Optionally truncate to the subtitle's on-screen duration
        if truncate_to_subtitle_duration:
            wav = stretch_audio_to_duration(wav, subtitle_duration, SAMPLE_RATE)

        audio_segments.append(wav)

        # Determine gap to next subtitle
        if i < len(subtitles) - 1:
            next_start = seconds_from_timedelta(subtitles[i + 1].start)
            generated_duration = duration[0]
            gap = next_start - (start_sec + generated_duration)
            if gap_padding > 0:
                gap = max(gap, gap_padding)
            if gap > 0:
                audio_segments.append(generate_silence(gap, SAMPLE_RATE))
            # If gap < 0 (overlap), we just concatenate — audio will overlap naturally

    # --- Concatenate all segments ---
    full_audio = np.concatenate(audio_segments, axis=1)  # shape: (1, total_samples)
    total_duration = full_audio.shape[1] / SAMPLE_RATE

    print(f"\nTotal audio duration: {total_duration:.2f}s")
    print(f"Saving to '{output_path}'...")

    tts.save_audio(full_audio, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an SRT subtitle file to audio using Supertonic TTS, preserving timing."
    )
    parser.add_argument("srt_path", help="Path to the input .srt file.")
    parser.add_argument("output_path", help="Path to the output .wav file.")
    parser.add_argument("--lang", default="en", help="Language code (default: en).")
    parser.add_argument("--voice-name", default="M1", help="Supertonic voice style name (default: M1).")
    parser.add_argument("--total-steps", type=int, default=8, help="TTS quality 5–12 (default: 8).")
    parser.add_argument("--speed", type=float, default=1.05, help="TTS speed 0.7–2.0 (default: 1.05).")
    parser.add_argument("--gap-padding", type=float, default=0.0, help="Min silence gap in seconds after each entry (default: 0).")
    parser.add_argument(
        "--truncate-to-duration", action="store_true",
        help="Truncate TTS audio if it exceeds the subtitle's on-screen duration."
    )

    args = parser.parse_args()

    srt_to_audio(
        srt_path=args.srt_path,
        output_path=args.output_path,
        lang=args.lang,
        voice_name=args.voice_name,
        total_steps=args.total_steps,
        speed=args.speed,
        gap_padding=args.gap_padding,
        truncate_to_subtitle_duration=args.truncate_to_duration,
    )


if __name__ == "__main__":
    main()

