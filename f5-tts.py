from pathlib import Path
import subprocess

reference_audio = "voice_preview_ranga.wav"

text = """
AI duniya mein naya kha hai? Aayiye dekhte hai XDash dot A I - joki kaafi saare hindustani bhasha mein soch sakte hain.
"""

result = subprocess.run([
    "f5-tts_infer-cli",
    "--model", "F5TTS_v1_Base",
    "--ref_audio", reference_audio,
    "--ref_text", "",
    "--gen_text", text    
],
    capture_output=True,
    text=True)

print("STDOUT:")
print(result.stdout)

print("STDERR:")
print(result.stderr)