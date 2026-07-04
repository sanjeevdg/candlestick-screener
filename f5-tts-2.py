from f5_tts.api import F5TTS
import soundfile as sf

f5 = F5TTS(
    model="F5TTS_v1_Base",
    device="cpu"  # or cpu
)

wav, sr, _ = f5.infer(
    ref_audio="voice_preview_ranga.wav",
    ref_text="",
    gen_text="AI duniya mein naya kha hai? Aayiye dekhte hai XDash dot A I - joki kaafi saare hindustani bhasha mein soch sakte hain."
)

sf.write("generated2.wav", wav, sr)

print("Saved generated.wav")