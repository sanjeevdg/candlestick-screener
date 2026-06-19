'''
Eleven labs api key
sk_5d89dd3f3023b58a231135bdff3e296caef6ff0c44085dc4
'''

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

client = ElevenLabs(
    api_key="sk_5d89dd3f3023b58a231135bdff3e296caef6ff0c44085dc4"
)

audio = client.text_to_speech.convert(
    text="AI duniya mein naya kha hai? Aayiye dekhte hai XDash dot A I - joki kaafi saare hindustani bhasha mein soch sakte hain",
    voice_id="pzT3Axu7WJzqmpRAWYc5",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)
