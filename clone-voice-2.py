import requests

API_KEY = "16a060daba8dcd21686aea59fcff8342"

url = "https://supertoneapi.com/v1/custom-voices/cloned-voice"

files = {
    "files": (
        "voice_preview_ranga.wav",
        open("voice_preview_ranga.wav", "rb"),
        "audio/wav"
    )
}
# open("voice_preview_ranga.wav", "rb")
data = {
    "name": "My Custom Voice",
    "description": "Voice cloned from sample"
}

headers = {
    "x-sup-api-key": API_KEY
}

response = requests.post(
    url,
    headers=headers,
    files=files,
    data=data
)

print(response.json())