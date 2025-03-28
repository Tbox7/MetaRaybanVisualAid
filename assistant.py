import cv2
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from msrest.authentication import CognitiveServicesCredentials

__all__ = ["AzureVision", "AzureSpeech"]

class AzureVision:
    def __init__(self, key, endpoint):
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    def extract_text(self, frame):
        _, img_bytes = cv2.imencode(".jpg", frame)
        img_stream = BytesIO(img_bytes.tobytes())
        result = self.client.read_in_stream(img_stream, raw=True)
        operation = result.headers["Operation-Location"].split("/")[-1]
        while True:
            read_result = self.client.get_read_result(operation)
            if read_result.status.lower() not in ["notstarted", "running"]:
                break
        if read_result.status.lower() == "succeeded":
            return " ".join([line.text for r in read_result.analyze_result.read_results for line in r.lines])
        return None


class AzureSpeech:
    def __init__(self, key, region):
        self.config = SpeechConfig(subscription=key, region=region)
        self.synthesizer = SpeechSynthesizer(
            speech_config=self.config,
            audio_config=AudioOutputConfig(use_default_speaker=True)
        )

    def speak(self, text):
        print(f"[TTS] {text}")
        self.synthesizer.speak_text_async(text)