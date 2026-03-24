# config.py (권장 리팩터)
import os
import torch
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    # API 및 모델 설정
    MODEL_ID: str = "gpt-realtime-2025-08-28"
    API_KEY: str | None = os.getenv("API_KEY")
    URL: str = ""
    HEADERS: list[str] = None  # __post_init__에서 채움

    # 장치 설정
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIC_INDEX: int = 3
    CAM_INDEX: int = 4

    # 오디오 처리 설정
    RATE_MIC: int = 48000
    RATE_VAD: int = 16000
    RATE_SEND: int = 24000
    CHUNK_MS: int = 32
    CHUNK_MIC: int = 0  # __post_init__에서 계산

    # VAD 및 음성 관련 설정
    VAD_THRESHOLD: float = 0.35
    MIN_SPEECH_FRAMES: int = 3
    MIN_SILENCE_FRAMES: int = 40
    PREBUFFER_CHUNKS: int = 15
    SILENCE_TIMEOUT_S: float = 1.5
    SILENCE_WAKEWORD_S:float=4

    # 화자 인증 설정
    VERIFICATION_THRESHOLD: float = 0.6
    MIN_VERIFICATION_SECONDS: float = 1.2
    AUTHORIZED_SPEAKER_WAV_PATH: str = "20_man.wav"
    AUTHORIZED_SPEAKER_WAV_PATH_2: str = "30_man.wav"
    AUTHORIZED_SPEAKER_WAV_PATH_3: str = "20_woman.wav"
    AUTHORIZED_SPEAKER_WAV_PATH_4: str = "30_woman.wav"

    # 호출어(Wake Word) 설정
    PICOVOICE_ACCESS_KEY: str | None = os.getenv("PICOVOICE_ACCESS_KEY")
    KEYWORD_PATHS: list[str] = None
    PORCUPINE_MODEL_PATH: str = "/home/joongi/Desktop/gesture/porcupine_params_ko.pv"

    # 제스처 설정
    GESTURE_MODEL_PATH: str = "./hmi_FGCS.pt"
    GESTURE_SYNC_WINDOW_MS: int = 1000

    # 프롬프트/스키마는 **지연 바인딩** (순환 import 방지)
    INTENT_SCHEMA: object | None = None
    SYSTEM_PROMPT2: object | None = None

    def __post_init__(self):
        self.URL = f"wss://api.openai.com/v1/realtime?model={self.MODEL_ID}"
        self.HEADERS = [f"Authorization: Bearer {self.API_KEY}", "OpenAI-Beta: realtime=v1"]
        self.CHUNK_MIC = self.RATE_MIC * self.CHUNK_MS // 1000
        # 기본 키워드 경로 (None이면 빈 리스트)
        if self.KEYWORD_PATHS is None:
            self.KEYWORD_PATHS = ["/home/joongi/Downloads/율지스_ko_linux_v3_0_0/율지스_ko_linux_v3_0_0.ppn"]

config = AppConfig()
