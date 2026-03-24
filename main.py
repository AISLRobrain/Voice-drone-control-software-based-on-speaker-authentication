# main.py
import threading
import signal

from config import config
from app_state import AppState

from services.websocket_client import WebSocketClient
from services.gesture_handler import GestureHandler
from services.audio_processor import AudioProcessor

# 각 컴포넌트 "클래스" 명시 임포트
from services.audio_components.microphone_stream import MicrophoneStream
from services.audio_components.vad_handler import VADHandler
from services.audio_components.speaker_verifier import SpeakerVerifier
from services.audio_components.wake_word_detector import WakeWordDetector
from services.fuctions import SYSTEM_PROMPT2, INTENT_SCHEMA


def main():
    print("🚀 애플리케이션을 시작합니다.")
    app_state = AppState()

    # --- 의존성 생성 ---
    mic_stream = MicrophoneStream(config)  # 콜백/폴링 구현 중 하나
    vad = VADHandler(config)
    verifier = SpeakerVerifier(config, app_state, vad)
    wake = WakeWordDetector(config)

    ws_client = WebSocketClient(
        config,
        app_state,
        instructions=SYSTEM_PROMPT2,
        tools=[INTENT_SCHEMA],
        # start_audio_on_session_ready 옵션은 쓰지 않고, main에서 스레드를 직접 시작
        # start_audio_on_session_ready=None,
    )

    audio_processor = AudioProcessor(
        config=config,
        state=app_state,
        ws_client=ws_client,
        mic_stream=mic_stream,
        vad_handler=vad,
        speaker_verifier=verifier,
        wake_word_detector=wake,
    )

    gesture_handler = GestureHandler(config, app_state, ws_client)

    # --- 종료 핸들러 ---
    def shutdown_handler(sig, frame):
        if not app_state.shutdown_event.is_set():
            print("\n...시스템을 종료합니다...")
            app_state.shutdown_event.set()
            # 정리 순서는 대체로 입력→네트워크
            try:
                mic_stream.close()
            except Exception:
                pass
            try:
                ws_client.close()
            except Exception:
                pass
            try:
                # Porcupine 사용 시 리소스 해제
                if hasattr(wake, "release"):
                    wake.release()
            except Exception:
                pass

    signal.signal(signal.SIGINT, shutdown_handler)

    # --- 스레드 시작 ---
    ws_thread = threading.Thread(target=ws_client.run, name="WS", daemon=True)
    audio_thread = threading.Thread(target=audio_processor.run, name="Audio", daemon=True)
    gesture_thread = threading.Thread(target=gesture_handler.run, name="Gesture", daemon=True)

    ws_thread.start()
    audio_thread.start()
    gesture_thread.start()

    print("[Main] 스레드 시작. Ctrl+C로 종료하세요.")
    app_state.shutdown_event.wait()

    # (선택) 깔끔한 종료 대기
    ws_thread.join(timeout=2)
    audio_thread.join(timeout=2)
    gesture_thread.join(timeout=2)

    print("✅ 시스템이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    main()
