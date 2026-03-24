# services/gesture_handler.py

import cv2
import time
from typing import Optional, Deque, List, Dict

from config import AppConfig
from app_state import AppState
from .interfaces import IWebSocketClient


from .fgcs_gesture import GestureRecogniton   


class GestureHandler:
    """
    카메라 입력을 받아 제스처를 인식하고,
    오디오 타임라인과 동기화하여 메타데이터를 서버로 보냅니다.
    """
    def __init__(self, config: AppConfig, state: AppState, ws_client: IWebSocketClient):
        self.config = config
        self.state = state
        self.ws_client = ws_client

        # 디바운스 상태
        self._last_action: Optional[str] = None
        self._last_action_ts: float = 0.0
        self._debounce_sec: float = 0.7   # 동일 제스처 재송신 최소 간격

        if GestureRecogniton is None:
            self.recognizer = None
            self.cap = None
            print("[GestureHandler] 비활성화 상태로 초기화되었습니다.")
            return

        self.recognizer = GestureRecogniton(
            model_path=self.config.GESTURE_MODEL_PATH,
            device=self.config.DEVICE
        )
        self.cap = cv2.VideoCapture(self.config.CAM_INDEX)

        if not self.cap.isOpened():
            print(f"[오류] 카메라({self.config.CAM_INDEX})를 열 수 없습니다.")
            self.cap = None

        # 선택: 초기 프레임 크기 지정으로 지연 안정화
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("[GestureHandler] 초기화 완료.")

    def run(self):
        if self.cap is None:
            print("[GestureHandler] 카메라가 없어 실행할 수 없습니다. 스레드를 종료합니다.")
            return

        try:
            while not self.state.shutdown_event.is_set():
                # 1) 동기화 트리거가 올라왔으면 처리
                if self.state.sync_trigger_event.is_set():
                    self._synchronize_and_send()

                # 2) 녹음 중이 아니면 대기 (CPU 절약)
                if not self.state.is_recording:
                    time.sleep(0.02)
                    continue

                # 3) 프레임 캡처
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)  # 거울 모드

                # 4) 제스처 추론
                self.recognizer.process_frame(frame)
                action = self.recognizer.predict_action()

                # 5) 디바운스 + 큐 적재
                if action:
                    now = time.time()
                    if not self._is_bounced(action, now):
                        self._last_action, self._last_action_ts = action, now
                        # 큐에 thread-safe하게 적재 (append는 보통 원자적이지만, 안전 위해 잠깐 스냅샷/락 고려 가능)
                        self.state.gesture_queue.append({"timestamp": now, "intent": action})
                        print(f"[✋] 제스처 '{action}' 인식 @ {now:.2f}s")

        finally:
            self.cleanup()

    def _is_bounced(self, action: str, ts: float) -> bool:
        """동일 제스처가 너무 짧은 간격으로 반복되는 것을 억제"""
        return (self._last_action == action) and ((ts - self._last_action_ts) < self._debounce_sec)

    def _synchronize_and_send(self):
        """음성 발화 시간과 제스처 시간을 동기화하여 서버에 전송합니다."""
        window = self.config.GESTURE_SYNC_WINDOW_MS / 1000.0

        # 음성 구간 스냅샷
        with self.state.speech_time_lock:
            t_start, t_end = self.state.speech_start_time, self.state.speech_end_time

        # 제스처 큐 스냅샷 (순회 중 변경 방지)
        gestures_snapshot: List[Dict] = list(self.state.gesture_queue)

        # 후보 필터링
        candidates = [g for g in gestures_snapshot
                      if (t_start - window) <= g["timestamp"] <= (t_end + window)]

        if candidates:
            # 시작점에 가장 가까운 제스처 선택
            best = min(candidates, key=lambda g: abs(g["timestamp"] - t_start))
            print(f"[🔄] 동기화 성공: '{best['intent']}' @ {best['timestamp']:.2f}")

            # 서버에 '제스처 메타'만 보냄 (오디오 커밋/response.create는 오디오 파이프라인에서)
            self.ws_client.send({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"[사용자 제스처: {best['intent']}]"}
                    ]
                }
            })
            

        


        # 로컬 상태 정리
        self.state.gesture_queue.clear()
        self.state.sync_trigger_event.clear()
 

    def cleanup(self):
        if self.cap:
            self.cap.release()
        print("[GestureHandler] 카메라 리소스를 해제하고 종료합니다.")
