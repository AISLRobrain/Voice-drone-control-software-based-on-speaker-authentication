# services/websocket_client.py
import json
import websocket
import threading
import base64
import queue  # queue.Empty 예외 처리를 위해 임포트
from typing import Optional, Callable

from .interfaces import IWebSocketClient

class WebSocketClient(IWebSocketClient):
    def __init__(self, config, state, *,
                 instructions=None, tools=None, tool_choice="required",
                 turn_detection=None, input_audio_format="pcm16",
                 start_audio_on_session_ready: Optional[Callable[[], None]] = None):
        """
        instructions/tools를 '주입' 받아 순환 의존을 피함.
        세션 준비 후 오디오 스레드를 시작하려면 start_audio_on_session_ready 콜백을 넘겨도 됨.
        """
        self.config = config
        self.state = state
        self.ws: Optional[websocket.WebSocketApp] = None
        self._start_audio_cb = start_audio_on_session_ready

        # 세션 설정(없으면 config의 지연 바인딩 필드 사용)
        self.instructions = instructions if instructions is not None else getattr(config, "SYSTEM_PROMPT2", None)
        self.tools = tools if tools is not None else [getattr(config, "INTENT_SCHEMA", None)]
        self.tool_choice = tool_choice
        self.turn_detection = turn_detection
        self.input_audio_format = input_audio_format

    def send(self, payload: dict):
        # 오디오 바이트는 base64로 인코딩
        if "audio" in payload and isinstance(payload["audio"], (bytes, bytearray)):
            payload["audio"] = base64.b64encode(payload["audio"]).decode("utf-8")

        try:
            if self.ws and self.ws.sock and self.ws.sock.connected:
                data = json.dumps(payload, ensure_ascii=False)
                with self.state.ws_send_lock:
                    self.ws.send(data)  # 텍스트 프레임
        except Exception as e:
            print(f"[WS] Send 실패: {e}")

    def close(self):
        try:
            if self.ws and self.ws.sock and self.ws.sock.connected:
                self.ws.close()
        except Exception:
            pass

    def run(self):
        self.ws = websocket.WebSocketApp(
            self.config.URL,
            header=self.config.HEADERS,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        # 핑/재연결 옵션(필요시 값 조정)
        self.ws.run_forever(
            ping_interval=20,
            ping_timeout=10,
            ping_payload="ping",
            reconnect=5
        )

    # --- 콜백들 ---
    def _on_open(self, ws):
        print("✅ WebSocket 연결 성공")

    def _on_error(self, ws, e):
        print(f"❌ WebSocket 오류: {e}")

    def _on_close(self, ws, *a):
        print("🔒 WebSocket 연결 종료")

    def _on_message(self, ws, raw):
        try:
            evt = json.loads(raw)
            t = evt.get("type", "")

            if t == "session.created":
                session_cfg = {
                    "instructions": self.instructions,
                    "tools": self.tools,
                    "tool_choice": self.tool_choice,
                    "turn_detection": self.turn_detection,
                    "input_audio_format": self.input_audio_format
                }
                self.send({"type": "session.update", "session": session_cfg})
                self.send({"type": "input_audio_buffer.clear"})

            elif t == "session.updated":
                print("✅ OpenAI 세션 준비 완료.")
                self.state.session_ready.set()
                if self._start_audio_cb:
                    threading.Thread(target=self._start_audio_cb, daemon=True).start()

            # 🔽🔽🔽 --- 수정된 부분 --- 🔽🔽🔽
            elif t == "response.done":
                self.send({"type": "input_audio_buffer.clear"})
                output = evt.get("response", {}).get("output", [])

                if not (output and output[0].get("type") == "function_call"):
                    print(f"[🤖] 함수 호출이 아닌 응답: {t}")
                    return

                try:
                    # 1. 성공 결과를 기다립니다. (timeout 시간 내에 결과가 없으면 실패로 간주)
                    # ❗️ 이 timeout 값이 실질적인 '인증 대기 시간'이 됩니다.
                    result = self.state.verification_result_queue.get(timeout=10.0)

                    # 2. 큐에서 무언가 성공적으로 꺼내졌다면, '인증 성공'으로 간주합니다.
                    if result:
                        args_str = output[0].get("arguments", "{}")
                        try:
                            data = json.loads(args_str)
                            print("\n" + "="*40, "\n✅ 인증 성공 및 함수 호출:\n",
                                  json.dumps(data, indent=2, ensure_ascii=False),
                                  "\n" + "="*40)
                            # 여기에 실제 함수를 호출하는 로직 추가

                        except json.JSONDecodeError:
                            print(f"❌ 함수 인자 JSON 파싱 실패: {args_str}")

                except queue.Empty:
                    status = getattr(self.state, "verification_status", "UNKNOWN")
                    if status == "FINISHED":
                        print("⚠️ 화자 인증에 실패하여 작업을 취소합니다.")
                    elif status=="UNKNOWN":
                        print("⏰ 화자 인증 시스템이 응답하지 않거나 시간 초과되었습니다. 작업을 취소합니다.")

                except Exception as e:
                    # 4. 기타 예외 처리
                    print(f"❌ 함수 호출 처리 중 예외 발생: {e}")
            # 🔼🔼🔼 --- 수정된 부분 --- 🔼🔼🔼

        except Exception as e:
            print(f"❌ on_message 예외: {e} (Raw: {raw})")