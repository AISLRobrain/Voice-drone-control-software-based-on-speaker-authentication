import time
import threading
import queue
import gc
from collections import deque
from enum import Enum, auto
from typing import Protocol, Iterator, Tuple, NamedTuple, Optional


import numpy as np
import torch
import torchaudio


from .interfaces import IWebSocketClient
from app_state import AppState
from config import AppConfig



# --- 의존성 인터페이스 (기존과 동일) ---
class MicrophoneStream(Protocol):
    def listen(self) -> Iterator[bytes]: ...
    def close(self) -> None: ...



class VADHandler(Protocol):
    def is_speech(self, t16_chunk: torch.Tensor) -> bool: ...
    def reset(self) -> None: ...



class SpeakerVerifierComp(Protocol):
    def process_chunk(self, t16_chunk: torch.Tensor) -> bool: ...
    def is_processing(self) -> bool: ...
    def reset(self) -> None: ...
    def finalize_verification(self) -> bool: ...



class WakeWordDetectorComp(Protocol):
    def process(self, t16_chunk: torch.Tensor) -> bool: ...
    def reset(self) -> None: ...



class AudioState(Enum):
    LISTENING_FOR_WAKE_WORD = auto()
    PROCESSING_COMMAND = auto()



class VerificationResult(NamedTuple):
    is_verified: bool
    timestamp: float



class AudioProcessor:
    _SENTINEL = object()


    def __init__(
        self,
        config: AppConfig,
        state: AppState,
        ws_client: IWebSocketClient,
        mic_stream: MicrophoneStream,
        vad_handler: VADHandler,
        speaker_verifier: SpeakerVerifierComp,
        wake_word_detector: WakeWordDetectorComp,
    ):
        self.config = config
        self.state = state
        self.ws_client = ws_client
        self.mic_stream = mic_stream
        self.vad_handler = vad_handler
        self.speaker_verifier = speaker_verifier
        self.wake_word_detector = wake_word_detector


        # --- 동기화 객체들 ---
        self.state_lock = threading.Lock()
        self.completion_lock = threading.Lock()


        # 추가 락
        self._reset_lock = threading.Lock()          # 리셋 함수 보호
        self._command_state_lock = threading.Lock()  # 공유 변수 보호


        # 이번 라운드 완료 신호
        self.command_completed = threading.Event()
        self.verification_completed = threading.Event()


        # 검증 진행 중 신호
        self._verification_in_progress = threading.Event()


        # 리셋 요청 플래그
        self._reset_requested = threading.Event()


        # 일시정지(명령 처리 루프용)
        self._pause_processing_event = threading.Event()
        self._pause_processing_event.set()


        self.verification_queue: "queue.Queue[object]" = queue.Queue(maxsize=100)
        self.command_queue: "queue.Queue[object]" = queue.Queue(maxsize=100)


        self._verification_worker: Optional[threading.Thread] = None
        self._command_worker: Optional[threading.Thread] = None


        # 리샘플러(CPU 고정)
        self.resampler_to_vad = torchaudio.transforms.Resample(
            self.config.RATE_MIC, self.config.RATE_VAD
        )
        self.resampler_to_send = torchaudio.transforms.Resample(
            self.config.RATE_MIC, self.config.RATE_SEND
        )


        # 상태 변수
        self.current_state = AudioState.LISTENING_FOR_WAKE_WORD
        self._speech_frames = 0
        self._silence_frames = 0
        pre_len = max(1, int(self.config.MIN_SPEECH_FRAMES))
        self._prebuffer: deque[bytes] = deque(maxlen=pre_len)
        self._sending = False
        self._no_speech_frames_after_wake = 0


        # 라운드 관리/중복 방지
        self._round_id = 0                      # 호출어→명령 라운드 식별자
        self._response_sent = False             # response.create 1회 보장
        self._round_start_ts: float = 0.0       # 라운드 시작 시간


    # ------------- 안전 전송 래퍼 -------------
    def _safe_send(self, payload: dict) -> None:
        try:
            self.ws_client.send(payload)
        except Exception as e:
            print(f"[WS] send 예외: {e}")


    # -------------------- 워커 루프 --------------------
    def _verification_worker_loop(self):
        while not self.state.shutdown_event.is_set():
            item = None
            try:
                item = self.verification_queue.get(timeout=0.1)


                if item is self._SENTINEL:
                    self.verification_queue.task_done()
                    break


                # 리셋 요청이면 스킵 및 메모리 정리
                if self._reset_requested.is_set():
                    self._cleanup_queue_item(item)
                    self.verification_queue.task_done()
                    continue


                # item: (t16_chunk,)
                t16_chunk = item[0]


                was_processing_before = self.speaker_verifier.is_processing()
                is_verified = self.speaker_verifier.process_chunk(t16_chunk)
                is_processing_after = self.speaker_verifier.is_processing()


                # [상태 변경 1] 인증이 막 시작되었을 때
                if not was_processing_before and is_processing_after:
                    self.state.verification_status = "IN_PROGRESS"


                if is_processing_after:
                    self._verification_in_progress.set()
                else:
                    self._verification_in_progress.clear()


                now = time.time()
                if is_verified:
                    self.state.verification_status = "FINISHED"
                    result = VerificationResult(is_verified=True, timestamp=now)
                    try:
                        self.state.verification_result_queue.put(result)
                    except Exception as e:
                        print(f"[VerificationWorker] 결과큐 put 예외: {e}")
                    print("[VerificationWorker] ✅ 화자 인증 성공")
                    self.verification_completed.set()
                    self._check_both_completed()
                elif was_processing_before and not is_processing_after:
                    self.state.verification_status = "FINISHED"
                    print("[VerificationWorker] ❌ 화자 인증 실패")
                    self.verification_completed.set()
                    self._check_both_completed()


            except queue.Empty:
                # 라운드 타임아웃 감시(검증 행잉 방지)
                self._maybe_round_timeout()
                continue
            except Exception as e:
                print(f"[VerificationWorker] 예외: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if item is not None and item is not self._SENTINEL:
                    self._cleanup_queue_item(item)
                    self.verification_queue.task_done()


    def _command_worker_loop(self):
        while not self.state.shutdown_event.is_set():
            item = None
            try:
                item = self.command_queue.get(timeout=0.2)
                if item is self._SENTINEL:
                    self.command_queue.task_done()
                    break


                # 리셋 요청이면 스킵 및 메모리 정리
                if self._reset_requested.is_set():
                    self._cleanup_queue_item(item)
                    self.command_queue.task_done()
                    continue


                # item: (t16_chunk, t24_chunk)
                t16_chunk, t24_chunk = item


                with self._command_state_lock:
                    is_speech = self.vad_handler.is_speech(t16_chunk)


                    if is_speech:
                        self._speech_frames += 1
                        self._silence_frames = 0
                        self._no_speech_frames_after_wake = 0


                        if (not self._sending) and (self._speech_frames >= self.config.MIN_SPEECH_FRAMES):
                            self._begin_stream()


                        pcm16_bytes = self._to_pcm16_bytes(t24_chunk)


                        if self._sending:
                            # prebuffer flush
                            while self._prebuffer:
                                self._safe_send({"type": "input_audio_buffer.append", "audio": self._prebuffer.popleft()})
                            self._safe_send({"type": "input_audio_buffer.append", "audio": pcm16_bytes})
                        else:
                            self._prebuffer.append(pcm16_bytes)


                    else:
                        self._silence_frames += 1
                        self._speech_frames = 0


                        # 발화 종료
                        if self._sending and self._silence_frames >= self.config.MIN_SILENCE_FRAMES:
                            if not self._response_sent:
                                print("🛑 발화 종료 감지.")
                                self._safe_send({"type": "input_audio_buffer.commit"})
                                self._safe_send({"type": "response.create"})
                                self._response_sent = True
                            self.command_completed.set()
                            self._check_both_completed()


                        # 웨이크 후 무발화 타임아웃
                        elif (not self._sending) and (self.current_state == AudioState.PROCESSING_COMMAND):
                            self._no_speech_frames_after_wake += 1
                            if not self._verification_in_progress.is_set():
                                timeout_ms = int(self.config.SILENCE_WAKEWORD_S * 1000)
                                if (self._no_speech_frames_after_wake * self.config.CHUNK_MS) >= timeout_ms:
                                    print("⌛ 웨이크 후 무발화 타임아웃 → 호출어 대기 복귀")
                                    self._safe_send({"type": "input_audio_buffer.clear"})
                                    with self.state_lock:
                                        self.current_state = AudioState.LISTENING_FOR_WAKE_WORD
                                    self._reset_command_state()


            except queue.Empty:
                # 라운드 타임아웃 감시
                self._maybe_round_timeout()
                continue
            except Exception as e:
                print(f"[CommandWorker] 루프 예외: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if item is not None and item is not self._SENTINEL:
                    self._cleanup_queue_item(item)
                    self.command_queue.task_done()


    # ------ 메모리 정리 헬퍼 ------
    def _cleanup_queue_item(self, item):
        """큐 아이템의 텐서 메모리를 명시적으로 해제"""
        try:
            if isinstance(item, tuple):
                for tensor in item:
                    if isinstance(tensor, torch.Tensor):
                        del tensor
            del item
        except Exception as e:
            print(f"[AudioProcessor] 큐 아이템 정리 예외: {e}")


    # ------ 라운드 타임아웃(검증 행잉/응답 미도달 보호) ------
    def _maybe_round_timeout(self):
        if self.current_state != AudioState.PROCESSING_COMMAND:
            return
        if self._round_start_ts <= 0:
            return
        # 기본 15초(설정값 없으면) 타임아웃
        timeout_s = getattr(self.config, "ROUND_TIMEOUT_S", 15.0)
        if (time.time() - self._round_start_ts) >= timeout_s:
            print("⏱️ 라운드 타임아웃 → 강제 정리")
            # 아직 전송 중이면 마무리 커밋/응답 1회 보장
            if self._sending and not self._response_sent:
                self._safe_send({"type": "input_audio_buffer.commit"})
                self._safe_send({"type": "response.create"})
                self._response_sent = True
            # 이벤트 정리 후 리셋
            self.command_completed.set()
            self.verification_completed.set()
            self._check_both_completed()


    # -------------------- 공용 유틸 --------------------
    def _check_both_completed(self):
        """두 워커 완료를 원자적으로 확인하고 상태 전이"""
        with self.completion_lock:
            if self.command_completed.is_set() and self.verification_completed.is_set():
                # 상태 전이 및 리셋
                if self.current_state == AudioState.PROCESSING_COMMAND:
                    with self.state_lock:
                        self.current_state = AudioState.LISTENING_FOR_WAKE_WORD
                    self._reset_command_state()
                # 다음 라운드 대비 초기화
                self.command_completed.clear()
                self.verification_completed.clear()
                print("✅ 명령 처리 & 화자 인증 완료 - 호출어 대기로 전환")


    def _start_workers(self):
        if (self._verification_worker is None) or (not self._verification_worker.is_alive()):
            self._verification_worker = threading.Thread(
                target=self._verification_worker_loop, daemon=True, name="VerificationWorker"
            )
            self._verification_worker.start()
        if (self._command_worker is None) or (not self._command_worker.is_alive()):
            self._command_worker = threading.Thread(
                target=self._command_worker_loop, daemon=True, name="CommandWorker"
            )
            self._command_worker.start()


    def _stop_workers(self):
        print("[AudioProcessor] 워커 스레드 종료 중...")
        for q in (self.verification_queue, self.command_queue):
            try:
                q.put(self._SENTINEL, timeout=0.5)
            except queue.Full:
                print("[AudioProcessor] 경고: 종료 신호 전송 실패(큐 Full) - 강제 종료 시도")


        if self._verification_worker and self._verification_worker.is_alive():
            self._verification_worker.join(timeout=2.0)
        if self._command_worker and self._command_worker.is_alive():
            self._command_worker.join(timeout=2.0)


        # 큐 비우기 및 메모리 정리
        for q in (self.verification_queue, self.command_queue):
            try:
                while True:
                    it = q.get_nowait()
                    if it is not self._SENTINEL:
                        self._cleanup_queue_item(it)
                        q.task_done()
            except queue.Empty:
                pass


        # 명시적 GC
        gc.collect()
        print("[AudioProcessor] 워커 스레드 종료 완료.")


    # -------------------- 메인 루프 --------------------
    def run(self):
        print("[AudioProcessor] 시작. 세션 준비 대기 중...")
        self.state.session_ready.wait()
        self._start_workers()
        print("[AudioProcessor] 세션 준비 완료. 오디오 처리 시작.")


        try:
            for pcm48_bytes in self.mic_stream.listen():
                if self.state.shutdown_event.is_set():
                    break


                try:
                    # 버퍼 크기 검증
                    expected_samples = int(self.config.CHUNK_MS * self.config.RATE_MIC / 1000)
                    expected_bytes = expected_samples * 2  # int16 = 2 bytes
                    
                    if len(pcm48_bytes) != expected_bytes:
                        print(f"[AudioProcessor] 경고: 버퍼 크기 불일치 (예상: {expected_bytes}, 실제: {len(pcm48_bytes)})")
                        continue


                    with torch.no_grad():
                        # 명시적 크기 지정 및 복사
                        pcm48_np = (
                            np.frombuffer(pcm48_bytes, dtype=np.int16, count=expected_samples)
                            .astype(np.float32, copy=True) / 32768.0
                        )
                        t48 = torch.from_numpy(pcm48_np)
                        t16 = self.resampler_to_vad(t48.unsqueeze(0)).squeeze(0).clone()
                        
                        # 중간 텐서 명시적 해제
                        del t48, pcm48_np
                        
                except Exception as e:
                    print(f"[AudioProcessor] 전처리 오류: {e}")
                    continue


                with self.state_lock:
                    state = self.current_state


                if state == AudioState.LISTENING_FOR_WAKE_WORD:
                    self.handle_wake_word(t16)
                    del t16
                    continue


                if state == AudioState.PROCESSING_COMMAND:
                    if not self._pause_processing_event.is_set():
                        del t16
                        continue


                    try:
                        with torch.no_grad():
                            # t48 재생성 (이미 해제됨)
                            pcm48_np = (
                                np.frombuffer(pcm48_bytes, dtype=np.int16, count=expected_samples)
                                .astype(np.float32, copy=True) / 32768.0
                            )
                            t48_new = torch.from_numpy(pcm48_np)
                            t24 = self.resampler_to_send(t48_new.unsqueeze(0)).squeeze(0).clone()
                            
                            # 중간 텐서 해제
                            del t48_new, pcm48_np
                        
                        # 메모리 격리를 위해 numpy 경유
                        t16_isolated = torch.from_numpy(t16.numpy().copy())
                        t24_isolated = torch.from_numpy(t24.numpy().copy())
                        
                        # 큐에 넣기
                        self.verification_queue.put_nowait((t16_isolated,))
                        self.command_queue.put_nowait((t16_isolated.clone(), t24_isolated))
                        
                        # 원본 해제
                        del t16, t24, t16_isolated, t24_isolated
                        
                    except queue.Full:
                        print("[AudioProcessor] 경고: 처리 큐가 가득 찼습니다. 프레임 스킵.")
                        del t16
        finally:
            self._stop_workers()
            self.mic_stream.close()
            print("[AudioProcessor] 종료.")


    # -------------------- 상태 전이/헬퍼 --------------------
    def handle_wake_word(self, t16_chunk: torch.Tensor):
        """호출어 감지 처리"""
        try:
            if self.wake_word_detector.process(t16_chunk):
                print("🗣️ 호출어 감지! 명령 녹음을 시작합니다.")
                with self.state_lock:
                    self.current_state = AudioState.PROCESSING_COMMAND
                # 라운드 정보 초기화
                self._round_id += 1
                self._round_start_ts = time.time()
                self._response_sent = False
                # 명령 상태 리셋 후 재개
                self._reset_command_state()
                self._pause_processing_event.set()
        except Exception as e:
            print(f"[AudioProcessor] 호출어 처리 예외: {e}")


    def _begin_stream(self):
        self._sending = True
        with self.state.speech_time_lock:
            self.state.speech_start_time = time.time()
        self.state.is_recording = True
        print("[AudioProcessor] 📤 오디오 전송 시작")


    def _reset_command_state(self):
        """명령 처리 관련 전체 상태 초기화"""
        # 리셋 플래그 설정
        self._reset_requested.set()


        with self._reset_lock:
            with self._command_state_lock:
                self._speech_frames = 0
                self._silence_frames = 0
                self._prebuffer.clear()
                self._sending = False
                self._no_speech_frames_after_wake = 0


            # 이벤트 및 진행상태 초기화
            self.command_completed.clear()
            self.verification_completed.clear()
            self._verification_in_progress.clear()
            self.state.verification_status = "IDLE"
            self._response_sent = False


            # 구성요소 리셋
            try:
                self.vad_handler.reset()
            except Exception as e:
                print(f"[AudioProcessor] VAD 리셋 예외: {e}")


            if getattr(self, "wake_word_detector", None):
                try:
                    self.wake_word_detector.reset()
                except Exception as e:
                    print(f"[AudioProcessor] 호출어 리셋 예외: {e}")


            print("[AudioProcessor] 화자 인증기 상태를 리셋합니다.")
            try:
                self.speaker_verifier.reset()
            except Exception as e:
                print(f"[AudioProcessor] 화자 인증기 리셋 예외: {e}")


            # 두 큐 비우기(남은 항목 task_done 보장 및 메모리 해제)
            for q in (self.verification_queue, self.command_queue):
                cleared = 0
                try:
                    while True:
                        it = q.get_nowait()
                        if it is not self._SENTINEL:
                            self._cleanup_queue_item(it)
                            q.task_done()
                            cleared += 1
                except queue.Empty:
                    pass
                if cleared > 0:
                    print(f"[AudioProcessor] 큐에서 {cleared}개 항목 제거됨")


            # 명시적 GC
            gc.collect()


        # 리셋 플래그 해제
        self._reset_requested.clear()
        print("[AudioProcessor] 상태 리셋 완료")


    # -------------------- 오디오 변환 --------------------
    @staticmethod
    def _to_pcm16_bytes(t: torch.Tensor) -> bytes:
        """
        PyTorch 텐서를 PCM16 바이트로 안전하게 변환
        메모리 손상 방지를 위해 명시적 복사 사용
        """
        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a torch.Tensor")
        
        with torch.no_grad():
            # 텐서 정규화 및 변환
            t = torch.clamp(t, -1.0, 1.0).detach().cpu().contiguous().view(-1)
            x = (t * 32767.0).round().to(torch.int16).numpy()
            
            # 명시적 복사로 메모리 안전성 보장
            x = x.astype('<i2', copy=True)
            result = x.tobytes()
            
            # 명시적 메모리 해제
            del x, t
            
            return result
