import torch
from collections import deque
import threading


class VADHandler:
    """Silero VAD 기반 음성 활동 감지기 (음성 앞부분/뒷부분 보호)"""
    
    def __init__(self, config):
        self.config = config

        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=False,
        )
        self.model = self.model.to('cpu')
        self.model.eval()
        self.model.reset_states()
        
        # 스레드 안전성을 위한 Lock (멀티스레딩 환경 대비)
        self._lock = threading.Lock()
        
        # 에러 복구 카운터
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
        
        # 음성 앞부분 보호 파라미터
        self._pre_speech_buffer_size = 15  # 약 200-300ms 사전 버퍼
        self._onset_threshold_ratio = 0.7  # 시작 감지용 낮은 임계값 비율
        
        # 음성 끝부분 보호 파라미터
        self._speech_padding_frames = 15  # 약 300-450ms 패딩
        self._min_silence_frames = 20      # 최소 무음 지속 프레임
        
        # 상태 변수
        self._frames_since_speech = 0
        self._is_in_speech = False
        self._first_speech_detected = False
        
        # EMA 스무딩
        self._ema_prob = None  # None으로 초기화하여 첫 프레임 빠른 반응
        self._ema_alpha_normal = 0.15  # 일반 상황
        self._ema_alpha_onset = 0.35   # 음성 시작 감지 시 빠른 반응
        
        # 사전 버퍼 (음성 앞부분 보호용)
        self._probability_buffer = deque(maxlen=self._pre_speech_buffer_size)
        
        print("[VADHandler] Silero VAD 모델 로드 완료 (음성 앞부분/뒷부분 보호 활성화).")
    
    def is_speech(self, chunk: torch.Tensor) -> bool:
        """음성 앞부분/뒷부분 보호 로직이 적용된 음성 감지"""
        with self._lock:  # 스레드 안전성 보장
            try:
                # 입력 검증 및 정규화
                if not isinstance(chunk, torch.Tensor):
                    raise TypeError("입력은 torch.Tensor여야 합니다.")
                if chunk.ndim != 1:
                    chunk = chunk.view(-1)
                if chunk.dtype != torch.float32:
                    chunk = chunk.to(torch.float32)
                if chunk.device.type != "cpu":
                    chunk = chunk.cpu()
                chunk = torch.clamp(chunk, -1.0, 1.0).contiguous()
                
                # VAD 확률 추정 (배치 차원 추가 - 핵심 수정!)
                with torch.no_grad():
                    # unsqueeze(0)로 [samples] -> [1, samples] 변환
                    chunk_2d = chunk.unsqueeze(0)
                    prob = float(self.model(chunk_2d, self.config.RATE_VAD).item())
                
                # 에러 카운터 리셋 (정상 처리됨)
                self._consecutive_errors = 0
                
            except RuntimeError as e:
                # Dimension 관련 에러 또는 상태 손상 시 복구 시도
                if "Dimension out of range" in str(e) or "size mismatch" in str(e):
                    self._consecutive_errors += 1
                    print(f"[VADHandler] VAD 에러 감지 (시도 {self._consecutive_errors}/{self._max_consecutive_errors}): {e}")
                    
                    # 상태 리셋 후 재시도
                    self.model.reset_states()
                    self._ema_prob = None
                    
                    if self._consecutive_errors < self._max_consecutive_errors:
                        try:
                            # 재시도
                            with torch.no_grad():
                                chunk_2d = chunk.unsqueeze(0)
                                prob = float(self.model(chunk_2d, self.config.RATE_VAD).item())
                            print("[VADHandler] VAD 상태 리셋 후 복구 성공")
                            self._consecutive_errors = 0
                        except Exception as retry_error:
                            print(f"[VADHandler] VAD 재시도 실패: {retry_error}")
                            return self._is_in_speech  # 이전 상태 유지
                    else:
                        print("[VADHandler] 최대 에러 횟수 초과. 이전 상태 반환.")
                        return self._is_in_speech
                else:
                    # 다른 종류의 에러는 재발생
                    raise
            
            except Exception as e:
                print(f"[VADHandler] 예상치 못한 에러: {e}")
                return self._is_in_speech  # 안전하게 이전 상태 반환
            
            # EMA 스무딩 (첫 프레임은 직접 할당)
            if self._ema_prob is None:
                self._ema_prob = prob
                smoothed = prob
            else:
                # 음성 시작 감지 시 더 빠른 반응
                alpha = self._ema_alpha_onset if prob > self.config.VAD_THRESHOLD else self._ema_alpha_normal
                self._ema_prob = alpha * prob + (1 - alpha) * self._ema_prob
                smoothed = self._ema_prob
            
            # 확률을 버퍼에 저장 (음성 앞부분 보호용)
            self._probability_buffer.append(smoothed)
            
            # 듀얼 임계값 (히스테리시스)
            onset_threshold = self.config.VAD_THRESHOLD * self._onset_threshold_ratio
            offset_threshold = self.config.VAD_THRESHOLD
            
            # 현재 프레임의 음성 여부 판정
            if not self._is_in_speech:
                # 음성 시작 감지: 낮은 임계값 사용
                current_speech = smoothed > onset_threshold
                
                # 음성 시작이 감지되면 버퍼 내 과거 프레임들도 재검사
                if current_speech:
                    self._is_in_speech = True
                    self._frames_since_speech = 0
                    self._first_speech_detected = True
                    
                    # 버퍼의 과거 프레임들 중 임계값 근처의 프레임들 확인
                    # (음성 앞부분이 잘린 프레임들을 복구)
                    lookback_count = 0
                    for past_prob in reversed(self._probability_buffer):
                        if past_prob > onset_threshold * 0.8:  # 약간 더 낮은 임계값
                            lookback_count += 1
                        else:
                            break
                    
                    return True
            else:
                # 음성 진행 중: 높은 임계값 사용
                current_speech = smoothed > offset_threshold
            
            # 음성 끝부분 보호 로직
            if current_speech:
                self._frames_since_speech = 0
                return True
            else:
                # 음성이 감지되지 않음
                if self._is_in_speech:
                    self._frames_since_speech += 1
                    
                    # 패딩 프레임 범위 내라면 계속 음성으로 처리
                    if self._frames_since_speech <= self._speech_padding_frames:
                        return True
                    
                    # 최소 무음 지속 시간을 초과하면 음성 종료
                    elif self._frames_since_speech > self._min_silence_frames:
                        self._is_in_speech = False
                        return False
                    else:
                        return True
                
                return False
    
    def reset(self):
        """VAD 내부 상태 및 모든 카운터 초기화"""
        with self._lock:
            self.model.reset_states()
            self._ema_prob = None
            self._frames_since_speech = 0
            self._is_in_speech = False
            self._first_speech_detected = False
            self._probability_buffer.clear()
            self._consecutive_errors = 0
            print("[VADHandler] VAD 상태 초기화 완료")
    
    def get_state_info(self) -> dict:
        """현재 VAD 상태 정보 반환 (디버깅용)"""
        with self._lock:
            return {
                "is_in_speech": self._is_in_speech,
                "frames_since_speech": self._frames_since_speech,
                "ema_prob": self._ema_prob,
                "buffer_size": len(self._probability_buffer),
                "first_speech_detected": self._first_speech_detected,
                "consecutive_errors": self._consecutive_errors
            }
