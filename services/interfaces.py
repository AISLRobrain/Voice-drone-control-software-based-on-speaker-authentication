from abc import ABC, abstractmethod
from typing import Iterator, Mapping, Any, Optional

class IWebSocketClient(ABC):
    @abstractmethod
    def send(self, payload: Mapping[str, Any]) -> None:
        """서버로 메시지를 전송한다. 바이너리는 상위에서 base64 등으로 처리."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """연결을 닫고 자원을 해제한다."""
        raise NotImplementedError

    # 선택: 연결 상태 노출
    def is_connected(self) -> bool:
        return False


class IAudioStream(ABC):
    @abstractmethod
    def listen(self) -> Iterator[bytes]:
        """오디오 청크(PCM16 bytes)를 지속적으로 yield."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """스트림과 관련 자원을 해제."""
        raise NotImplementedError

    # 선택: 컨텍스트 매니저를 강제해도 좋음
    def __enter__(self) -> "IAudioStream":
        return self
    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self.close()
        return None
