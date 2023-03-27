from numpy import ndarray


def check_hubert_min_duration(audio: ndarray, sr: int) -> bool:
    return len(audio) / sr >= 0.3
