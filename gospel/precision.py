import torch

# List all dtypes we consider supported.
supported_dtypes = {
    torch.float64,
    torch.complex128,
    torch.float32,
    torch.complex64,
    torch.float16,
    torch.complex32,
    torch.bfloat16,
}


def is_complex_dtype(dtype: torch.dtype) -> bool:
    """Returns True if dtype is one of the complex dtypes."""
    return dtype in {torch.complex32, torch.complex64, torch.complex128}


def is_DP(dtype: torch.dtype) -> bool:
    """Returns True if dtype is double precision: float64 or complex128."""
    if dtype is None:
        return None
    if dtype not in supported_dtypes:
        raise RuntimeError(f"[is_DP] Unsupported dtype: {dtype}")
    return dtype in [torch.float64, torch.complex128]


def is_SP(dtype: torch.dtype) -> bool:
    """Returns True if dtype is single precision: float32 or complex64."""
    if dtype is None:
        return None
    if dtype not in supported_dtypes:
        raise RuntimeError(f"[is_SP] Unsupported dtype: {dtype}")
    return dtype in [torch.float32, torch.complex64]


def is_HP(dtype: torch.dtype) -> bool:
    """Returns True if dtype is half precision: float16 or complex32."""
    if dtype is None:
        return None
    if dtype not in supported_dtypes:
        raise RuntimeError(f"[is_HP] Unsupported dtype: {dtype}")
    return dtype in [torch.float16, torch.complex32]


def to_DP(dtype: torch.dtype) -> torch.dtype:
    """
    Converts the given dtype to double precision:
      - float64 if real
      - complex128 if complex
    """
    if dtype is None:
        return None
    if dtype not in supported_dtypes:
        raise RuntimeError(f"[to_DP] Unsupported dtype: {dtype}")
    return torch.complex128 if is_complex_dtype(dtype) else torch.float64


def to_SP(dtype: torch.dtype) -> torch.dtype:
    """
    Converts the given dtype to single precision:
      - float32 if real
      - complex64 if complex
    """
    if dtype is None:
        return None
    if dtype not in supported_dtypes:
        raise RuntimeError(f"[to_SP] Unsupported dtype: {dtype}")
    return torch.complex64 if is_complex_dtype(dtype) else torch.float32


def to_HP(dtype: torch.dtype) -> torch.dtype:
    """
    Converts the given dtype to half precision:
      - float16 if real
      - complex32 if complex
    """
    if dtype is None:
        return None
    if dtype not in supported_dtypes:
        raise RuntimeError(f"[to_HP] Unsupported dtype: {dtype}")
    return torch.complex32 if is_complex_dtype(dtype) else torch.float16


def to_BF16(dtype: torch.dtype) -> torch.dtype:
    """
    Converts the given dtype to bfloat16:
      - bfloat16 if real
      - complex type is not supported
    """
    if dtype is None:
        return None
    if dtype not in supported_dtypes:
        raise RuntimeError(f"[to_BF16] Unsupported dtype: {dtype}")
    if is_complex_dtype(dtype):
        raise RuntimeError("[to_BF16] Complex types are not supported.")
    return torch.bfloat16


def convert_dtype(dtype: torch.dtype, fp: str) -> torch.dtype:
    """
    Converts the given dtype to the specified floating-point precision:
      - 'DP' -> double precision (float64 / complex128)
      - 'SP' -> single precision (float32 / complex64)
      - 'HP' -> half precision   (float16 / complex32)
    """
    if dtype is None:
        return None
    if dtype not in supported_dtypes:
        raise RuntimeError(f"[convert_dtype] Unsupported dtype: {dtype}")

    if fp == "DP":
        return to_DP(dtype)
    elif fp == "SP":
        return to_SP(dtype)
    elif fp == "HP":
        return to_HP(dtype)
    else:
        raise RuntimeError("[convert_dtype] fp must be one of ['DP', 'SP', 'HP'].")
