# Adapted from deepseek-ai/FlashMLA(https://github.com/deepseek-ai/FlashMLA)
__version__ = "1.0.1"

__all__ = ["__version__", "get_mla_metadata", "flash_mla_with_kvcache"]


def _load_interface():
    try:
        from flash_mla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata
    except ImportError as exc:
        raise ImportError(
            "flash_mla_cuda is not available. Build and install FlashMLA from source "
            "in a configured MACA environment before calling FlashMLA kernels."
        ) from exc
    return get_mla_metadata, flash_mla_with_kvcache


def get_mla_metadata(*args, **kwargs):
    metadata_func, _ = _load_interface()
    return metadata_func(*args, **kwargs)


def flash_mla_with_kvcache(*args, **kwargs):
    _, flash_func = _load_interface()
    return flash_func(*args, **kwargs)
