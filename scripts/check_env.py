import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_torch():
    try:
        torch_imported = True
        torch_version = torch.__version__
    except ImportError:
        torch_imported = False
        torch_version = None
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        tf32_supported = torch.cuda.get_device_capability(0)[0] >= 8 # Ampere GPUs (sm_80) and newer support TF32
        bf16_supported = torch.cuda.is_bf16_supported()
    else:
        cuda_version = None
        gpu_count = 0
        tf32_supported = False
        bf16_supported = False
    
    return {
        "torch_imported": torch_imported,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_count": gpu_count,
        "tf32_supported": tf32_supported,
        "bf16_supported": bf16_supported
    }

if __name__ == "__main__":
    result = check_torch()
    for key, value in result.items():
        logger.info(f"{key}: {value}")
