"""
Device manager for heterogeneous computing
"""

import torch
import psutil
from typing import Dict, List, Optional
from ..utils.logger import get_logger
from ..utils.config import DeviceConfig


class DeviceManager:
    """Manage CPU and GPU devices for heterogeneous computing"""
    
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.logger = get_logger()
        self.gpu_devices = self._init_gpu_devices()
        self.cpu_device = torch.device("cpu")
        
        self.logger.info(f"Initialized DeviceManager with {len(self.gpu_devices)} GPU devices")
    
    def _init_gpu_devices(self) -> List[torch.device]:
        """Initialize GPU devices"""
        gpu_devices = []
        if torch.cuda.is_available():
            for device_id in self.config.gpu_devices:
                if device_id < torch.cuda.device_count():
                    gpu_devices.append(torch.device(f"cuda:{device_id}"))
                    self.logger.info(f"GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
                else:
                    self.logger.warning(f"GPU {device_id} not available")
        else:
            self.logger.warning("CUDA not available, using CPU only")
        
        return gpu_devices
    
    def get_optimal_device(self, tensor_size: int, compute_intensity: float) -> torch.device:
        """Choose optimal device based on tensor size and compute intensity"""
        if compute_intensity > self.config.gpu_threshold and self._has_gpu_memory(tensor_size):
            return self._select_best_gpu()
        else:
            return self.cpu_device
    
    def _has_gpu_memory(self, required_memory: int) -> bool:
        """Check if GPU has sufficient memory"""
        if not self.gpu_devices:
            return False
        
        available_memory = self.get_gpu_memory_capacity()
        return available_memory >= required_memory
    
    def _select_best_gpu(self) -> torch.device:
        """Select the best GPU based on available memory"""
        if not self.gpu_devices:
            return self.cpu_device
        
        # Simple strategy: select GPU with most available memory
        best_device = self.gpu_devices[0]
        max_memory = 0
        
        for device in self.gpu_devices:
            device_id = device.index
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            available = total - reserved
            
            if available > max_memory:
                max_memory = available
                best_device = device
        
        return best_device
    
    def transfer_tensor(self, tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """Transfer tensor to target device with optimization"""
        if tensor.device == target_device:
            return tensor
        
        # Use non-blocking transfer for GPU
        if target_device.type == "cuda":
            return tensor.to(target_device, non_blocking=True)
        else:
            return tensor.to(target_device)
    
    def get_gpu_memory_capacity(self) -> int:
        """Get total available GPU memory in bytes"""
        if not self.gpu_devices:
            return 0
        
        total_available = 0
        for device in self.gpu_devices:
            device_id = device.index
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            available = total - reserved
            total_available += available
        
        return total_available
    
    def get_cpu_memory_capacity(self) -> int:
        """Get available CPU memory in bytes"""
        memory = psutil.virtual_memory()
        return memory.available
    
    def get_device_memory_usage(self) -> Dict[str, float]:
        """Get memory usage for all devices"""
        usage = {}
        
        # GPU memory usage
        for i, device in enumerate(self.gpu_devices):
            device_id = device.index
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device_id) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
            
            usage[f"gpu_{i}_allocated_gb"] = allocated
            usage[f"gpu_{i}_reserved_gb"] = reserved
            usage[f"gpu_{i}_total_gb"] = total
            usage[f"gpu_{i}_available_gb"] = total - reserved
        
        # CPU memory usage
        memory = psutil.virtual_memory()
        usage["cpu_used_gb"] = memory.used / (1024**3)
        usage["cpu_available_gb"] = memory.available / (1024**3)
        usage["cpu_total_gb"] = memory.total / (1024**3)
        
        return usage
    
    def has_sufficient_gpu_memory(self, required_memory_bytes: int) -> bool:
        """Check if there's sufficient GPU memory"""
        available_memory = self.get_gpu_memory_capacity()
        return available_memory >= required_memory_bytes
    
    def get_expert_memory_usage(self, expert_module: torch.nn.Module) -> int:
        """Estimate memory usage of an expert module"""
        total_params = sum(p.numel() for p in expert_module.parameters())
        # Assume float16 precision
        memory_bytes = total_params * 2  # 2 bytes per parameter for float16
        return memory_bytes
    
    def log_memory_status(self):
        """Log current memory status"""
        usage = self.get_device_memory_usage()
        self.logger.info("Memory Status:")
        for key, value in usage.items():
            self.logger.info(f"  {key}: {value:.2f} GB") 