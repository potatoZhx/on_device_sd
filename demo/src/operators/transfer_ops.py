import torch
from typing import Dict, Optional
import threading
from ..core.types import ExpertID
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TransferManager:
    """
    Manages asynchronous CPU-GPU data transfers.
    Uses CUDA streams for non-blocking transfers.
    """
    
    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream_idx = 0
        
        # Track ongoing transfers
        self.active_transfers: Dict[ExpertID, torch.cuda.Stream] = {}
        self.transfer_lock = threading.Lock()
    
    def transfer_to_gpu_async(
        self,
        expert_id: ExpertID,
        cpu_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Asynchronously transfer expert parameters to GPU.
        
        Args:
            expert_id: Expert identifier
            cpu_params: Parameters on CPU
        
        Returns:
            GPU tensors (transfer may still be in progress)
        """
        # Get a stream
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
        
        with torch.cuda.stream(stream):
            gpu_params = {
                k: v.cuda(non_blocking=True) 
                for k, v in cpu_params.items()
            }
        
        # Track transfer
        with self.transfer_lock:
            self.active_transfers[expert_id] = stream
        
        logger.debug(f"Started async transfer for {expert_id}")
        
        return gpu_params
    
    def wait_for_transfer(self, expert_id: ExpertID) -> None:
        """Wait for a specific transfer to complete"""
        with self.transfer_lock:
            if expert_id in self.active_transfers:
                stream = self.active_transfers[expert_id]
                stream.synchronize()
                del self.active_transfers[expert_id]
                logger.debug(f"Transfer complete for {expert_id}")
    
    def transfer_to_cpu(
        self,
        gpu_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer parameters from GPU to CPU (synchronous).
        
        Args:
            gpu_params: Parameters on GPU
        
        Returns:
            CPU tensors
        """
        return {k: v.cpu() for k, v in gpu_params.items()}
    
    def synchronize_all(self) -> None:
        """Wait for all active transfers to complete"""
        with self.transfer_lock:
            for stream in self.active_transfers.values():
                stream.synchronize()
            self.active_transfers.clear()
        
        logger.debug("All transfers synchronized")