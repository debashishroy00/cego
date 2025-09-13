"""
Resource monitoring utilities for container deployment.

This module provides resource monitoring and enforcement capabilities
to ensure CEGO stays within Docker container limits.
"""

import os
import logging
from typing import Dict, Optional

# Handle Windows compatibility - resource module is Unix-only
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    resource = None
    HAS_RESOURCE = False

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Monitor and enforce resource limits in container environment.
    
    Tracks memory and CPU usage to prevent container OOM kills
    and ensure stable operation within Docker resource constraints.
    """
    
    def __init__(self):
        self.memory_limit_mb = int(os.getenv('MEMORY_LIMIT_MB', '1024'))
        self.cpu_limit_cores = int(os.getenv('CPU_LIMIT_CORES', '2'))
        self.memory_warning_threshold = 0.85  # 85%
        self.memory_critical_threshold = 0.95  # 95%
        
        self._set_resource_limits()
        logger.info(f"Resource monitor initialized: {self.memory_limit_mb}MB memory limit")
    
    def _set_resource_limits(self):
        """
        Set resource limits for the container process.
        
        This helps prevent runaway processes from consuming all
        container resources and triggering OOM kills.
        """
        if not HAS_RESOURCE:
            logger.warning("Resource module not available (Windows compatibility mode)")
            return
            
        try:
            # Set memory limit (soft and hard)
            memory_bytes = self.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # Optional: Set CPU time limit (uncomment if needed)
            # cpu_seconds = 300  # 5 minutes per request
            # resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            
        except (OSError, Exception) as e:
            logger.warning(f"Could not set resource limits: {e}")
    
    def check_memory(self) -> bool:
        """
        Check if memory usage is within safe limits.
        
        Returns:
            True if memory usage is safe, False if approaching limits
        """
        try:
            import psutil
            current_mb = psutil.Process().memory_info().rss / 1024 / 1024
            usage_ratio = current_mb / self.memory_limit_mb
            
            if usage_ratio >= self.memory_critical_threshold:
                logger.critical(f"Memory usage critical: {current_mb:.0f}MB / {self.memory_limit_mb}MB")
                return False
            elif usage_ratio >= self.memory_warning_threshold:
                logger.warning(f"Memory usage high: {current_mb:.0f}MB / {self.memory_limit_mb}MB")
            
            return usage_ratio < self.memory_critical_threshold
            
        except ImportError:
            logger.warning("psutil not available, cannot monitor memory")
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return True  # Fail safe
    
    def get_resource_stats(self) -> Dict:
        """
        Get current resource usage statistics.
        
        Returns:
            Dictionary with current resource usage metrics
        """
        try:
            import psutil
            process = psutil.Process()
            
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            
            return {
                'memory': {
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'limit_mb': self.memory_limit_mb,
                    'usage_percent': (memory_info.rss / 1024 / 1024) / self.memory_limit_mb * 100
                },
                'cpu': {
                    'percent': cpu_percent,
                    'limit_cores': self.cpu_limit_cores
                },
                'process': {
                    'pid': process.pid,
                    'threads': process.num_threads(),
                    'open_files': len(process.open_files())
                }
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def trigger_gc_if_needed(self):
        """
        Trigger garbage collection if memory usage is high.
        
        This can help prevent OOM situations by freeing unused objects.
        """
        try:
            import psutil
            import gc
            
            current_mb = psutil.Process().memory_info().rss / 1024 / 1024
            usage_ratio = current_mb / self.memory_limit_mb
            
            if usage_ratio >= self.memory_warning_threshold:
                logger.info("Triggering garbage collection due to high memory usage")
                collected = gc.collect()
                logger.info(f"Garbage collection freed {collected} objects")
                
                # Check memory after GC
                new_mb = psutil.Process().memory_info().rss / 1024 / 1024
                freed_mb = current_mb - new_mb
                logger.info(f"Memory freed: {freed_mb:.1f}MB")
                
        except ImportError:
            pass  # psutil not available
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
    
    def enforce_limits(self) -> bool:
        """
        Enforce resource limits and take corrective action if needed.
        
        Returns:
            True if within limits, False if limits exceeded
        """
        if not self.check_memory():
            self.trigger_gc_if_needed()
            
            # Check again after GC
            if not self.check_memory():
                logger.critical("Memory limit exceeded even after garbage collection")
                return False
        
        return True


# Global instance for easy access
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """
    Get global resource monitor instance.
    
    Returns:
        Singleton ResourceMonitor instance
    """
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor