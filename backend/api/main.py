"""
CEGO FastAPI Main Application.

This module provides the main HTTP API for CEGO optimization services,
following the Docker deployment specification.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import time
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .quick_wins_api import QuickWinsAPI
from ..utils.resource_monitor import ResourceMonitor

# Optional entropy features - graceful fallback if dependencies missing
try:
    from .entropy_api import EntropyAPI
    ENTROPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Entropy features unavailable: {e}")
    ENTROPY_AVAILABLE = False
    EntropyAPI = None

# Initialize FastAPI app
app = FastAPI(
    title="CEGO API", 
    description="Context Entropy Gradient Optimization Service",
    version="1.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "http://localhost:4201", "http://127.0.0.1:4201"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
quick_wins_api = QuickWinsAPI()
entropy_api = EntropyAPI() if ENTROPY_AVAILABLE else None
resource_monitor = ResourceMonitor()


class OptimizationRequest(BaseModel):
    """Request model for optimization endpoint."""
    query: str
    context_pool: List[str]
    max_tokens: Optional[int] = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "CEGO API",
        "version": "1.0.0",
        "phase": "Quick Wins",
        "status": "active",
        "endpoints": ["/optimize", "/optimize/entropy", "/analyze/entropy", "/health", "/ready", "/metrics"] if ENTROPY_AVAILABLE else ["/optimize", "/health", "/ready", "/metrics"]
    }


@app.post("/optimize")
async def optimize_context(request: OptimizationRequest) -> Dict:
    """
    Optimize context for a given query.
    
    This is the main optimization endpoint that applies Quick Wins
    algorithms to reduce token usage while maintaining relevance.
    """
    try:
        # Check resource limits (temporarily disabled for debugging)
        # if not resource_monitor.check_memory():
        #     raise HTTPException(status_code=503, detail="Memory limit approaching")
        
        # Apply optimization
        result = quick_wins_api.optimize(
            query=request.query,
            context_pool=request.context_pool,
            max_tokens=request.max_tokens
        )
        
        # Check for errors in optimization
        if 'error' in result.get('stats', {}):
            raise HTTPException(status_code=400, detail=result['stats']['error'])
        
        return result
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if ENTROPY_AVAILABLE:
    @app.post("/optimize/entropy")
    async def optimize_with_entropy(request: OptimizationRequest) -> Dict:
        """
        Advanced optimization using Shannon and multi-dimensional entropy analysis.
        
        This endpoint applies the Week 2 MVP algorithms including:
        - Shannon entropy-guided content selection
        - Multi-dimensional entropy analysis
        - Gradient descent optimization
        - Phase transition detection
        """
        try:
            # Check resource limits (temporarily disabled for debugging)
            # if not resource_monitor.check_memory():
            #     raise HTTPException(status_code=503, detail="Memory limit approaching")
            
            # Apply entropy-based optimization
            result = entropy_api.optimize_with_analysis(
                query=request.query,
                context_pool=request.context_pool,
                max_tokens=request.max_tokens
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Entropy optimization failed: {e}")
            # Fallback to Quick Wins
            try:
                fallback_result = quick_wins_api.optimize(
                    query=request.query,
                    context_pool=request.context_pool,
                    max_tokens=request.max_tokens
                )
                fallback_result['fallback_used'] = True
                fallback_result['fallback_reason'] = str(e)
                return fallback_result
            except Exception as fallback_error:
                raise HTTPException(status_code=500, detail=f"Both entropy and fallback failed: {fallback_error}")

    @app.post("/analyze/entropy")
    async def analyze_entropy(request: OptimizationRequest) -> Dict:
        """
        Analyze content entropy without optimization.
        
        Provides detailed entropy analysis including:
        - Shannon entropy calculations
        - Multi-dimensional entropy breakdown
        - Content distribution analysis
        - Optimization recommendations
        """
        try:
            # Check resource limits (temporarily disabled for debugging)
            # if not resource_monitor.check_memory():
            #     raise HTTPException(status_code=503, detail="Memory limit approaching")
            
            # Perform entropy analysis only
            result = entropy_api.analyze_content_entropy(
                query=request.query,
                context_pool=request.context_pool
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Entropy analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
else:
    # Placeholder endpoints when entropy features are unavailable
    @app.post("/optimize/entropy")
    async def optimize_with_entropy_unavailable(request: OptimizationRequest) -> Dict:
        """Entropy optimization unavailable - missing numpy/scipy dependencies."""
        raise HTTPException(
            status_code=503, 
            detail="Entropy features unavailable. Install numpy/scipy dependencies and restart."
        )
    
    @app.post("/analyze/entropy") 
    async def analyze_entropy_unavailable(request: OptimizationRequest) -> Dict:
        """Entropy analysis unavailable - missing numpy/scipy dependencies."""
        raise HTTPException(
            status_code=503,
            detail="Entropy features unavailable. Install numpy/scipy dependencies and restart."
        )


@app.get("/health")
async def health_check() -> Dict:
    """
    Docker health check endpoint.
    
    Returns basic health status for container health checks.
    """
    try:
        import psutil
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "optimizer": quick_wins_api.optimizer.algorithm_name,
            "version": quick_wins_api.version
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/ready")
async def readiness_check() -> Dict:
    """
    Kubernetes readiness probe.
    
    Verifies that the service is ready to handle requests.
    """
    try:
        # Test optimization to ensure everything is loaded
        test_result = quick_wins_api.optimize(
            query="test",
            context_pool=["test context"],
            max_tokens=100
        )
        
        if 'error' in test_result.get('stats', {}):
            return {"status": "not_ready", "error": test_result['stats']['error']}
        
        return {
            "status": "ready",
            "optimizer": "loaded",
            "test_reduction": test_result['stats']['reduction']['token_reduction_pct']
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready", "error": str(e)}


@app.get("/metrics")
async def get_metrics() -> Dict:
    """
    Prometheus-compatible metrics endpoint.
    
    Provides operational metrics for monitoring.
    """
    try:
        import psutil
        
        # Basic system metrics
        process = psutil.Process()
        
        return {
            "cego_memory_usage_bytes": process.memory_info().rss,
            "cego_cpu_usage_percent": process.cpu_percent(),
            "cego_uptime_seconds": time.time() - process.create_time(),
            "cego_version_info": {
                "version": quick_wins_api.version,
                "optimizer": quick_wins_api.optimizer.algorithm_name,
                "phase": "quick_wins"
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)