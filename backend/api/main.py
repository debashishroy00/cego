"""
CEGO FastAPI Main Application.

This module provides the main HTTP API for CEGO optimization services,
following the Docker deployment specification.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import time
import logging
import re

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .pattern_recognition_api import PatternRecognitionAPI
from ..utils.resource_monitor import ResourceMonitor

# Optional entropy features - re-enabled with performance fixes
try:
    from .entropy_api import EntropyAPI
    ENTROPY_AVAILABLE = True
    logger.info("Entropy features enabled with performance optimizations")
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
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:4201",
        "http://127.0.0.1:4201",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "null",  # For local file:// HTML files
        "*"      # Allow all origins for development
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pattern_recognition_api = PatternRecognitionAPI()
entropy_api = EntropyAPI() if ENTROPY_AVAILABLE else None
resource_monitor = ResourceMonitor()


class OptimizationRequest(BaseModel):
    """Request model for optimization endpoint."""
    query: str
    context_pool: List[str]
    max_tokens: Optional[int] = None

    @validator('query')
    def validate_query(cls, v):
        """Validate query input."""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 10000:
            raise ValueError('Query too long (max 10000 characters)')
        # Check for potential injection patterns
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Query contains potentially dangerous content')
        return v.strip()

    @validator('context_pool')
    def validate_context_pool(cls, v):
        """Validate context pool input."""
        if not v:
            raise ValueError('Context pool cannot be empty')
        if len(v) > 1000:
            raise ValueError('Context pool too large (max 1000 items)')

        # Validate each context item
        validated_pool = []
        for i, context in enumerate(v):
            if not isinstance(context, str):
                raise ValueError(f'Context item {i} must be a string')
            if len(context) > 50000:
                raise ValueError(f'Context item {i} too long (max 50000 characters)')
            if context.strip():  # Only add non-empty contexts
                validated_pool.append(context.strip())

        if not validated_pool:
            raise ValueError('No valid context items after validation')
        return validated_pool

    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        """Validate max_tokens parameter."""
        if v is not None:
            if v <= 0:
                raise ValueError('max_tokens must be positive')
            if v > 100000:
                raise ValueError('max_tokens too large (max 100000)')
        return v


class EntropyOptimizationResponse(BaseModel):
    """Standardized response model for entropy optimization endpoint."""
    optimized_context: List[str]
    final_context: List[str]
    original_count: int
    optimized_count: int
    token_reduction_percentage: float
    processing_time_ms: float
    optimization_time_ms: float
    entropy_analysis: Dict[str, Any]
    confidence: float
    method_used: str
    phase_transitions: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    stats: Dict[str, Any]
    semantic_retention: float

    class Config:
        """Allow arbitrary types for complex nested structures."""
        arbitrary_types_allowed = True


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "CEGO API",
        "version": "1.0.0",
        "phase": "Pattern Recognition",
        "status": "active",
        "endpoints": ["/optimize", "/optimize/entropy", "/analyze/entropy", "/health", "/ready", "/metrics"] if ENTROPY_AVAILABLE else ["/optimize", "/health", "/ready", "/metrics"]
    }


@app.post("/optimize")
async def optimize_context(request: OptimizationRequest) -> Dict:
    """
    Optimize context for a given query.

    This is the main optimization endpoint that applies Pattern Recognition
    algorithms to reduce token usage while maintaining relevance.
    """
    import asyncio

    try:
        # Check resource limits
        if not resource_monitor.check_memory():
            raise HTTPException(status_code=503, detail="Memory limit approaching")

        # Apply optimization
        result = pattern_recognition_api.optimize(
            query=request.query,
            context_pool=request.context_pool,
            max_tokens=request.max_tokens
        )

        # Check for errors in optimization
        if 'error' in result.get('stats', {}):
            raise HTTPException(status_code=400, detail=result['stats']['error'])

        return result

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if ENTROPY_AVAILABLE:
    @app.post("/optimize/entropy", response_model=EntropyOptimizationResponse)
    async def optimize_with_entropy(request: OptimizationRequest) -> EntropyOptimizationResponse:
        """
        Advanced optimization using Shannon and multi-dimensional entropy analysis.

        This endpoint uses the improved entropy optimizer with gradient-based selection,
        phase transition detection, and dynamic lambda tuning for superior performance.
        """
        import asyncio

        try:
            # Check resource limits
            if not resource_monitor.check_memory():
                raise HTTPException(status_code=503, detail="Memory limit approaching")

            start_time = time.time()

            # Use the enhanced entropy API for optimization
            result = entropy_api.optimize_with_analysis(
                query=request.query,
                context_pool=request.context_pool,
                max_tokens=request.max_tokens
            )

            # Check for errors in entropy optimization
            if 'error' in result:
                logger.warning(f"Entropy optimization error: {result['error']}")
                raise HTTPException(status_code=500, detail=f"Entropy optimization failed: {result['error']}")

            # Build structured response with explicit field mapping
            return EntropyOptimizationResponse(
                optimized_context=result["optimized_context"],
                final_context=result.get("final_context", result["optimized_context"]),
                original_count=len(request.context_pool),
                optimized_count=len(result["optimized_context"]),
                token_reduction_percentage=result["token_reduction_percentage"],
                processing_time_ms=result["processing_time_ms"],
                optimization_time_ms=result.get("optimization_time_ms", result["processing_time_ms"]),
                entropy_analysis=result["entropy_analysis"],
                confidence=result["confidence"],
                method_used=result["method_used"],
                phase_transitions=result["phase_transitions"],
                metadata=result["metadata"],
                stats=result.get("stats", {}),
                semantic_retention=result.get("semantic_retention", 0.0)
            )

        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Entropy optimization endpoint failed: {e}")
            raise HTTPException(status_code=500, detail=f"Entropy optimization failed: {str(e)}")

    @app.post("/analyze/entropy")
    async def analyze_content_entropy(request: OptimizationRequest) -> Dict:
        """
        Analyze content entropy without performing optimization.

        Provides detailed entropy metrics and optimization recommendations.
        """
        try:
            # Check resource limits
            if not resource_monitor.check_memory():
                raise HTTPException(status_code=503, detail="Memory limit approaching")

            # Perform entropy analysis
            result = entropy_api.analyze_content_entropy(
                query=request.query,
                context_pool=request.context_pool
            )

            return result

        except Exception as e:
            logger.error(f"Entropy analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

else:
    # Fallback endpoints when entropy features are unavailable
    @app.post("/optimize/entropy")
    async def optimize_with_entropy_fallback(request: OptimizationRequest) -> Dict:
        """Entropy optimization fallback - using Pattern Recognition for stability."""
        try:
            # Check resource limits
            if not resource_monitor.check_memory():
                raise HTTPException(status_code=503, detail="Memory limit approaching")

            # Use Pattern Recognition as fallback for entropy optimization
            result = pattern_recognition_api.optimize(
                query=request.query,
                context_pool=request.context_pool,
                max_tokens=request.max_tokens
            )

            # Check for errors in optimization
            if 'error' in result.get('stats', {}):
                raise HTTPException(status_code=400, detail=result['stats']['error'])

            # Add entropy fallback metadata to indicate this is not true entropy optimization
            result.update({
                "method_used": "pattern_recognition_fallback",
                "entropy_fallback_reason": "Entropy features temporarily disabled for performance"
            })

            return result

        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Entropy fallback optimization failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
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
        # Simplified health check to avoid hanging
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "optimizer": "PatternRecognition" if pattern_recognition_api else "None",
            "entropy_available": ENTROPY_AVAILABLE,
            "version": "1.0.0"
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
        test_result = pattern_recognition_api.optimize(
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
                "version": pattern_recognition_api.version,
                "optimizer": pattern_recognition_api.optimizer.algorithm_name,
                "phase": "pattern_recognition"
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)