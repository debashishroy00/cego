export interface OptimizationRequest {
  query: string;
  contextPool: string[];
  maxTokens: number;
  algorithm: 'quick-wins' | 'entropy' | 'both';
}

export interface OptimizationStats {
  original: {
    pieces: number;
    tokens: number;
  };
  final: {
    pieces: number;
    tokens: number;
  };
  reduction: {
    token_reduction_pct: number;
    pieces_saved: number;
  };
  processing_time_ms: number;
  algorithm_used: string;
  phase_transitions?: any[];
}

export interface OptimizationResult {
  optimized_context: string[];
  stats?: OptimizationStats;
  token_reduction_percentage?: number;
  processing_time_ms?: number;
  confidence_score?: number;
  method_used?: string;
  entropy_analysis?: {
    dimension_entropies: { [key: string]: number };
    total_entropy: number;
  };
  metadata?: {
    version: string;
    timestamp: string;
    rollback_available: boolean;
    api_version: string;
    endpoint: string;
  };
}

export interface TestScenario {
  id: string;
  name: string;
  description: string;
  query: string;
  chunks: string[];
  category: 'rag' | 'support' | 'code' | 'custom';
  expectedReduction?: number;
}

export interface ComparisonResult {
  quickWins: OptimizationResult | null;
  entropy: OptimizationResult | null;
  improvement: number;
  query: string;
  timestamp: Date;
}

export interface HealthResponse {
  status: string;
  timestamp: number;
  memory_usage_mb: number;
  cpu_percent: number;
  optimizer: string;
  version: string;
}