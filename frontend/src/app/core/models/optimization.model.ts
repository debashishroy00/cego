export type EntropyOptimizationResponse = {
  optimized_context: string[];
  final_context: string[];
  original_count?: number;
  optimized_count?: number;
  token_reduction_percentage: number; // 0..1 decimal from backend
  processing_time_ms: number;
  optimization_time_ms: number;
  entropy_analysis: {
    total_entropy: number;
    dimension_entropies: Record<string, number>;
    dimension_weights: Record<string, number>;
    confidence: number;
    metadata: Record<string, unknown>;
  };
  confidence: number;
  method_used: string;
  phase_transitions: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
  stats: Record<string, unknown>;
};

export interface OptimizationRequest {
  query: string;
  contextPool: string[];
  maxTokens: number;
  algorithm: 'pattern-recognition' | 'entropy' | 'both';
}

export interface OptimizationStats {
  original?: {
    pieces: number;
    tokens: number;
  };
  final?: {
    pieces: number;
    tokens: number;
  };
  reduction?: {
    token_reduction_pct: number;
    pieces_saved?: number;
  };
  processing_time_ms?: number;
  algorithm_used?: string;
  phase_transitions?: any[];
}

export interface OptimizationResult {
  optimized_context: string[];
  stats?: OptimizationStats | any;
  optimization_time_ms?: number;
  token_reduction_percentage?: number;
  processing_time_ms?: number;
  confidence_score?: number;
  method_used?: string;
  error?: string;
  entropy_analysis?: {
    dimension_entropies?: { [key: string]: number };
    total_entropy?: number;
  } | null | undefined;
  metadata?: {
    version?: string;
    timestamp?: string;
    rollback_available?: boolean;
    api_version?: string;
    endpoint?: string;
  } | any;
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
  patternRecognition: OptimizationResult | null;
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

export type OptimizeRequest = {
  query: string;
  context_pool: string[];
  max_tokens?: number | null;
};