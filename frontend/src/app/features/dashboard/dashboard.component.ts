import { Component, OnInit, OnDestroy, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatChipsModule } from '@angular/material/chips';
import { MatDividerModule } from '@angular/material/divider';
import { Subject, takeUntil, of, forkJoin } from 'rxjs';
import { catchError, timeout, finalize } from 'rxjs/operators';
import { CegoApiService } from '../../core/services/cego-api.service';
import { ScenarioManagerService } from '../../core/services/scenario-manager.service';
import { EntropyOptimizationResponse, OptimizationResult, TestScenario, ComparisonResult, OptimizeRequest } from '../../core/models/optimization.model';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatToolbarModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatProgressBarModule,
    MatIconModule,
    MatSelectModule,
    MatChipsModule,
    MatDividerModule
  ],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class DashboardComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  // Input fields
  query = '';
  inputText = '';
  maxTokens = 2000;

  // Results
  patternRecognitionResult?: OptimizationResult;
  entropyResult?: EntropyOptimizationResponse;
  comparisonResult?: ComparisonResult;

  // UI state
  loading = false;
  errorMsg = '';
  backendUp = false;

  // Scenarios
  scenarios: TestScenario[] = [];
  selectedScenario?: TestScenario;

  constructor(
    private cegoApi: CegoApiService,
    private scenarioManager: ScenarioManagerService
  ) {}

  ngOnInit(): void {
    this.scenarios = this.scenarioManager.getScenarios();
    this.checkBackendHealth();

    // Subscribe to scenario selection
    this.scenarioManager.selectedScenario$
      .pipe(takeUntil(this.destroy$))
      .subscribe(scenario => {
        if (scenario) {
          this.loadScenario(scenario);
        }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private checkBackendHealth(): void {
    this.cegoApi.checkHealth()
      .pipe(
        timeout(3000),
        catchError(() => of(null)),
        takeUntil(this.destroy$)
      )
      .subscribe(response => {
        this.backendUp = !!response;
      });
  }

  loadScenario(scenario: TestScenario): void {
    this.selectedScenario = scenario;
    this.query = scenario.query;
    this.inputText = scenario.chunks.join('\\n');
    this.clearResults();
  }

  clearResults(): void {
    this.patternRecognitionResult = undefined;
    this.entropyResult = undefined;
    this.comparisonResult = undefined;
    this.errorMsg = '';
  }

  private buildRequest(): OptimizeRequest {
    const contextPool = (this.inputText || '')
      .split(/\\r?\\n/)
      .map(s => s.trim())
      .filter(Boolean); // one chunk per line

    return {
      query: this.query?.trim() || '',
      context_pool: contextPool,
      max_tokens: this.maxTokens || null
    };
  }

  runOptimization(): void {
    this.loading = true;
    this.errorMsg = '';
    this.clearResults();

    // Build request: split textarea into array of lines
    const contextPool = (this.inputText || '')
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(Boolean);

    const req = {
      query: this.query?.trim() || '',
      context_pool: contextPool,
      max_tokens: this.maxTokens ?? null,
    };

    const pattern$ = this.cegoApi.optimizePatternRecognition(req).pipe(
      timeout(15000),
      catchError(err => {
        console.warn('Pattern Recognition failed:', err);
        return of(null);
      })
    );

    const entropy$ = this.cegoApi.optimizeEntropy(req).pipe(
      timeout(15000),
      catchError(err => {
        console.warn('Entropy optimization failed:', err);
        return of(null);
      })
    );

    // Run both, but don't let one block the other
    forkJoin({ pattern: pattern$, entropy: entropy$ })
      .pipe(
        finalize(() => {
          this.loading = false;
          // Create comparison if we have at least one result
          if (this.patternRecognitionResult || this.entropyResult) {
            this.createComparisonResult();
          } else if (!this.patternRecognitionResult && !this.entropyResult) {
            this.errorMsg = 'Both optimizers failed or timed out. Please check the backend logs.';
          }
        }),
        takeUntil(this.destroy$)
      )
      .subscribe(({ pattern, entropy }) => {
        if (pattern) {
          this.patternRecognitionResult = pattern;
        }
        if (entropy) {
          this.entropyResult = entropy;
        }
      });
  }

  private createComparisonResult(): void {
    if (!this.patternRecognitionResult && !this.entropyResult) return;

    const quickWinsReduction = this.patternRecognitionResult?.token_reduction_percentage || 0;
    const entropyReduction = this.entropyResult?.token_reduction_percentage || 0;

    this.comparisonResult = {
      patternRecognition: this.patternRecognitionResult || null,
      entropy: this.entropyResult as OptimizationResult || null,
      improvement: entropyReduction - quickWinsReduction,
      query: this.query,
      timestamp: new Date()
    };
  }

  private extractErrorMessage(error: unknown): string {
    return this.extractError(error);
  }

  private extractError(err: any): string {
    return (
      err?.error?.metadata?.error ||
      err?.error?.detail ||
      err?.message ||
      'Unknown error'
    );
  }

  // Strict-safe getters
  get entropyPct(): number {
    return Math.round(100 * (this.entropyResult?.token_reduction_percentage ?? 0));
  }

  get patternPct(): number {
    return Math.round(100 * (this.patternRecognitionResult?.token_reduction_percentage ?? 0));
  }

  get entropyConfidence(): number {
    const confidence = this.entropyResult?.confidence ??
                     this.entropyResult?.entropy_analysis?.confidence ?? 0;
    return Math.max(0, Math.min(1, confidence));
  }

  get entropyChunks(): string[] {
    return this.entropyResult?.optimized_context ?? [];
  }

  get patternChunks(): string[] {
    return this.patternRecognitionResult?.optimized_context ?? [];
  }

  get hasEntropyError(): boolean {
    return !!(this.entropyResult?.metadata?.['error'] as string);
  }

  get entropyErrorMsg(): string {
    return (this.entropyResult?.metadata?.['error'] as string) || '';
  }

  get hasPatternError(): boolean {
    return !!(this.patternRecognitionResult?.error);
  }

  get patternErrorMsg(): string {
    return this.patternRecognitionResult?.error || '';
  }

  get entropyDimensionEntries(): [string, number][] {
    if (!this.entropyResult?.entropy_analysis?.dimension_entropies) {
      return [];
    }
    return Object.entries(this.entropyResult.entropy_analysis.dimension_entropies) as [string, number][];
  }

  calculateCostSavings(reductionPct: number, originalTokens: number): number {
    const costPerToken = 0.002; // Example: $0.002 per 1K tokens
    const tokensSaved = originalTokens * (reductionPct / 100);
    return (tokensSaved * costPerToken) / 1000;
  }

  abs(value: number): number {
    return Math.abs(value);
  }

  getScenariosByCategory(category: string): TestScenario[] {
    return this.scenarioManager.getScenariosByCategory(category);
  }

  exportResults(): void {
    if (!this.comparisonResult) return;

    const exportData = {
      scenario: this.selectedScenario?.name || 'Custom Query',
      query: this.comparisonResult.query,
      timestamp: this.comparisonResult.timestamp,
      patternRecognition: {
        reduction: this.patternPct,
        processingTime: this.patternRecognitionResult?.processing_time_ms || 0,
        originalTokens: 0, // Will need to extract from stats if available
        finalTokens: 0     // Will need to extract from stats if available
      },
      entropy: {
        reduction: this.entropyPct,
        processingTime: this.entropyResult?.processing_time_ms || 0,
        originalTokens: this.entropyResult?.original_count || 0,
        finalTokens: this.entropyResult?.optimized_count || 0
      },
      improvement: this.comparisonResult.improvement
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `cego-results-${Date.now()}.json`;
    link.click();
    window.URL.revokeObjectURL(url);
  }

  // TrackBy function for ngFor performance
  trackByIndex = (_: number, __: unknown) => _;

  // Helper for template
  Object = Object;
}