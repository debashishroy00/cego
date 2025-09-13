import { Component, OnInit, OnDestroy } from '@angular/core';
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
import { Subject, takeUntil, forkJoin } from 'rxjs';
import { CegoApiService } from '../../core/services/cego-api.service';
import { ScenarioManagerService } from '../../core/services/scenario-manager.service';
import { OptimizationRequest, OptimizationResult, TestScenario, ComparisonResult } from '../../core/models/optimization.model';

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
  styleUrl: './dashboard.component.scss'
})
export class DashboardComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  // Input fields
  query = '';
  inputText = '';
  maxTokens = 2000;

  // Results
  quickWinsResult: OptimizationResult | null = null;
  entropyResult: OptimizationResult | null = null;
  comparisonResult: ComparisonResult | null = null;
  
  // UI state
  loading = false;
  error: string | null = null;
  apiConnected = false;

  // Scenarios
  scenarios: TestScenario[] = [];
  selectedScenario: TestScenario | null = null;

  constructor(
    private cegoApi: CegoApiService,
    private scenarioManager: ScenarioManagerService
  ) {}

  ngOnInit(): void {
    this.scenarios = this.scenarioManager.getScenarios();
    this.checkApiConnection();
    
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

  checkApiConnection(): void {
    this.cegoApi.checkHealth()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (health) => {
          this.apiConnected = health.status === 'healthy';
        },
        error: (error) => {
          this.apiConnected = false;
          console.warn('API connection check failed:', error);
        }
      });
  }

  loadScenario(scenario: TestScenario): void {
    this.selectedScenario = scenario;
    this.query = scenario.query;
    this.inputText = scenario.chunks.join('\n');
    this.clearResults();
  }

  clearResults(): void {
    this.quickWinsResult = null;
    this.entropyResult = null;
    this.comparisonResult = null;
    this.error = null;
  }

  runOptimization(): void {
    if (!this.query || !this.inputText) {
      this.error = 'Please enter both query and context content';
      return;
    }

    this.loading = true;
    this.error = null;
    this.clearResults();

    const chunks = this.inputText.split('\n').filter(line => line.trim());
    const request: OptimizationRequest = {
      query: this.query,
      contextPool: chunks,
      maxTokens: this.maxTokens,
      algorithm: 'both'
    };

    // Run both optimizations in parallel
    forkJoin({
      quickWins: this.cegoApi.optimizeQuickWins(request),
      entropy: this.cegoApi.optimizeEntropy(request)
    }).pipe(
      takeUntil(this.destroy$)
    ).subscribe({
      next: (results) => {
        this.quickWinsResult = results.quickWins;
        this.entropyResult = results.entropy;
        this.createComparisonResult();
        this.loading = false;
      },
      error: (error) => {
        this.error = error.message;
        this.loading = false;
        
        // Try individual optimizations as fallback
        this.runFallbackOptimization(request);
      }
    });
  }

  private runFallbackOptimization(request: OptimizationRequest): void {
    // Try Quick Wins first
    this.cegoApi.optimizeQuickWins(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          this.quickWinsResult = result;
          this.tryEntropyOptimization(request);
        },
        error: (error) => {
          this.error = `Quick Wins failed: ${error.message}`;
        }
      });
  }

  private tryEntropyOptimization(request: OptimizationRequest): void {
    this.cegoApi.optimizeEntropy(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          this.entropyResult = result;
          this.createComparisonResult();
        },
        error: (error) => {
          console.warn('Entropy optimization failed:', error.message);
          // Keep Quick Wins result, show partial comparison
          this.createComparisonResult();
        }
      });
  }

  private createComparisonResult(): void {
    if (!this.quickWinsResult && !this.entropyResult) return;

    const quickWinsReduction = this.quickWinsResult?.stats?.reduction?.token_reduction_pct || 0;
    const entropyReduction = this.entropyResult?.token_reduction_percentage || 
                           this.entropyResult?.stats?.reduction?.token_reduction_pct || 0;

    this.comparisonResult = {
      quickWins: this.quickWinsResult,
      entropy: this.entropyResult,
      improvement: entropyReduction - quickWinsReduction,
      query: this.query,
      timestamp: new Date()
    };
  }

  getReductionPercentage(result: OptimizationResult | null): number {
    if (!result) return 0;
    return result.token_reduction_percentage || 
           result.stats?.reduction?.token_reduction_pct || 0;
  }

  getProcessingTime(result: OptimizationResult | null): number {
    if (!result) return 0;
    return result.processing_time_ms || 
           result.stats?.processing_time_ms || 0;
  }

  getOriginalTokens(result: OptimizationResult | null): number {
    if (!result) return 0;
    return result.stats?.original?.tokens || 0;
  }

  getFinalTokens(result: OptimizationResult | null): number {
    if (!result) return 0;
    return result.stats?.final?.tokens || 0;
  }

  calculateCostSavings(reductionPct: number, originalTokens: number): number {
    const costPerToken = 0.002; // Example: $0.002 per 1K tokens
    const tokensSaved = originalTokens * (reductionPct / 100);
    return (tokensSaved * costPerToken) / 1000;
  }

  getScenariosByCategory(category: string): TestScenario[] {
    return this.scenarioManager.getScenariosByCategory(category);
  }

  // Helper for template
  Object = Object;

  exportResults(): void {
    if (!this.comparisonResult) return;

    const exportData = {
      scenario: this.selectedScenario?.name || 'Custom Query',
      query: this.comparisonResult.query,
      timestamp: this.comparisonResult.timestamp,
      quickWins: {
        reduction: this.getReductionPercentage(this.comparisonResult.quickWins),
        processingTime: this.getProcessingTime(this.comparisonResult.quickWins),
        originalTokens: this.getOriginalTokens(this.comparisonResult.quickWins),
        finalTokens: this.getFinalTokens(this.comparisonResult.quickWins)
      },
      entropy: {
        reduction: this.getReductionPercentage(this.comparisonResult.entropy),
        processingTime: this.getProcessingTime(this.comparisonResult.entropy),
        originalTokens: this.getOriginalTokens(this.comparisonResult.entropy),
        finalTokens: this.getFinalTokens(this.comparisonResult.entropy)
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
}