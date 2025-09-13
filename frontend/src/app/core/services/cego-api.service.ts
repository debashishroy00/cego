import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { OptimizationRequest, OptimizationResult, HealthResponse } from '../models/optimization.model';

@Injectable({
  providedIn: 'root'
})
export class CegoApiService {
  private apiUrl = 'http://localhost:8001';

  constructor(private http: HttpClient) {}

  checkHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.apiUrl}/health`)
      .pipe(catchError(this.handleError));
  }

  optimizeQuickWins(request: OptimizationRequest): Observable<OptimizationResult> {
    const payload = {
      query: request.query,
      context_pool: request.contextPool,
      max_tokens: request.maxTokens
    };

    return this.http.post<OptimizationResult>(`${this.apiUrl}/optimize`, payload)
      .pipe(catchError(this.handleError));
  }

  optimizeEntropy(request: OptimizationRequest): Observable<OptimizationResult> {
    const payload = {
      query: request.query,
      context_pool: request.contextPool,
      max_tokens: request.maxTokens
    };

    return this.http.post<OptimizationResult>(`${this.apiUrl}/optimize/entropy`, payload)
      .pipe(
        catchError((error: HttpErrorResponse) => {
          if (error.status === 503) {
            // Service unavailable - entropy features not available
            return throwError(() => new Error('Entropy optimization not available: ' + error.error?.detail || 'Service unavailable'));
          }
          return this.handleError(error);
        })
      );
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An unknown error occurred';
    
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      if (error.status === 0) {
        errorMessage = 'Unable to connect to CEGO API. Please ensure the backend is running.';
      } else if (error.error?.detail) {
        errorMessage = error.error.detail;
      } else {
        errorMessage = `Error ${error.status}: ${error.message}`;
      }
    }
    
    console.error('CEGO API Error:', errorMessage);
    return throwError(() => new Error(errorMessage));
  }
}