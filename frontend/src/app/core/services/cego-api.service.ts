import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { EntropyOptimizationResponse, OptimizationResult, HealthResponse, OptimizeRequest } from '../models/optimization.model';

@Injectable({
  providedIn: 'root'
})
export class CegoApiService {
  private apiUrl = 'http://localhost:8003';

  constructor(private http: HttpClient) {}

  checkHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.apiUrl}/health`)
      .pipe(catchError(this.handleError));
  }

  optimizePatternRecognition(request: OptimizeRequest): Observable<OptimizationResult> {
    const payload = {
      query: request.query,
      context_pool: request.context_pool,
      max_tokens: request.max_tokens
    };

    return this.http.post<OptimizationResult>(`${this.apiUrl}/optimize`, payload)
      .pipe(catchError(this.handleError));
  }

  optimizeEntropy(request: OptimizeRequest): Observable<EntropyOptimizationResponse> {
    const payload = {
      query: request.query,
      context_pool: request.context_pool,
      max_tokens: request.max_tokens
    };

    return this.http.post<EntropyOptimizationResponse>(`${this.apiUrl}/optimize/entropy`, payload)
      .pipe(catchError(this.handleError));
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