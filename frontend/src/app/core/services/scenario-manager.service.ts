import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { TestScenario } from '../models/optimization.model';

@Injectable({ providedIn: 'root' })
export class ScenarioManagerService {
  private scenarios: TestScenario[] = [
    {
      id: 'rag-1',
      name: 'RAG Knowledge Base Query',
      description: 'Customer support knowledge base retrieval scenario',
      query: 'how to reset password and recover account',
      category: 'rag',
      chunks: [
        'To reset your password, go to the login page and click "Forgot Password"',
        'Password reset requires email verification for security',
        'Account recovery can be done through email or SMS verification',
        'For security, password reset links expire after 24 hours',
        'If you cannot access your email, contact customer support',
        'Two-factor authentication adds extra security to password reset',
        'Password reset is not available for suspended accounts',
        'To reset your password, go to the login page and click "Forgot Password"', // Duplicate
        'Account lockout happens after 5 failed login attempts',
        'Customer support can manually reset passwords for verified accounts'
      ],
      expectedReduction: 25
    },
    {
      id: 'code-1', 
      name: 'Code Documentation Search',
      description: 'Searching through API documentation and code examples',
      query: 'implement REST API authentication with JWT tokens',
      category: 'code',
      chunks: [
        'JWT (JSON Web Tokens) provide stateless authentication for REST APIs',
        'Authentication headers should include Bearer token for API requests',
        'Token expiration should be set to reasonable timeframes (15-60 minutes)',
        'Refresh tokens allow obtaining new access tokens without re-authentication',
        'API endpoints should validate JWT signatures before processing requests',
        'def authenticate_user(token): return jwt.decode(token, secret_key)',
        'app.use(authenticateToken) middleware validates JWT on protected routes',
        'JWT tokens contain claims about user identity and permissions',
        'Secure token storage prevents XSS and CSRF attacks',
        'Authentication headers should include Bearer token for API requests', // Duplicate
        'Rate limiting prevents brute force attacks on authentication endpoints',
        'HTTPS is required for secure token transmission in production'
      ],
      expectedReduction: 30
    },
    {
      id: 'support-1',
      name: 'Customer Support Tickets',
      description: 'Common customer service scenarios and resolutions',
      query: 'resolve billing issues and payment problems',
      category: 'support',
      chunks: [
        'Billing issues can be resolved by checking payment method status',
        'Failed payments are often due to expired credit cards',
        'Contact billing support for subscription and invoice questions',
        'Refunds are processed within 5-7 business days',
        'Payment failures can cause service interruptions',
        'Update payment information in account settings',
        'Billing disputes should be filed within 30 days',
        'Failed payments are often due to expired credit cards', // Duplicate
        'Proration applies when upgrading or downgrading plans',
        'Tax rates vary by location and are applied automatically',
        'Payment history is available in the billing section',
        'Contact billing support for subscription and invoice questions' // Duplicate
      ],
      expectedReduction: 25
    }
  ];

  private selectedScenarioSubject = new BehaviorSubject<TestScenario | null>(null);
  public selectedScenario$ = this.selectedScenarioSubject.asObservable();

  constructor() {}

  getScenarios(): TestScenario[] {
    return [...this.scenarios];
  }

  getScenariosByCategory(category: string): TestScenario[] {
    return this.scenarios.filter(scenario => scenario.category === category);
  }

  getScenarioById(id: string): TestScenario | undefined {
    return this.scenarios.find(scenario => scenario.id === id);
  }

  selectScenario(scenario: TestScenario): void {
    this.selectedScenarioSubject.next(scenario);
  }

  clearSelection(): void {
    this.selectedScenarioSubject.next(null);
  }
}