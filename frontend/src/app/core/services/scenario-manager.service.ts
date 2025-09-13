import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { TestScenario } from '../models/optimization.model';

@Injectable({ providedIn: 'root' })
export class ScenarioManagerService {
  private scenarios: TestScenario[] = [
    {
      id: 'rag-1',
      name: 'RAG Knowledge Base Query',
      description: 'Customer support knowledge base retrieval scenario with high redundancy',
      query: 'how to reset password and recover account',
      category: 'rag',
      chunks: [
        'To reset your password, go to the login page and click "Forgot Password"',
        'Password reset requires email verification for security purposes',
        'Account recovery can be done through email or SMS verification methods',
        'For security reasons, password reset links expire after 24 hours',
        'If you cannot access your email, contact customer support immediately',
        'Two-factor authentication adds extra security to password reset process',
        'Password reset is not available for suspended accounts',
        'To reset your password, go to the login page and click "Forgot Password"', // Exact duplicate
        'Account lockout happens after 5 failed login attempts in a row',
        'Customer support can manually reset passwords for verified accounts',
        'Password reset functionality requires email verification for security',
        'You can reset your password by going to the login page and clicking "Forgot Password"', // Near duplicate
        'Email verification is required for security when resetting passwords',
        'Contact customer support if you cannot access your email address',
        'Password reset links will expire after twenty-four hours for security',
        'Two-factor authentication provides additional security for password resets',
        'Suspended accounts do not have access to password reset features',
        'After five failed login attempts, your account will be locked out',
        'Verified accounts can have passwords manually reset by customer support',
        'The password reset process requires verification through email or SMS',
        'Login page contains a "Forgot Password" link for password resets',
        'Security protocols require password reset links to expire after 24 hours',
        'Customer support assistance is available if email access is unavailable',
        'Additional security is provided by two-factor authentication during reset',
        'Password reset functionality is disabled for accounts that are suspended'
      ],
      expectedReduction: 60
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
    },
    {
      id: 'demo-1',
      name: 'High Redundancy Demo',
      description: 'Extreme redundancy scenario designed to showcase CEGO optimization',
      query: 'database connection troubleshooting',
      category: 'support',
      chunks: [
        'Database connection failed - check network connectivity',
        'Unable to connect to database server - verify connection string',
        'Database connection timeout - increase timeout value',
        'Database connection failed - check network connectivity', // Exact duplicate
        'Connection to database server failed - check network settings',
        'Database server unreachable - verify network connection',
        'Unable to establish database connection - check server status',
        'Database connection failed - check network connectivity', // Another exact duplicate
        'Failed to connect to database - verify credentials',
        'Database connection string is invalid - check configuration',
        'Database connection timeout occurred - adjust timeout settings',
        'Unable to connect to database server - verify connection string', // Exact duplicate
        'Cannot reach database server - check network connectivity',
        'Database authentication failed - verify username and password',
        'Connection to database timed out - increase timeout value',
        'Database server not responding - check server status',
        'Unable to establish database connection - check server status', // Near duplicate
        'Database connection failed - check network connectivity', // Another exact duplicate
        'Failed database connection - verify network settings',
        'Database server connection timeout - adjust timeout configuration',
        'Unable to connect to database - check connection parameters',
        'Database connection string error - verify configuration settings',
        'Cannot establish connection to database server - check network',
        'Database authentication error - verify login credentials',
        'Connection timeout to database server - increase timeout value',
        'Database server unreachable - verify network connection', // Exact duplicate
        'Failed to connect to database server - check connectivity',
        'Database connection failed - check network connectivity', // Yet another exact duplicate
        'Unable to reach database server - verify network settings',
        'Database connection timeout - increase timeout value', // Exact duplicate
        'Cannot connect to database - check server availability',
        'Database authentication failed - verify username and password', // Near duplicate
        'Connection string invalid for database - check configuration',
        'Database server not accessible - verify network connection',
        'Unable to establish database connection - check server status', // Exact duplicate
        'Failed database connection - verify network settings', // Near duplicate
        'Database connection timeout occurred - adjust timeout settings', // Near duplicate
        'Cannot reach database server - check network connectivity', // Near duplicate
        'Database server connection failed - verify settings',
        'Unable to connect to database server - verify connection string' // Exact duplicate
      ],
      expectedReduction: 80
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