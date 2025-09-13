import { TestBed } from '@angular/core/testing';

import { ScenarioManagerService } from './scenario-manager.service';

describe('ScenarioManagerService', () => {
  let service: ScenarioManagerService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ScenarioManagerService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
