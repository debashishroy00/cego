import { TestBed } from '@angular/core/testing';

import { CegoApiService } from './cego-api.service';

describe('CegoApiService', () => {
  let service: CegoApiService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(CegoApiService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
