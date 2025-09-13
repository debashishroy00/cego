import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TestingLabComponent } from './testing-lab.component';

describe('TestingLabComponent', () => {
  let component: TestingLabComponent;
  let fixture: ComponentFixture<TestingLabComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestingLabComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TestingLabComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
