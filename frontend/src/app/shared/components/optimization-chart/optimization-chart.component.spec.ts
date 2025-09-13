import { ComponentFixture, TestBed } from '@angular/core/testing';

import { OptimizationChartComponent } from './optimization-chart.component';

describe('OptimizationChartComponent', () => {
  let component: OptimizationChartComponent;
  let fixture: ComponentFixture<OptimizationChartComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [OptimizationChartComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(OptimizationChartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
