import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ScenarioSelectorComponent } from './scenario-selector.component';

describe('ScenarioSelectorComponent', () => {
  let component: ScenarioSelectorComponent;
  let fixture: ComponentFixture<ScenarioSelectorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ScenarioSelectorComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ScenarioSelectorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
