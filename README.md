# Neural ODE for TBI — Module 2.6

Mechanistic Neural ODE models for traumatic brain injury patient trajectory prediction,
integrated with hard physiological constraints and the EvidenceOS CCR decision-support stack.

## Repository Structure

```
neural-ode-tbi/
├── ode_systems/          # Mechanistic ODE right-hand-side definitions
│   ├── icp_dynamics.py
│   ├── cerebral_autoregulation.py
│   ├── biomarker_kinetics.py
│   └── multi_compartment.py
├── models/               # Neural ODE model wrappers
│   ├── hybrid_ode.py     # Physics-informed hybrid Neural ODE
│   └── latent_ode.py     # Latent ODE baseline comparator
├── constraints/          # Hard physiological constraint enforcement
│   ├── monro_kellie.py
│   ├── cerebral_perfusion.py
│   └── physiological_bounds.py
├── evaluation/           # Metrics, calibration, safety compliance
│   ├── trajectory_metrics.py
│   ├── event_prediction.py
│   └── constraint_compliance.py
├── data/                 # Data loading and preprocessing
│   └── tbi_dataset.py
├── experiments/          # Experiment configs and runners
│   └── train_hybrid.py
├── tests/                # Unit and integration tests
│   ├── test_ode_systems.py
│   ├── test_constraints.py
│   └── test_models.py
├── requirements.txt
└── README.md
```

## Key Design Principles

1. **Mechanistic priors**: ODE systems encode known TBI physiology (Monro-Kellie, cerebral autoregulation, biomarker kinetics).
2. **Hard constraints**: Physiological bounds are enforced as projection layers, not soft penalties.
3. **Hybrid architecture**: Neural networks augment mechanistic terms, not replace them.
4. **Clinical safety**: Constraint compliance is a first-class evaluation metric alongside trajectory accuracy.
5. **NINDS alignment**: Temporal phases (acute/subacute/initial-outcome/late-effects) are explicit in model design.

## Canonical Thresholds

| Biomarker | CT-Indication Threshold | Source |
|-----------|------------------------|--------|
| GFAP      | ≥30 pg/mL              | Abbott i-STAT / NINDS |
| UCH-L1    | ≥360 pg/mL             | Abbott i-STAT / NINDS |

| Physiological Parameter | Critical Threshold | Action |
|------------------------|-------------------|--------|
| ICP                    | >22 mmHg          | Treatment escalation |
| CPP                    | <60 mmHg          | Vasopressor/volume |
| PbtO2                  | <20 mmHg          | Oxygenation optimization |
| PaCO2                  | <25 mmHg          | Reduce hyperventilation |
| SBP                    | <90 mmHg          | Fluid resuscitation |

## Quick Start

```bash
pip install -r requirements.txt
python experiments/train_hybrid.py --config experiments/default_config.yaml
```

## Integration with EvidenceOS Platform

Model outputs feed into:
- `ccr-reason` edge function (candidate generation)
- `ccr-decision-support` (rule-based disposition)
- `run-bridge-model` (BRIDGE-TBI inference with physiological validation)

All via the shared `threshold-engine.ts` and `tbi-constraint-validator.ts` contracts.
