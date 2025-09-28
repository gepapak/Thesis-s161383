# IEEE Standards Compliance Baseline (Baseline3)

This directory contains the IEEE standards-compliant baseline evaluation system for renewable energy investment systems, structured to match the other baseline approaches.

## Overview

The IEEE baseline system provides comprehensive IEEE standards compliance evaluation and benchmarking capabilities for academic research and publication. This baseline focuses purely on standards compliance evaluation without any optimization algorithms.

### IEEE Standards Evaluated

- **IEEE 1547**: Interconnection and Interoperability of Distributed Energy Resources
- **IEEE 1815**: Electric Power Systems Communications (DNP3)
- **IEEE 2030**: Smart Grid Interoperability Reference Model
- **IEEE 3000**: Power and Energy Standards Collection

### Key Features

- Pure IEEE standards compliance evaluation framework
- NO optimization algorithms, NO machine learning, NO heuristics
- Statistical significance testing for academic publication
- Comprehensive IEEE-compliant performance reporting
- Reproducible evaluation methodologies

## Files Structure

### Core Files (Matching Other Baselines)

- **`ieee_standards_optimizer.py`**: IEEE standards compliance evaluation engine
- **`run_ieee_baseline.py`**: Main runner script for IEEE baseline evaluation

### Legacy Files (Deprecated)

- `ieee_baseline_comparison.py`: Legacy file (replaced by new structure)
- `IEEE_BASELINE_README.md`: Old documentation file

## Usage

### Command Line Usage (Recommended)

```bash
# Run IEEE baseline evaluation
python run_ieee_baseline.py --data_path trainingdata.csv --timesteps 10000

# With custom output directory
python run_ieee_baseline.py --data_path trainingdata.csv --timesteps 5000 --output_dir my_results
```

### Integration with All Baselines

```bash
# Run all three baselines including IEEE
python ../run_all_baselines.py --data_path trainingdata.csv --timesteps 10000
```

### Programmatic Usage

```python
from ieee_standards_optimizer import IEEEStandardsOptimizer, IEEEComplianceConfig

# Initialize IEEE standards optimizer
config = IEEEComplianceConfig()
optimizer = IEEEStandardsOptimizer(config)

# Evaluate single timestep
result = optimizer.step(data_row, timestep)

# Get comprehensive summary
summary = optimizer.get_summary()
```

## IEEE Compliance Evaluation

### IEEE 1547 (Grid Integration Standards)
- Voltage regulation compliance (±5% tolerance)
- Frequency regulation compliance (±0.1 Hz tolerance)
- Power factor compliance (≥95% minimum)
- Grid support metrics evaluation

### IEEE 2030 (Smart Grid Interoperability)
- Interoperability score assessment (≥80% threshold)
- Communication latency evaluation (≤100ms maximum)
- Data integrity verification (≥99% minimum)
- System integration metrics

### IEEE 1815 (Communication Protocol Standards)
- DNP3 compliance testing (≥95% threshold)
- SCADA reliability assessment (≥98% minimum)
- Communication protocol validation
- Data transmission integrity

### IEEE 3000 (Power System Standards)
- Power quality assessment (≥90% threshold)
- Harmonic distortion analysis (≤5% maximum)
- Electrical performance metrics
- System stability evaluation

## Output Structure

Results are saved in the `results/` directory:

```
results/
├── detailed_results.csv              # Timestep-by-timestep compliance data
├── ieee_compliance_analysis.json     # Detailed IEEE standards analysis
├── ieee_compliance_report.png        # Visualization report
└── summary_metrics.json              # Overall compliance summary
```

## Configuration

The system uses `IEEEComplianceConfig` for configuration:

```python
config = IEEEComplianceConfig(
    # IEEE 1547 Grid Integration Standards
    voltage_regulation_tolerance=0.05,      # ±5% voltage regulation
    frequency_regulation_tolerance=0.1,     # ±0.1 Hz frequency regulation
    power_factor_minimum=0.95,              # 95% minimum power factor
    
    # IEEE 2030 Smart Grid Interoperability
    interoperability_score_threshold=0.8,   # 80% interoperability score
    communication_latency_max=100.0,        # 100ms max communication latency
    data_integrity_minimum=0.99,            # 99% data integrity
    
    # IEEE 1815 Communication Protocols
    dnp3_compliance_threshold=0.95,         # 95% DNP3 compliance
    scada_reliability_minimum=0.98,         # 98% SCADA reliability
    
    # IEEE 3000 Power Systems
    power_quality_threshold=0.9,            # 90% power quality score
    harmonic_distortion_max=0.05,           # 5% max harmonic distortion
    
    # Academic evaluation parameters
    statistical_significance_level=0.05,    # p < 0.05
    confidence_interval=0.95                # 95% confidence interval
)
```

## Academic Publication Support

The IEEE baseline provides:

- **Standardized Evaluation**: IEEE-compliant baseline comparison methodologies
- **Statistical Analysis**: Significance testing with p-values and confidence intervals
- **Reproducible Results**: Consistent evaluation framework for peer review
- **Publication-Ready Metrics**: IEEE-standard performance metrics
- **Comprehensive Documentation**: Detailed compliance reporting

## Comparison with Other Baselines

| Aspect | Baseline1 (Traditional) | Baseline2 (Rule-Based) | Baseline3 (IEEE) |
|--------|-------------------------|------------------------|-------------------|
| **Focus** | Classical Portfolio Optimization | Expert System Heuristics | IEEE Standards Compliance |
| **Approach** | Mathematical Optimization | Domain Knowledge Rules | Standards Evaluation |
| **Theory** | Modern Portfolio Theory | Expert Systems | IEEE Standards Framework |
| **Output** | Portfolio Performance | Decision Analysis | Compliance Assessment |
| **Academic Use** | Financial Benchmarking | Heuristic Comparison | Standards Validation |

## Integration with Main Project

The IEEE baseline integrates seamlessly with the main renewable energy investment evaluation framework:

1. **Consistent Structure**: Matches Baseline1 and Baseline2 architecture
2. **Shared Data**: Uses same training data as other baselines
3. **Unified Execution**: Runs through `run_all_baselines.py` script
4. **Comparable Results**: Provides complementary evaluation perspective

## Academic Citation

When using this IEEE baseline system in academic publications:

```
IEEE Standards Compliance Baseline for Renewable Energy Investment Systems
Baseline3_IEEE: IEEE 1547, 1815, 2030, 3000 Standards Evaluation Framework
[Your Institution], [Year]
```

## Notes

- **Pure Evaluation Framework**: NO optimization algorithms, NO machine learning, NO heuristics
- **IEEE Compliance**: All metrics are IEEE-standards compliant for academic publication
- **Statistical Rigor**: Includes significance testing for research validation
- **Reproducible Research**: Results are consistent and peer-review ready
- **Complementary Approach**: Provides standards-based perspective alongside optimization and heuristic baselines
