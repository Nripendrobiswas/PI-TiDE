# Contributing to PI-TiDE

Thank you for your interest in contributing to the Physics-Informed Time-Series Dense Encoder (PI-TiDE) project!

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker
- Provide a clear title and description
- Include steps to reproduce
- Specify your environment (Python version, PyTorch version, OS)

### Submitting Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Format code: `black src/ experiments/`
6. Lint: `flake8 src/ experiments/`
7. Submit PR with clear description

### Code Style
- Follow PEP 8
- Use `black` for formatting (line length 88)
- Type hints encouraged for new code
- Docstrings in Google/NumPy style

### Adding New Physics Constraints
1. Add the constraint function in `src/pitide/loss.py`
2. Register in `physics_informed_loss()` with a `lambda_*` weight
3. Add corresponding violation rate in `compute_violation_rates()`
4. Update `DegreeDayPhysics` if new physics parameters needed
5. Add ablation config in `experiments/run_ablation.py`

### Adding New Datasets
1. Add loader in `src/pitide/data.py`
2. Ensure columns: `DateTime`, `demand`, `temp`, covariates
3. Add preprocessing (outlier removal, time features)
4. Test with both baseline and physics modes

## Development Setup

```bash
git clone https://github.com/yourusername/PI-TiDE.git
cd PI-TiDE
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_pitide.py -v

# With coverage
pytest --cov=pitide tests/
```

## Architecture Overview

```
src/pitide/
├── __init__.py       # Public API
├── pitide.py         # PITiDE model (ResidualBlock, ResidualStack, PITiDE)
├── physics.py        # DegreeDayPhysics, balance temp estimation
├── loss.py           # Physics-informed loss functions
├── training.py       # Training loop with AMP + dynamic weighting
├── data.py           # Data loading and preprocessing
├── evaluation.py     # Metrics, violation analysis, ablation
└── hypertune.py      # Grid search hyperparameter tuning
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.