# Contributing to US Traffic Accident Forecasting

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the US Traffic Accident Forecasting project.

## Getting Started

1. Fork the repository to your GitHub account
2. Clone your fork locally
   ```bash
   git clone https://github.com/YOUR_USERNAME/us-traffic-accident-forecasting.git
   cd us-traffic-accident-forecasting
   ```
3. Set up the development environment
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

### 1. Create a Feature Branch

Always create a new branch for your work:

```bash
git checkout -b feature/AmazingFeature
```

Branch naming conventions:
- `feature/` - for new features
- `bugfix/` - for bug fixes
- `docs/` - for documentation improvements
- `refactor/` - for code refactoring

### 2. Make Your Changes

- Write clean, readable code
- Follow Python PEP 8 style guidelines
- Add comments for complex logic
- Update documentation as needed

### 3. Test Your Changes

Before committing, ensure your changes work correctly:
- Test any new model implementations
- Verify data processing pipelines
- Check that visualizations render properly
- Test the Streamlit dashboard if you made UI changes

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description of changes"
```

Good commit message examples:
- "Add GRU hyperparameter optimization with Optuna"
- "Fix data preprocessing bug in state-level aggregation"
- "Update dashboard to include ablation study results"

### 5. Push to Your Fork

```bash
git push origin feature/AmazingFeature
```

### 6. Open a Pull Request

1. Go to the original repository
2. Click "New Pull Request"
3. Select your fork and branch
4. Provide a clear description of your changes
5. Reference any related issues

## What to Contribute

We welcome contributions in the following areas:

### Model Improvements
- New deep learning architectures
- Hyperparameter optimization experiments
- Feature engineering techniques
- Model ensemble methods

### Data Analysis
- Additional exploratory data analysis
- New visualization techniques
- Statistical analysis improvements
- Data quality enhancements

### Dashboard Features
- New interactive visualizations
- Performance metrics displays
- User interface improvements
- Additional filtering options

### Documentation
- Code documentation improvements
- Tutorial additions
- README enhancements
- Example notebooks

### Bug Fixes
- Data processing issues
- Model training bugs
- Dashboard errors
- Performance optimizations

## Code Standards

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Keep functions focused and modular
- Add docstrings to functions and classes

### Model Development
- Use consistent random seeds for reproducibility
- Document model architectures clearly
- Include performance metrics in model evaluations
- Save model checkpoints appropriately

### Data Processing
- Validate data integrity
- Handle missing values appropriately
- Document data transformations
- Use efficient pandas/numpy operations

## Pull Request Guidelines

Your pull request should:
- Have a clear title and description
- Reference related issues (if applicable)
- Include tests or validation results
- Update relevant documentation
- Not break existing functionality

## Questions or Issues?

If you have questions or encounter issues:
- Check existing issues first
- Open a new issue with a clear description
- Provide reproducible examples when reporting bugs
- Include relevant error messages and logs

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help others learn
- Focus on what is best for the project
- Show empathy towards other contributors

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to improving traffic accident forecasting!
