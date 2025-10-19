# Contributing to Fraud Detection System

Thank you for your interest in contributing to the Fraud Detection System! This document provides guidelines and instructions for contributing.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## 🚀 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

1. **Clear title** describing the bug  
2. **Detailed description** of the issue  
3. **Steps to reproduce** the problem  
4. **Expected behavior** vs actual behavior  
5. **Environment details** (OS, Python version, etc.)  
6. **Screenshots** if applicable  

### Suggesting Enhancements

We welcome suggestions for new features! Please create an issue with:

1. **Clear title** describing the enhancement  
2. **Detailed description** of the proposed feature  
3. **Use case** explaining why it would be useful  
4. **Possible implementation** approach (optional)  

### Pull Requests

1. **Fork the repository** and create your branch from `main`  
2. **Make your changes** following our coding standards  
3. **Add tests** for any new functionality  
4. **Update documentation** as needed  
5. **Ensure all tests pass** before submitting  
6. **Submit a pull request** with a clear description  

## 💻 Development Setup

### 1. Clone and Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/fraud_detection.git
cd fraud_detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

2.Create Branch
# Feature branch
git checkout -b feature/your-feature-name

# Bugfix branch
git checkout -b bugfix/issue-number-description

📝 Coding Standards

Python Style Guide

We follow PEP 8 with some modifications:
	•	Line length: Maximum 100 characters
	•	Indentation: 4 spaces
	•	Naming conventions:
	•	Classes: PascalCase
	•	Functions/variables: snake_case
	•	Constants: UPPER_CASE
	•	Docstrings: Use Google style docstrings

Code Formatting
# Format code with Black
black src tests

# Sort imports with isort
isort src tests

# Lint with flake8
flake8 src tests

# Type checking with mypy
mypy src

Example Docstring
def train_model(X_train, y_train, model_type='logistic'):
    """
    Train a fraud detection model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        model_type (str): Type of model to train
        
    Returns:
        model: Trained model object
        
    Raises:
        ValueError: If model_type is not supported
        
    Example:
        >>> model = train_model(X_train, y_train, 'xgboost')
    """
    pass

🧪 Testing
 Running Tests

# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_model.py::TestModelTrainer::test_train_single_model

Writing Tests
	•	Place tests in the tests/ directory
	•	Use descriptive test names: test_<what>_<condition>_<expected>
	•	Use fixtures for common setup
	•	Aim for high code coverage (>80%)

Example test:
	def test_preprocessor_removes_duplicates():
    """Test that preprocessor removes duplicate rows"""
    preprocessor = DataPreprocessor()
    df_with_dupes = create_sample_data_with_duplicates()
    df_cleaned = preprocessor.clean_data(df_with_dupes)
    assert df_cleaned.duplicated().sum() == 0

📚 Documentation

Code Documentation
	•	All public functions and classes must have docstrings
	•	Include type hints for function parameters
	•	Provide examples in docstrings when helpful

README Updates

If your changes affect:
	•	Installation process
	•	Usage examples
	•	Configuration
	•	API endpoints

Please update the README.md accordingly.

🔄 Git Workflow

Commit Messages

Follow the conventional commits format:
	<type>(<scope>): <subject>

		<body>

		<footer>

Types:
	•	feat: New feature
	•	fix: Bug fix
	•	docs: Documentation changes
	•	style: Code style changes (formatting)
	•	refactor: Code refactoring
	•	test: Adding or updating tests
	•	chore: Maintenance tasks

Examples:
	feat(preprocessing): add SMOTE oversampling support

	Add support for SMOTE oversampling to handle imbalanced datasets.
	This improves model performance on fraud detection.

	Closes #123

	fix(api): correct threshold validation logic

	The threshold validation was accepting values outside [0,1] range.
	Added proper validation to raise ValueError for invalid inputs.
	
	Fixes #456

Pull Request Process
	1.	Update your branch with latest main:
			git checkout main
			git pull origin main
			git checkout your-branch
			git rebase main

	2.	Run all checks before submitting:
			# Format code
			black src tests
			isort src tests

			# Run tests
			pytest

			# Check linting
			flake8 src tests
	3.	Submit PR with:

	•	Clear title and description
	•	Link to related issues
	•	Screenshots/examples if applicable
	•	List of changes made

	4.	Address review comments promptly
	5.	Squash commits if requested before merge

🎯 Priority Areas

High Priority
	•	SHAP/LIME model explainability
	•	Real-time streaming predictions
	•	Performance optimization
	•	Additional model algorithms

Medium Priority
	•	Enhanced visualization dashboard
	•	A/B testing framework
	•	Alert system (email/SMS)
	•	Model versioning

Documentation
	•	API usage examples
	•	Tutorial notebooks
	•	Video tutorials
	•	Architecture diagrams

📞 Questions?

If you have questions:
	1.	Check existing issues and documentation
	2.	Join our discussions
	3.	Ask in pull request comments
	4.	Contact maintainers

🏆 Recognition

Contributors will be:
	•	Listed in CONTRIBUTORS.md
	•	Mentioned in release notes
	•	Acknowledged in documentation

Thank you for contributing! 🙏