# MatKit Tests

This directory contains automated tests for the MatKit library.

## Running Tests

### Install pytest

```bash
pip install pytest pytest-cov
```

### Run all tests

```bash
# From the MatKit root directory
pytest

# Or with coverage report
pytest --cov=matkit --cov-report=html
```

### Run specific test files

```bash
# Test latex_array module only
pytest tests/test_latex_array.py

# Test with verbose output
pytest tests/test_latex_array.py -v
```

### Run specific test classes or functions

```bash
# Run a specific test class
pytest tests/test_latex_array.py::TestLatexArray

# Run a specific test function
pytest tests/test_latex_array.py::TestLatexArray::test_scalar
```

## Test Structure

- `test_latex_array.py` - Tests for LaTeX rendering functionality
- `test_field_management.py` - Tests for mesh field management
- `test_seamless.py` - Tests for seamless integration features
- `test_sympy_display.py` - Tests for SymPy display functionality

## Writing Tests

When adding new tests:

1. Create test files with the `test_*.py` naming pattern
2. Use descriptive test function names starting with `test_`
3. Group related tests in classes prefixed with `Test`
4. Add docstrings to explain what each test validates
5. Use appropriate assertions to verify behavior

Example:

```python
class TestMyFeature:
    """Tests for my new feature."""

    def test_basic_functionality(self):
        """Test that basic feature works correctly."""
        result = my_function(input_data)
        assert result == expected_output
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. They should:

- Run quickly (under 10 seconds total)
- Not require user interaction
- Have no external dependencies beyond listed in `pyproject.toml`
- Pass consistently across different environments
