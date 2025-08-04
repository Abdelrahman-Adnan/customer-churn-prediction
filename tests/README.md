# Test Suite Documentation

## Overview
This test suite ensures our Customer Churn Prediction MLOps system works reliably. Each component is thoroughly tested with simple, focused tests.

## Test Files

### `test_fastapi_app.py`
**FastAPI Application Tests**

Tests our REST API that serves churn predictions:
- **App Configuration**: Checks title, version, and setup
- **Model Loading**: Ensures MLflow model loading works
- **Input Validation**: Tests Pydantic data models
- **Endpoints**: Verifies `/predict` and `/health` endpoints
- **Error Handling**: Tests invalid input scenarios
- **Data Processing**: Checks prediction data flow

**Key Features Tested:**
- Customer data validation with proper types
- MLflow model integration
- API response formatting
- Error handling for edge cases

---

### `test_streamlit_app.py`
**Streamlit Dashboard Tests**

Tests our interactive web interface:
- **UI Components**: Validates form inputs and layout
- **API Communication**: Tests FastAPI backend calls
- **Data Visualization**: Checks chart creation logic
- **Risk Calculation**: Validates churn probability categorization
- **User Experience**: Tests form validation and error handling
- **Model Info Display**: Ensures model metadata is shown

**Key Features Tested:**
- Real-time prediction requests
- Risk level categorization (High/Medium/Low)
- Interactive form validation
- Chart and graph generation

---

### `test_churn_pipeline.py`
**ML Training Pipeline Tests**

Tests our machine learning training system:
- **Data Loading**: Validates data ingestion and structure
- **Preprocessing**: Tests data cleaning and transformation
- **Feature Engineering**: Checks feature creation logic
- **Model Training**: Validates ML model training process
- **Evaluation**: Tests metrics calculation (accuracy, precision, etc.)
- **MLflow Integration**: Checks experiment tracking and logging
- **Model Persistence**: Tests model saving and loading

**Key Features Tested:**
- Data validation and quality checks
- Train/validation/test data splitting
- Model performance evaluation
- MLflow experiment tracking
- Model serialization and storage

---

## Running the Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Individual Test Files
```bash
# FastAPI tests
python -m pytest tests/test_fastapi_app.py -v

# Streamlit tests
python -m pytest tests/test_streamlit_app.py -v

# ML Pipeline tests
python -m pytest tests/test_churn_pipeline.py -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=services --cov=src --cov-report=html
```

## Test Strategy

### Simple & Focused
- Each test checks one specific functionality
- Clear naming conventions
- Minimal setup with mocked dependencies

### Practical Testing
- Tests core business logic without external dependencies
- Uses sample data for realistic scenarios
- Mocks external services (MLflow, APIs) for isolation

### Robust Coverage
- **Unit Tests**: Individual functions and components
- **Integration Tests**: Component interactions
- **API Tests**: Endpoint functionality and data flow
- **Pipeline Tests**: End-to-end ML workflow validation

## Test Results Expected

When you run the tests, you should see:
- **Green passes** for working functionality
- **Mocked dependencies** working correctly
- **High coverage** of critical code paths
- **Fast execution** (under 30 seconds total)

## Troubleshooting

If tests fail:
1. **Import Errors**: Install missing dependencies with `pip install pytest`
2. **Path Issues**: Ensure you're running from project root directory
3. **Mock Failures**: Check that external services are properly mocked
4. **Data Issues**: Verify sample data generation in fixtures
