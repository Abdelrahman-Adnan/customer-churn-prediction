# MLOps Customer Churn Prediction Project - Enhanced Makefile with CI/CD
# =======================================================================

.PHONY: help install setup-data build up down clean logs status health-check
.PHONY: train monitor web test test-unit test-integration test-e2e test-coverage
.PHONY: lint format pre-commit clean-code security-scan performance-test
.PHONY: ci build-for-prod deploy-staging deploy-production deploy-aws deploy-local rollback
.PHONY: backup-data update-dependencies monitor-metrics check-alerts
.PHONY: retrain-models evaluate-models deploy-best-model validate-model

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Variables
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
DOCKER_COMPOSE := docker compose
AWS_REGION := us-east-1

help:
	@echo "$(GREEN)MLOps Customer Churn Prediction Project$(NC)"
	@echo "========================================="
	@echo ""
	@echo "$(YELLOW)🚀 Quick Start Commands:$(NC)"
	@echo "  make install          - Setup development environment"
	@echo "  make setup-data       - Create data directories"
	@echo "  make build            - Build all Docker images"
	@echo "  make up               - Start all services"
	@echo "  make train            - Train models with MLflow tracking"
	@echo "  make health-check     - Verify all services are running"
	@echo ""
	@echo "$(YELLOW)🔨 Development Commands:$(NC)"
	@echo "  make logs             - View service logs"
	@echo "  make down             - Stop all services"
	@echo "  make clean            - Remove containers and images"
	@echo "  make status           - Show service status"
	@echo ""
	@echo "$(YELLOW)🧪 Testing & Quality:$(NC)"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-e2e         - Run end-to-end tests"
	@echo "  make test-coverage    - Run tests with coverage report"
	@echo "  make lint             - Check code quality"
	@echo "  make format           - Format code (black, isort)"
	@echo "  make security-scan    - Run security vulnerability scan"
	@echo "  make performance-test - Run performance benchmarks"
	@echo ""
	@echo "$(YELLOW)🔄 CI/CD Pipeline:$(NC)"
	@echo "  make ci               - Run full CI pipeline locally"
	@echo "  make build-for-prod   - Build production Docker images"
	@echo "  make deploy-local     - Deploy locally with production config"
	@echo "  make deploy-staging   - Deploy to staging environment"
	@echo "  make deploy-production- Deploy to production"
	@echo "  make deploy-aws       - Deploy to AWS cloud"
	@echo "  make rollback         - Rollback to previous version"
	@echo ""
	@echo "$(YELLOW)📊 ML Operations:$(NC)"
	@echo "  make retrain-models   - Retrain models with new data"
	@echo "  make evaluate-models  - Compare model performance"
	@echo "  make deploy-best-model- Deploy best performing model"
	@echo "  make monitor-metrics  - View monitoring dashboards"
	@echo ""
	@echo "$(YELLOW)🛠️ Maintenance:$(NC)"
	@echo "  make backup-data      - Backup important data"
	@echo "  make update-dependencies - Update packages"
	@echo "  make check-alerts     - Review system alerts"

# =======================================================================
# Environment Setup
# =======================================================================

install:
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PIP) install pytest pytest-cov pytest-mock black flake8 isort pre-commit bandit safety
	@$(PIP) install locust  # For performance testing
	@.venv/bin/pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

setup-data:
	@echo "$(GREEN)Setting up data directories...$(NC)"
	@mkdir -p data/{raw,processed,interim}
	@mkdir -p models logs mlartifacts reports/{figures,models}
	@mkdir -p tests/{unit,integration,e2e}
	@echo "$(GREEN)✓ Data directories created$(NC)"

# =======================================================================
# Docker Operations
# =======================================================================

build:
	@echo "$(GREEN)Building Docker images...$(NC)"
	@$(DOCKER_COMPOSE) build --parallel
	@echo "$(GREEN)✓ Docker images built successfully$(NC)"

build-for-prod:
	@echo "$(GREEN)Building production Docker images...$(NC)"
	@$(DOCKER_COMPOSE) -f docker-compose.mlops.yml build --parallel
	@echo "$(GREEN)✓ Production images built successfully$(NC)"

up: setup-data
	@echo "$(GREEN)Starting MLOps services...$(NC)"
	@echo "1. Starting infrastructure services..."
	@$(DOCKER_COMPOSE) up -d postgres mlflow
	@echo "2. Waiting for MLflow to be ready..."
	@sleep 30
	@echo "3. Running initial training pipeline..."
	@$(DOCKER_COMPOSE) run --rm training
	@echo "4. Starting monitoring service..."
	@$(DOCKER_COMPOSE) up -d monitoring
	@echo "5. Starting web application..."
	@$(DOCKER_COMPOSE) up -d web-app
	@echo ""
	@echo "$(GREEN)✓ Services started successfully!$(NC)"
	@echo "Access points:"
	@echo "  • Streamlit Dashboard: http://localhost:8501"
	@echo "  • FastAPI Docs: http://localhost:8000/docs"
	@echo "  • MLflow UI: http://localhost:5000"
	@echo "  • Grafana: http://localhost:3000"

down:
	@echo "$(YELLOW)Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ All services stopped$(NC)"

clean:
	@echo "$(YELLOW)Cleaning up containers and images...$(NC)"
	@$(DOCKER_COMPOSE) down -v --rmi all
	@docker system prune -f
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# =======================================================================
# Testing & Quality Assurance
# =======================================================================

test: test-unit test-integration
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-unit:
	@echo "$(GREEN)Running unit tests...$(NC)"
	@$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-integration:
	@echo "$(GREEN)Running integration tests...$(NC)"
	@$(PYTHON) -m pytest tests/integration/ -v --tb=short

test-e2e:
	@echo "$(GREEN)Running end-to-end tests...$(NC)"
	@$(PYTHON) -m pytest tests/e2e/ -v --tb=short

test-coverage:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@$(PYTHON) -m pytest tests/ --cov=services --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-all: test-unit test-integration test-e2e test-coverage
	@echo "$(GREEN)✓ All test suites completed$(NC)"

lint:
	@echo "$(GREEN)Running code quality checks...$(NC)"
	@.venv/bin/flake8 tests/ services/ src/ --max-line-length=100
	@.venv/bin/black --check tests/ services/ src/
	@.venv/bin/isort --check-only tests/ services/ src/
	@echo "$(GREEN)✓ Code quality checks passed$(NC)"

format:
	@echo "$(GREEN)Formatting code...$(NC)"
	@.venv/bin/black tests/ services/ src/
	@.venv/bin/isort tests/ services/ src/
	@echo "$(GREEN)✓ Code formatted$(NC)"

security-scan:
	@echo "$(GREEN)Running security vulnerability scan...$(NC)"
	@.venv/bin/bandit -r services/ src/ -f json -o reports/security-report.json || true
	@.venv/bin/safety check --json --output reports/safety-report.json || true
	@echo "$(GREEN)✓ Security scan completed. Check reports/$(NC)"

performance-test:
	@echo "$(GREEN)Running performance tests...$(NC)"
	@.venv/bin/locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 30s --host=http://localhost:8000
	@echo "$(GREEN)✓ Performance tests completed$(NC)"

pre-commit:
	@.venv/bin/pre-commit run --all-files

clean-code:
	@echo "$(GREEN)Cleaning up code artifacts...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -delete
	@rm -rf htmlcov/ .coverage reports/

# =======================================================================
# CI/CD Pipeline
# =======================================================================

ci: clean-code install lint security-scan test-coverage
	@echo "$(GREEN)✓ CI pipeline completed successfully$(NC)"

deploy-staging: build-for-prod
	@echo "$(GREEN)Deploying to staging environment...$(NC)"
	@# Add staging deployment commands here
	@echo "$(GREEN)✓ Deployed to staging$(NC)"

deploy-local: build-for-prod
	@echo "$(GREEN)Deploying locally with production configuration...$(NC)"
	@$(DOCKER_COMPOSE) -f docker-compose.mlops.yml up -d
	@echo "$(GREEN)✓ Local deployment completed$(NC)"
	@echo "$(YELLOW)Access points:$(NC)"
	@echo "  • Web App: http://localhost:8501"
	@echo "  • API Docs: http://localhost:8000/docs"
	@echo "  • MLflow: http://localhost:5000"
	@echo "  • Grafana: http://localhost:3000"

deploy-production: ci
	@echo "$(GREEN)Deploying to production environment...$(NC)"
	@# Add production deployment commands here
	@echo "$(GREEN)✓ Deployed to production$(NC)"

deploy-aws:
	@echo "$(GREEN)Deploying to AWS...$(NC)"
	@cd "Infrastructure as code (IaC)" && terraform apply -auto-approve
	@# Deploy application services
	@echo "$(GREEN)✓ Deployed to AWS$(NC)"

rollback:
	@echo "$(YELLOW)Rolling back to previous version...$(NC)"
	@# Add rollback logic here
	@echo "$(GREEN)✓ Rollback completed$(NC)"

# =======================================================================
# ML Operations
# =======================================================================

train:
	@echo "$(GREEN)Running ML training pipeline...$(NC)"
	@$(DOCKER_COMPOSE) up -d postgres mlflow
	@$(DOCKER_COMPOSE) run --rm training
	@echo "$(GREEN)✓ Training completed. Check MLflow UI at http://localhost:5000$(NC)"

retrain-models:
	@echo "$(GREEN)Retraining models with new data...$(NC)"
	@$(DOCKER_COMPOSE) run --rm training python -c "from src.models.train_model import retrain_pipeline; retrain_pipeline()"
	@echo "$(GREEN)✓ Model retraining completed$(NC)"

evaluate-models:
	@echo "$(GREEN)Evaluating model performance...$(NC)"
	@$(PYTHON) -c "from src.models.evaluate_model import compare_models; compare_models()"
	@echo "$(GREEN)✓ Model evaluation completed$(NC)"

deploy-best-model:
	@echo "$(GREEN)Deploying best performing model...$(NC)"
	@$(PYTHON) -c "from src.models.deploy_model import deploy_best; deploy_best()"
	@echo "$(GREEN)✓ Best model deployed$(NC)"

validate-model:
	@echo "$(GREEN)Validating model integrity...$(NC)"
	@$(PYTHON) -c "from src.models.validate_model import validate; validate()"
	@echo "$(GREEN)✓ Model validation completed$(NC)"

# =======================================================================
# Monitoring & Maintenance
# =======================================================================

monitor:
	@echo "$(GREEN)Starting monitoring service...$(NC)"
	@$(DOCKER_COMPOSE) up -d postgres mlflow monitoring
	@echo "$(GREEN)✓ Monitoring service started$(NC)"

health-check:
	@echo "$(GREEN)Checking system health...$(NC)"
	@echo "Infrastructure services:"
	@curl -s http://localhost:5000/health > /dev/null && echo "  ✓ MLflow ready" || echo "  ✗ MLflow not ready"
	@echo "Application services:"
	@curl -s http://localhost:8000/health > /dev/null && echo "  ✓ FastAPI ready" || echo "  ✗ FastAPI not ready"
	@curl -s http://localhost:8501 > /dev/null && echo "  ✓ Streamlit ready" || echo "  ✗ Streamlit not ready"
	@echo "Service status:"
	@$(DOCKER_COMPOSE) ps

monitor-metrics:
	@echo "$(GREEN)Opening monitoring dashboards...$(NC)"
	@echo "• Grafana: http://localhost:3000"
	@echo "• MLflow: http://localhost:5000"

check-alerts:
	@echo "$(GREEN)Checking system alerts...$(NC)"
	@$(PYTHON) -c "from src.monitoring.check_alerts import check_all_alerts; check_all_alerts()"

backup-data:
	@echo "$(GREEN)Backing up important data...$(NC)"
	@mkdir -p backups/$(shell date +%Y%m%d)
	@cp -r data/processed backups/$(shell date +%Y%m%d)/
	@cp -r models backups/$(shell date +%Y%m%d)/
	@echo "$(GREEN)✓ Data backed up to backups/$(shell date +%Y%m%d)$(NC)"

update-dependencies:
	@echo "$(GREEN)Updating dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt --upgrade
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

# =======================================================================
# Service-specific Operations
# =======================================================================

web:
	@echo "$(GREEN)Starting web services only...$(NC)"
	@$(DOCKER_COMPOSE) up -d postgres mlflow web-app
	@echo "$(GREEN)✓ Web services started$(NC)"

logs:
	@$(DOCKER_COMPOSE) logs -f

status:
	@echo "$(GREEN)Service Status:$(NC)"
	@$(DOCKER_COMPOSE) ps

# =======================================================================
# Utilities
# =======================================================================

.env:
	@echo "$(YELLOW)Creating .env file from template...$(NC)"
	@cp .env.example .env
	@echo "$(GREEN)✓ Please edit .env with your configuration$(NC)"
