.PHONY: install dev prod lint test up down logs migrate migrate-new migrate-down migrate-history

install:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

lint:
	ruff check . --fix && ruff format .

test:
	pytest tests/ -v

# ── Docker ────────────────────────────────────────────────────────────────────
up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f app

# ── Migrations ────────────────────────────────────────────────────────────────
migrate:
	alembic upgrade head

migrate-new:
	@read -p "Migration name: " name; alembic revision --autogenerate -m "$$name"

migrate-down:
	alembic downgrade -1

migrate-history:
	alembic history --verbose

migrate-current:
	alembic current
