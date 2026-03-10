.PHONY: install install-dev data prepare-data compute-embeddings train-protonet \
        eval-lfw eval run-example serve test lint fmt format check-types clean \
        migrate migrate-status migrate-dry-run backup backup-s3

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
install:          ## Install package in editable mode (runtime deps only)
	pip install -e .

install-dev:      ## Install package + all dev tools
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt
	pre-commit install

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline – individual steps (can be re-run independently)
# ─────────────────────────────────────────────────────────────────────────────

# Step 1 – Download LFW to data/lfw/
prepare-data:     ## [Step 1] Download LFW dataset to data/lfw/
	python -m src.cli.run prepare-data --dataset lfw --out-dir data/lfw

# Alias kept for backward compat
data: prepare-data

# Step 2 – Compute backbone embeddings and save as .npz
compute-embeddings: ## [Step 2] Run backbone over LFW; save embeddings/lfw_embeddings.npz
	python -m src.cli.run compute-embeddings \
		--root data/lfw \
		--out  embeddings/lfw_embeddings.npz \
		--batch-size 64

# Step 3 – Train Prototypical Network from configs/proto.yaml
train-protonet:   ## [Step 3] Meta-train ProtoNet (requires prepare-data first)
	python -m src.cli.run train-protonet \
		--config configs/proto.yaml \
		--data-root data/lfw \
		--out-dir experiments/checkpoints

# Step 4 – Full episodic evaluation + report
eval-lfw:         ## [Step 4] Episodic eval (K=1,2,3,5) + CSV + PDF/PNG report
	python -m src.cli.run eval-lfw \
		--embeddings embeddings/lfw_embeddings.npz \
		--episodes 1000 \
		--k 1 2 3 5 \
		--out experiments/lfw_results

# ─────────────────────────────────────────────────────────────────────────────
# Run-example  (complete pipeline on a small subset for quick demo / CI smoke-test)
#
# Uses only 50 episodes and K=1,5 to finish quickly without a GPU.
# All four pipeline steps run end-to-end; outputs land in experiments/demo_run/.
# ─────────────────────────────────────────────────────────────────────────────
DEMO_OUT := experiments/demo_run
DEMO_EMB := $(DEMO_OUT)/embeddings.npz

run-example:      ## End-to-end demo: download → embed → eval → report (small subset)
	@echo "──────────────────────────────────────────────────────────────"
	@echo " Privacy Campus Security – full baseline demo (small subset)"
	@echo "──────────────────────────────────────────────────────────────"
	@echo
	@echo "[1/4] Preparing LFW dataset …"
	python -m src.cli.run prepare-data --dataset lfw --out-dir data/lfw
	@echo
	@echo "[2/4] Computing embeddings (batch-size=32) …"
	python -m src.cli.run compute-embeddings \
		--root data/lfw \
		--out  $(DEMO_EMB) \
		--batch-size 32
	@echo
	@echo "[3/4] Training ProtoNet (5-way 1-shot, 10 epochs) …"
	python -m src.cli.run train-protonet \
		--config  configs/proto.yaml \
		--data-root data/lfw \
		--out-dir $(DEMO_OUT)/checkpoints
	@echo
	@echo "[4/4] Evaluating (K=1,5 | 50 episodes each) …"
	python -m src.cli.run eval-lfw \
		--embeddings $(DEMO_EMB) \
		--episodes 50 \
		--k 1 5 \
		--skip-pairs \
		--seed 42 \
		--out $(DEMO_OUT)/results
	@echo
	@echo "──────────────────────────────────────────────────────────────"
	@echo " Demo complete – outputs written to $(DEMO_OUT)/"
	@echo "   Metrics CSV : $(DEMO_OUT)/results/metrics_summary.csv"
	@echo "   Report MD   : $(DEMO_OUT)/results/report.md"
	@echo "   Report PNG  : $(DEMO_OUT)/results/report.png"
	@echo "   Report PDF  : $(DEMO_OUT)/results/report.pdf"
	@echo "   Eval log    : $(DEMO_OUT)/results/eval.log"
	@echo "──────────────────────────────────────────────────────────────"

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (legacy pairs-only shortcut)
# ─────────────────────────────────────────────────────────────────────────────
eval:             ## Evaluate backbone on LFW pairs only (requires 'make data' first)
	python -m src.eval.evaluate_lfw --root data/lfw --batch-size 64
	@echo "Metrics written to experiments/sample_runs/lfw_metrics.json"

# ─────────────────────────────────────────────────────────────────────────────
# DB migrations
# Requires: no extra packages (uses only stdlib sqlite3)
# ─────────────────────────────────────────────────────────────────────────────
migrate:          ## Apply all pending DB schema migrations
	python scripts/migrate.py

migrate-status:   ## Show applied / pending migration status for all databases
	python scripts/migrate.py --status

migrate-dry-run:  ## Preview migrations without applying them
	python scripts/migrate.py --dry-run

# ─────────────────────────────────────────────────────────────────────────────
# DB backups
# Requires: no extra packages for local backup.
#           pip install boto3  for S3 upload (optional).
# Env:      PCS_BACKUP_DIR (default: data/backups)
#           PCS_BACKUP_RETAIN (default: 7 days)
#           PCS_BACKUP_S3_BUCKET (disabled when unset)
# ─────────────────────────────────────────────────────────────────────────────
backup:           ## Back up all SQLite databases with daily rotation
	python scripts/backup.py

backup-s3:        ## Back up all databases and upload to S3 (requires PCS_BACKUP_S3_BUCKET)
	python scripts/backup.py --s3-bucket "$${PCS_BACKUP_S3_BUCKET}"

# ─────────────────────────────────────────────────────────────────────────────
# Web UI
# ─────────────────────────────────────────────────────────────────────────────
serve:            ## Start Flask dashboard on http://localhost:5000
	python src/web/flask_app.py

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
test:             ## Run test suite with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:htmlcov
	@echo "Coverage HTML report: htmlcov/index.html"

# ─────────────────────────────────────────────────────────────────────────────
# Linting, formatting, type checking
# ─────────────────────────────────────────────────────────────────────────────
lint:             ## Ruff + flake8 + isort check (CI-safe, no mutations)
	ruff check src/ tests/
	flake8 src/ tests/
	isort --check-only src/ tests/

format:           ## Auto-format with black + isort + ruff
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

fmt: format       ## Alias for 'make format'

check-types:      ## Run mypy static type checks
	mypy src/

check: lint check-types  ## Run all static checks

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
clean:            ## Remove all build / cache artefacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov dist build *.egg-info

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
help:             ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
