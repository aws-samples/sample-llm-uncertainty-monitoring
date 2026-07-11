# Local pre-commit / lint / test runner.
# Mirrors what the GitLab CI pipeline does, so you can validate locally before pushing.
# Usage:  pwsh -File scripts/check.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> pre-commit hooks" -ForegroundColor Cyan
pre-commit run --all-files

Write-Host "==> ruff" -ForegroundColor Cyan
ruff check .

Write-Host "==> black --check" -ForegroundColor Cyan
black --check .

Write-Host "==> mypy" -ForegroundColor Cyan
mypy src/

Write-Host "==> pytest" -ForegroundColor Cyan
pytest --cov=src --cov-report=term

Write-Host "All checks passed." -ForegroundColor Green
