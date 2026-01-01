# Implementation Plan: Fix Exo Errors

**Feature**: Fix Exo Errors (Dependencies)
**Status**: Draft
**Author**: Antigravity

## Goal
Resolve runtime and static analysis errors caused by missing development dependencies in the `exo` project.

## Technical Context
- **Language**: Python 3.13
- **Dependency Manager**: `uv`
- **Testing Framework**: `pytest` (uses `fastapi.testclient` which requires `httpx`)
- **Static Analysis**: `basedpyright` (missing executable)

## Proposed Changes

### Configuration
1.  **Modify `pyproject.toml`**:
    -   Add `httpx` to `[dependency-groups.dev]` to support `starlette.testclient`/`fastapi.testclient`.
    -   Add `basedpyright` to `[dependency-groups.dev]` to ensure the `just check` command works.
    -   Run `uv sync` to install new dependencies.

## Verification Plan

### Automated Tests
-   **Run Tests**: Execute `just test` (or `uv run pytest src`) to verify tests collection and execution.
    -   *Success Criteria*: Tests collect and run without `ModuleNotFoundError: No module named 'httpx'`.
-   **Run Checks**: Execute `just check` (or `uv run basedpyright --project pyproject.toml`).
    -   *Success Criteria*: Command spawns successfully and performs type checking.

### Manual Verification
-   None required beyond automated commands.
