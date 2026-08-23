# RapidSegment Maintenance & Upgrade Guide

How to keep dependencies current and the package healthy **without breaking builds**.
This single document covers the ground rules, the CI pipeline, routine upkeep,
major upgrades, dry-runs, and troubleshooting.

---

## 1. Ground rules

- **Never** `pip install` a dependency directly. Always change `pyproject.toml`
  (or use `uv add`) and let `uv lock` update `uv.lock`.
- **Always** commit `uv.lock` alongside any `pyproject.toml` change.
- **Always** run the smoke test (`uv run pytest`) before committing an upgrade.
- **`uv.lock` is the single source of truth** for exact versions. Never edit it by hand.
- **Upper bounds** in `pyproject.toml` (e.g. `duckdb>=1.5.5,<2.0`) intentionally
  block accidental **major** jumps. An upgrade can never cross a bound unless you
  raise it first.
- **`uv_build` `source-exclude`** keeps the published package to the core library
  only — runtime artifacts (`.rapidsegment_suite`, `*.duckdb`, `*.db`, `*.json`,
  `experiments/`) and `docs/`, `Examples/`, `Banner.png`, `sync_readme.py` never ship
  in the sdist/wheel. (`uv_build` does **not** read `.gitignore`, so exclusion is
  done in `pyproject.toml`, not in `.gitignore`.)

---

## 2. Repo layout & where commands run

- The Python package, `pyproject.toml`, and `uv.lock` all live in **`rapidsegment/`**.
  **Run every `uv` command from that directory.**
- The git repo root is the parent of `rapidsegment/` (e.g. `D:\RapidSegment`).
  `docs/`, `Examples/`, `Banner.png`, and `sync_readme.py` are intentionally kept in
  git but excluded from the package.
- Supported Python: **3.11 and 3.12** (`requires-python = ">=3.11"`). On 3.11 the lock
  resolves `numpy 2.4.6`; on 3.12 it resolves `numpy 2.5.x`.

---

## 3. The CI pipeline (`.github/workflows/ci.yml`)

### Purpose — what it is for

CI (Continuous Integration) is the project's **automated safety net**. It runs on
GitHub every time you push code or open a pull request, and its job is to prove that:

1. the **locked dependencies still install** (`uv sync --all-extras`), and
2. the **test suite still passes** (`uv run pytest`)

— on **every supported Python version** and at the **declared lower bounds**.

In short: it catches a broken dependency upgrade (or a code change that breaks the
build) *before* it reaches `main`, so the main branch is always green and installable.
A green CI on your PR is the signal that an upgrade is safe to merge.

### What it does — two jobs

| Job | What it does | Local equivalent (run in `rapidsegment/`) |
|-----|--------------|-------------------------------------------|
| **test** | Matrix `python-version: ["3.11", "3.12"]`. For each: `uv sync --all-extras` then `uv run pytest`. Proves the locked deps install and the engine works on both supported Pythons, **including** the `ui`/`excel`/`gcp`/`prettytable` extras. | `uv sync --python 3.11 --all-extras && uv run pytest`  (repeat for 3.12) |
| **lowest** | `python-version: 3.11`, `uv sync --resolution lowest` then `uv run pytest`. Validates the declared **lower bounds** (recommended for published libraries). | `uv sync --resolution lowest && uv run pytest` |

### How to use it

- **You do not run it manually in the normal flow.** It runs automatically when you:
  - push a commit to any branch, or
  - open / update a pull request against `main`.
- **Prerequisites:** the workflow file lives at `.github/workflows/ci.yml` and the
  repo has GitHub Actions enabled (on by default for GitHub repositories). No extra
  setup is needed.
- **Manual run (optional):** the workflow declares `workflow_dispatch`, so you can also
  trigger it any time from the repo's **Actions → CI → Run workflow** button, without
  making a push.
- **Reproduce a job locally:** use the "Local equivalent" commands in the table above.
  If a CI job is red, run the matching command locally to see the failure before
  pushing a fix.
- **Reading the results:** on the PR, or in the repo's **Actions** tab, each job shows
  green ✓ (pass) or red ✗ (fail) with full logs. Click a failed job to see which
  Python version / dependency broke.
- **If a job fails:** do **not** merge. Reproduce locally, fix the code or the pin,
  push again, and let CI re-validate.

---

## 4. Routine maintenance tasks

### A. Periodic in-bounds upgrade (do this monthly)

Pulls in newer patch/minor releases that stay inside the declared bounds
(e.g. the latest `duckdb` 1.x). It will **not** cross an upper bound (e.g. `duckdb`
stays `<2.0` by design).

```bash
cd rapidsegment
uv lock --upgrade            # bump everything within current bounds
uv lock --check              # confirm lock is consistent with pyproject.toml
uv sync --all-extras
uv run pytest               # must stay green
uv tree                     # (optional) inspect the resolved graph
git add pyproject.toml uv.lock && git commit -m "chore: routine dependency upgrade"
git push                    # CI re-validates on both Pythons
```

### B. Upgrade a single package within its bounds

```bash
uv lock --upgrade-package duckdb
uv run pytest
git add uv.lock && git commit -m "chore: upgrade duckdb"
```

### C. Add a dependency

`uv add` updates both `pyproject.toml` and `uv.lock` together:

```bash
uv add "somepkg>=1.0"                  # runtime dependency
uv add --group dev pytest              # dev-only (pytest is already added)
uv add --optional gcp "new-gcp-pkg"    # an extra
```

### D. Remove or change a dependency

```bash
uv remove somepkg
# or edit the version string in pyproject.toml, then:
uv lock
```

---

## 5. Major upgrade runbook (crossing an upper bound)

"Major" = the version leaves the current range, e.g. adopting **duckdb 2.0**
(or pandas 3.0 / numpy 3.0). Do it deliberately:

1. **Raise the bound** in `pyproject.toml`, keeping the floor:

   ```toml
   duckdb = ">=1.5.5,<3.0"
   ```

   Optionally pin a specific release so a later `uv lock` does not jump to 2.1:

   ```bash
   uv lock --upgrade-package duckdb==2.0.0
   ```

2. **Resolve and sync** (the extras must still resolve against the new version):

   ```bash
   uv lock
   uv sync --all-extras
   ```

3. **Validate:**

   ```bash
   uv run pytest
   ```

4. **If green**, commit both files:

   ```bash
   git add pyproject.toml uv.lock && git commit -m "chore: upgrade duckdb to 2.x"
   ```

5. **If red**, fix the *source*, not the lock. Likely spots for a DuckDB bump:
   - `src/rapidsegment/builder.py` — `con.register(...)` / SQL built around the data
     view in `extract_segments`.
   - `src/rapidsegment/scorer.py` — DuckDB queries in `calculate_and_export_weights`.
   - `utils/data_loader.py` uses **pyarrow** (`pa_csv.read_csv`,
     `pa.parquet.read_table`), not DuckDB; `utils/undersampler.sql` is **BigQuery**
     SQL and is unaffected by a DuckDB upgrade.

   Re-run pytest, then commit.

---

## 6. Pre-flight dry-run (validate without committing)

Prove the safety net works **before** you ever ship a major bump:

```bash
cd rapidsegment
uv lock --upgrade-package duckdb==2.0.0
uv sync --all-extras
uv run pytest
git checkout pyproject.toml uv.lock     # revert if not ready to ship
```

(`uv.lock` is committed, so `git checkout` restores the last committed versions.)

---

## 7. Troubleshooting

- **"resolution failed for other Python versions" / numpy floor error**
  A dependency's floor is above `requires-python` (e.g. `numpy>=2.5.1` needs 3.12
  while `requires-python` says `>=3.9`). Fix: raise `requires-python` (currently
  `>=3.11`) or lower the dependency pin. Keep the two in agreement.

- **`uv lock --check` reports a mismatch**
  The lock is out of sync with `pyproject.toml`. Run `uv lock` to refresh.

- **`uv lock` says requirements are unsatisfiable**
  Conflicting pins. Relax a bound or use `uv lock --upgrade-package <pkg>`.

- **Tests fail after an upgrade**
  Revert the lock first to unblock work:
  `git checkout uv.lock pyproject.toml`, then attempt a smaller step or fix the source.

- **UI / Streamlit can't be unit-tested headlessly**
  The smoke test only *imports* `rapidsegment.ui` when `streamlit` is installed.
  After bumping the `ui` extra, manually run `rapidsegment-ui` to validate the UI.

---

## 8. Quick-reference cheat sheet

Run everything from `rapidsegment/`.

| Goal | Command |
|------|---------|
| Refresh all in-bounds deps | `uv lock --upgrade` |
| Upgrade one package | `uv lock --upgrade-package <pkg>` |
| Upgrade to an exact version | `uv lock --upgrade-package <pkg>==x.y.z` |
| Verify lock is consistent | `uv lock --check` |
| Install env (incl. extras) | `uv sync --all-extras` |
| Run tests | `uv run pytest` |
| Validate lower bounds (the `lowest` CI job) | `uv sync --resolution lowest && uv run pytest` |
| Add runtime / dev / extra dep | `uv add [--group dev \| --optional X] <pkg>` |
| Remove a dependency | `uv remove <pkg>` |
| Dry-run a major bump | raise bound → `uv lock --upgrade-package <pkg>==ver` → test → `git checkout` |
| Reproduce the CI `test` job (3.11) | `uv sync --python 3.11 --all-extras && uv run pytest` |

---

## 9. References

- `.github/workflows/ci.yml` — the CI pipeline described in §3.
- `rapidsegment/pyproject.toml` — dependency pins, extras, `source-exclude`, `requires-python`.
- `rapidsegment/uv.lock` — exact resolved versions (commit this).
- `rapidsegment/tests/test_smoke.py` — the safety-net tests run by CI.
