# RapidSegment Experiment Suite: Architecture & UI Design

## Overview
A modern, polished experiment suite designed to run natively within a Jupyter Lab or Jupyter Notebook environment. The suite acts as a control room for the `rapidsegment` library, allowing users to tune parameters, execute runs, inspect logs, track performance, and compare segments—all without spinning up an external localhost server (e.g., Streamlit).

---

## Global Tech Stack
*   **Environment:** Jupyter Lab / Jupyter Notebook
*   **UI Framework:** **Solara** (pure Python, React-like state management, Material Design) or **AnyWidget** (custom JS/HTML integration).
*   **Data Engine:** `rapidsegment` (core logic), Pandas/Polars (data manipulation).
*   **Storage Database:** **DuckDB** or **SQLite** (embedded, zero-configuration local database).
*   **Storage File System:** Standard Python `os`/`pathlib` for local directory management.

---

## Module 1: The Workbench (Create & Clone)
**Purpose:** The active workspace to configure, clone, and execute `rapidsegment` experiments without overwhelming the user with a wall of inputs.

**Design Details:**
*   **Two-Column Card Layout:** 
    *   *Left Column:* Grouped, collapsible parameter inputs (Data Scope, Segment Rules, Advanced Hyperparameters).
    *   *Right Column:* Real-time parameter summary and validation checks.
*   **State Injection:** A dropdown to select and instantly clone historical experiment parameters.
*   **Action Bar:** Sticky footer featuring a prominent, high-contrast black "Run Experiment" button.

**Module Tech Stack:**
*   **UI:** Solara reactive state (`use_state`) to bind UI sliders/dropdowns directly to Python variables.
*   **State Management:** JSON-based parameter dictionaries mapped to the UI components.

---

## Module 2: The Artifact Console (Outputs & Logs)
**Purpose:** Immediate, readable feedback during and after an experiment's execution.

**Design Details:**
*   **Split-Pane Layout:** Keeps the Jupyter cell at a fixed, predictable height.
    *   *Left Pane (Log Terminal):* Dark-mode, auto-scrolling terminal for real-time execution logs. Includes log-level filters (Info/Error).
    *   *Right Pane (SQL Inspector):* Code viewer with SQL syntax highlighting and collapsible blocks.
*   **Export Hub:** Buttons to download `.txt` logs or `.sql` files directly to the local machine.

**Module Tech Stack:**
*   **UI:** Solara Split-pane components.
*   **Real-Time Logs:** Python `logging` module piped to a reactive UI text component.
*   **Syntax Highlighting:** `Pygments` library rendered via HTML/CSS in the widget.
*   **I/O:** Python `pathlib` for reading/writing artifact files.

---

## Module 3: The Leaderboard (History & Performance)
**Purpose:** A centralized hub to track, rank, and route all saved experiments to identify the best-performing segments.

**Design Details:**
*   **Ranked Data Grid:** Sortable and filterable table displaying experiment names, abbreviated parameters, and North Star metrics (e.g., Conversion Rate, Lift).
*   **Inline Visualizations:** Embedded sparklines and relative bar charts within the table cells. *Styling note: Strict adherence to solid black visualizations to maintain a sharp, professional aesthetic and reduce visual fatigue.*
*   **Row-Level Actions:** Contextual menu per row to "Clone to Workbench" or select two segments to "Send to Arena".

**Module Tech Stack:**
*   **Database:** DuckDB (fast OLAP querying) or SQLite.
*   **Data Grid UI:** `ipydatagrid` or Solara's native dataframe viewer.
*   **Inline Visuals:** Lightweight SVG generation or `matplotlib` micro-charts rendered as HTML inside the grid cells.

---

## Module 4: The Arena (1v1 Comparison)
**Purpose:** A side-by-side analytical view to instantly identify why one segment outperformed another.

**Design Details:**
*   **KPI Face-Off:** Center-aligned metric names flanked by Segment A and Segment B values, underlined by a delta bar chart showing the numerical gap.
*   **Parameter Diff:** A filtered view showing *only* the parameters that differ between the two segments.
*   **SQL Diff Viewer:** A GitHub-style code comparison window highlighting the exact lines of SQL that changed.

**Module Tech Stack:**
*   **Data Fetching:** DuckDB `JOIN` queries to pull both experiment records simultaneously.
*   **Diff Engine:** Python's built-in `difflib` (for generating SQL line-by-line differences) or a JS-based diff viewer via AnyWidget.
*   **UI:** Multi-column layout components to structure the side-by-side comparisons cleanly.

---

## Underlying Architecture: Hybrid Local Storage
To achieve a zero-setup, serverless design, the suite uses a self-contained embedded storage model hidden within the project directory.

**Directory Structure:**
```text
my_project/
├── notebook.ipynb
└── .rapidsegment_suite/
    ├── suite_data.db          <-- DuckDB/SQLite (Structured Metadata, Metrics, JSON params)
    └── artifacts/             <-- Local File System
        ├── exp_01/
        │   ├── logs.txt
        │   └── query.sql
        └── exp_02/
            ├── logs.txt
            └── query.sql