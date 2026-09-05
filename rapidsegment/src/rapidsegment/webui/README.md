# RapidSegment — Web UI

React (Vite + TypeScript) frontend and FastAPI backend, mirroring the Streamlit
RapidSegment suite (modules 1–6) one-to-one in a cyberpunk emerald-on-black theme.

```
webui/
├── backend/            FastAPI app (JSON API + optional static frontend)
│   ├── main.py
│   ├── config.py       mirrors rapidsegment config (presets, maps, validate)
│   ├── storage.py      suite DB, artifacts, templates
│   ├── services/       engine orchestration + m1–m6 service logic
│   └── routers/        m1.py … m6.py, /api/health, /api/exit
└── frontend/           Vite + React + TypeScript SPA
    ├── vite.config.ts  dev proxy: /api → http://localhost:8000
    ├── src/api.ts      typed fetch wrapper
    ├── src/types.ts    API payload types
    ├── src/components/ Plot.tsx, ui.tsx (charts, tables, pills, terminal)
    └── src/pages/      Home, M1DataLoader … M6Arena, Exit, App
```

## Development

### Backend

```bash
cd backend
python3 -m uvicorn main:app --reload --port 8000
```

Runs the JSON API under `/api`. The suite database and artifacts live under
`backend/experiments` (see `storage.SUITE_DIR`). Module 5/6 endpoints read the
same experiment rows written by Module 3.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Starts the Vite dev server (default http://localhost:5173) with a proxy that
forwards `/api` to `http://localhost:8000` (configured in `vite.config.ts`).
Edit React sources under `src/`; HMR applies automatically.

## Production / single-service deploy

Build the SPA, then run only the backend — it mounts `frontend/dist` and serves
the SPA with a fallback to `index.html`, so deep links like `/m5` work.

```bash
cd frontend && npm run build
cd ../backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 .

## Hand-offs between pages

State that crosses page boundaries is kept minimal and tab-local:

- `sessionStorage["wb_pending_run"]` — a validated config object set by
  Module 2 ("Run") or Module 5 ("Clone → Workbench") and consumed by Module 3.
- `sessionStorage["m4_exp_id"]` — experiment id set by Module 5 ("View results")
  and consumed by Module 4 to preselect that run.

## Notes

- Plotly charts are rendered with `react-plotly.js` + `plotly.js-dist-min` and
  code-split into a separate lazy chunk.
- BigQuery loading in Module 1 uses environment credentials only
  (`gcloud auth application-default login` / `GOOGLE_APPLICATION_CREDENTIALS`);
  the app never stores secrets.
- Dataset files, materialization, extraction and experiment orchestration are
  strictly backend-side; the browser never reads raw suite DB files.
