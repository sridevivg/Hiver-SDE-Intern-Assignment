# SupportGraph AI — Frontend

> Minimal React + TypeScript + Vite application.

## Stack

- **React 18** — UI library
- **TypeScript** — static typing
- **Vite** — fast build tool and dev server

## Current State

This is a minimal scaffold confirming the frontend foundation.
The full dashboard will be built in **Phase 13 — Frontend Dashboard** (PLANNED).

It currently displays:

- SupportGraph AI — project name
- Agentic Customer Support Intelligence System — project subtitle
- Current Phase: Dataset Exploration
- Development roadmap

## Commands

```bash
npm install
npm run dev     # Start dev server on http://localhost:5173
npm run build   # Production build
```

## API Connection

The Vite dev server proxies `/api/*` to the FastAPI backend at `http://localhost:8000`.

Set `VITE_API_BASE_URL` in `.env.local` to override the backend URL.
