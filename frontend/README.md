# Flowcast — Frontend

A premium, map-first visualization for Bay Area bike-flow forecasts. Built to feel
like a modern mobility app: dark glass UI, GPU-accelerated map, choreographed
motion, and thoughtful interaction states.

![Flowcast preview](https://img.shields.io/badge/stack-React_18%20%2B%20TS%20%2B%20Vite-0b0d12?style=flat-square)
![Map](https://img.shields.io/badge/map-MapLibre%20GL%20%C2%B7%20no%20token-34e3c2?style=flat-square)
![Motion](https://img.shields.io/badge/motion-Framer%20Motion-ff6f91?style=flat-square)

---

## Quick start

From this `frontend/` directory:

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

The Flowcast FastAPI backend must be running locally. From the repo root:

```bash
python main.py --server
```

That starts uvicorn on **`http://127.0.0.1:9000`**. Vite's dev server proxies
`/stations` and `/predict` to it, so the browser stays on a single origin and
never hits CORS.

## Scripts

| Script            | Description                                     |
| ----------------- | ----------------------------------------------- |
| `npm run dev`     | Vite dev server with HMR + backend proxy        |
| `npm run build`   | Type-check (`tsc -b`) and build the prod bundle |
| `npm run preview` | Serve the prod build locally                    |

## Configuration

Environment variables live in `.env.local` (see `.env.example`).

- **`VITE_API_URL`** — leave unset in dev (use the proxy). In production, set
  it to the URL of your FastAPI backend, e.g. `https://api.flowcast.example.com`.

## Architecture

```
src/
  api/          REST client + endpoint helpers (typed)
  hooks/        TanStack Query hooks for stations + live forecast
  store/        Zustand store for UI state (selection, hover, search)
  components/
    map/        MapLibre instance, GPU station layer, animated overlays
    panels/     HeaderBar, Legend, StationDetailPanel, FlowMeter, BootSplash
    ui/         Skeleton, AnimatedNumber — small primitives
  lib/          Format helpers, color tokens, classname helper
  types/        Shared types (Station, FlowDirection, …)
  index.css     Tailwind layers + glassmorphism utilities + scrollbar styling
```

### How the map renders 600+ stations smoothly

All station dots live in a single MapLibre `geojson` source rendered by three
GPU-rasterized circle layers (glow, hover halo, crisp dot). That's one draw
call per layer regardless of station count — no per-marker DOM nodes, no
React reconciliation cost on pan/zoom.

The **selected** and **hovered** stations get an additional DOM overlay
positioned via `map.project()` so we can apply Framer Motion (spring scale-in,
animated pulse rings, glow). Best of both worlds: one draw call for the field,
full animation control for the focused dot.

### Stack choices

- **MapLibre GL JS + CARTO Dark Matter** — a free, beautiful, vector-tile
  basemap that ships out of the box (no Mapbox token required). MapLibre's API
  is a drop-in replacement for Mapbox GL, so swapping later is trivial.
- **Tailwind CSS** with a small custom token layer (`ink-*`, `flow-*`,
  glass utilities) — utility speed without "Tailwind spam".
- **TanStack Query** for caching + automatic refetch of `/live` data every 60s
  while a station is open.
- **Zustand** for the small slice of cross-cutting UI state.
- **Framer Motion** for spring panel entry, animated counters, pulse rings.
- **Inter Variable** + **JetBrains Mono Variable** via `@fontsource-variable`
  — fully self-hosted, no Google Fonts requests, with `tabular-nums` enabled
  for jitter-free numeric displays.

## Interaction model

- Click any station → panel springs in from the right, map flies to the
  station with cubic ease-out, the dot upgrades to a pulsing ring overlay.
- Hover any station → soft halo brightens via map feature-state.
- Press `/` → focus the search field. Escape → close the panel.
- Empty-map click → deselect.
- 60s background refresh on the open station's forecast.
- Skeleton shimmers for the metric and the bar while `/live` is in-flight;
  inline error card with one-click retry on failure.
