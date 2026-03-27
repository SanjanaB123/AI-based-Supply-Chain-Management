## Backend contract

The chat UI now reads the available stores from:

- `GET /api/stores`

It routes supported prompts to one of these backend endpoints using the selected store or a store code mentioned in the prompt:

- `GET /api/stock-levels?store=S001`
- `GET /api/sell-through?store=S001`
- `GET /api/days-of-supply?store=S001`
- `GET /api/stock-health?store=S001`
- `GET /api/lead-time-risk?store=S001`
- `GET /api/shrinkage?store=S001`

Set the API base URL with `VITE_API_BASE_URL` or edit it directly in the page.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

The default development URL is `http://localhost:5173`.

## Files to customize

- `src/config.ts`: project title, group member names, default backend base URL
- `src/App.tsx`: store loading, prompt routing, and response handling
- `src/styles.css`: page styling
