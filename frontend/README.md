## FastAPI contract

The UI currently sends:

```json
{
  "message": "your text"
}
```

Set the API URL with `VITE_FASTAPI_URL` or edit it directly in the page.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

The default development URL is `http://localhost:5173`.

## Files to customize

- `src/config.ts`: project title, group member names, default FastAPI endpoint
- `src/App.tsx`: request and response handling
- `src/styles.css`: page styling
