# ── Stage 1: Build ──────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (cached layer)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .

# Build args become env vars at build time (Vite inlines them)
ARG VITE_API_BASE_URL
ARG VITE_CLERK_PUBLISHABLE_KEY
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_CLERK_PUBLISHABLE_KEY=$VITE_CLERK_PUBLISHABLE_KEY

RUN npm run build

# ── Stage 2: Serve ──────────────────────────────────────────────
FROM node:20-alpine AS runner

WORKDIR /app

# Use a lightweight static file server
RUN npm install -g serve@14

# Copy built files from builder
COPY --from=builder /app/dist ./dist

EXPOSE 3000

# Serve the built files on port 3000
# -s enables single-page app mode (rewrites all routes to index.html)
CMD ["serve", "-s", "dist", "-l", "3000"]
