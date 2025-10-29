# ================================
# STAGE 1 - Build (Node)
# ================================
FROM node:20-slim AS builder

WORKDIR /app

# Copy dependency files first (for better caching)
COPY package*.json ./

# Install dependencies (including devDependencies for build)
RUN npm install

# Copy application source code
COPY . .

# Build the Vite app (ensures VITE_ env variables are injected)
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build


# ================================
# STAGE 2 - Serve (Nginx)
# ================================
FROM nginx:1.25-alpine AS production

# Copy built static files from the builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration (if you have one)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose HTTP port
EXPOSE 80

# Run nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
