# Stage 1: Build
FROM node:18-slim AS build

WORKDIR /app

# D4: Use npm ci for deterministic builds
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:stable-alpine

# Copy build artifacts to nginx public dir
COPY --from=build /app/dist /usr/share/nginx/html

# Default nginx config (optional: add custom for SPA routing)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
