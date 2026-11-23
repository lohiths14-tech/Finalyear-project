/**
 * AgriPulse Service Worker
 * Strategy:
 *   - Static assets (JS, CSS, fonts, icons) -> Cache-first
 *   - API calls (/api/v1/*) -> Network-first (fresh data, fallback to cache)
 *   - Navigation requests -> Network-first with offline fallback to index.html
 */

const CACHE_NAME = 'agripulse-v1';
const OFFLINE_URL = '/index.html';

const PRECACHE_ASSETS = ['/', '/index.html', '/manifest.json', '/favicon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) =>
      Promise.all(
        cacheNames.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  // In local development, always fetch fresh from dev server to avoid stale HMR cache
  if (url.hostname === 'localhost' || url.hostname === '127.0.0.1') return;

  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request));
    return;
  }
  if (url.pathname.match(/\.(js|css|png|jpg|jpeg|svg|webp|gif|woff2?|ttf|ico)$/)) {
    event.respondWith(cacheFirst(request));
    return;
  }
  if (request.mode === 'navigate') {
    event.respondWith(navigationFetch(request));
    return;
  }
  event.respondWith(networkFirst(request));
});

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (_) {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response(JSON.stringify({ error: 'Offline' }), { status: 503, headers: { 'Content-Type': 'application/json' } });
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (_) {
    return new Response('Offline: asset not cached.', { status: 503 });
  }
}

async function navigationFetch(request) {
  try {
    return await fetch(request);
  } catch (_) {
    const cached = await caches.match(OFFLINE_URL);
    return cached || new Response('<h1>Offline</h1><p>Connect to a network to use AgriPulse.</p>', {
      status: 503,
      headers: { 'Content-Type': 'text/html' },
    });
  }
}
