'use strict';

require('dotenv').config();

const express   = require('express');
const cors      = require('cors');
const helmet    = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();

// Behind a reverse proxy the client IP arrives in X-Forwarded-For; trust only
// one hop so the rate limiter cannot be defeated by a spoofed header chain.
app.set('trust proxy', 1);

app.disable('x-powered-by');
app.use(helmet());

// The previous build used bare cors(), which reflects any origin. This service
// holds patient records, so the allowlist is explicit.
const ALLOWED = (process.env.ALLOWED_ORIGINS
  || 'http://localhost:5173,http://127.0.0.1:5173')
  .split(',').map(s => s.trim()).filter(Boolean);

app.use(cors({
  origin(origin, cb) {
    // Same-origin/curl requests arrive with no Origin header.
    if (!origin || ALLOWED.includes(origin)) return cb(null, true);
    return cb(new Error('Origin not allowed'));
  },
  methods: ['GET', 'POST'],
  credentials: false,
}));

// A prediction payload is a few hundred bytes; cap it well below that ceiling.
app.use(express.json({ limit: '32kb' }));

app.use(rateLimit({
  windowMs: 60 * 1000,
  max: Number(process.env.RATE_LIMIT_PER_MIN || 60),
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please slow down.' },
}));

/**
 * Optional shared-secret gate. Off by default so the local demo still runs,
 * but the moment this is exposed beyond localhost API_KEY must be set --
 * patient records are otherwise readable by anyone who can reach the port.
 */
const API_KEY = process.env.API_KEY || '';
if (API_KEY) {
  app.use('/api', (req, res, next) => {
    if (req.path === '/health') return next();
    const given = req.get('x-api-key') || '';
    // Constant-time compare to avoid leaking the key through timing.
    const a = Buffer.from(given);
    const b = Buffer.from(API_KEY);
    const ok = a.length === b.length && require('crypto').timingSafeEqual(a, b);
    if (!ok) return res.status(401).json({ error: 'Unauthorized' });
    next();
  });
  console.log('API key authentication ENABLED');
} else {
  console.warn('API key authentication DISABLED (set API_KEY to enable). '
             + 'Do not expose this service beyond localhost.');
}

app.use('/api/patients', require('./routes/patients'));
app.use('/api/predict',  require('./routes/predict'));

app.get('/api/health', async (req, res) => {
  const out = { status: 'ok', ml: 'unreachable', db: 'unreachable' };
  try {
    const ml = await require('./mlClient').health();
    out.ml = ml.status || 'ok';
  } catch { /* left as unreachable */ }
  try {
    await require('./db').query('SELECT 1');
    out.db = 'ok';
  } catch { /* left as unreachable */ }
  // Both dependencies are required to serve a prediction, so report degraded
  // unless both answer.
  if (out.ml !== 'ok' || out.db !== 'ok') out.status = 'degraded';
  res.status(out.status === 'ok' ? 200 : 503).json(out);
});

app.use((req, res) => res.status(404).json({ error: 'Not found' }));

// Terminal error handler. Internal messages are logged, never returned:
// stack traces and driver errors are a disclosure channel.
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  if (err && err.message === 'Origin not allowed') {
    return res.status(403).json({ error: 'Origin not allowed' });
  }
  if (err && err.type === 'entity.too.large') {
    return res.status(413).json({ error: 'Request body too large' });
  }
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: 'Malformed JSON body' });
  }
  // A dead or unreachable database is an availability problem, not a bug in
  // the request: 503 tells the caller to retry rather than to change the input.
  const DB_DOWN = ['ECONNREFUSED', 'PROTOCOL_CONNECTION_LOST', 'ETIMEDOUT',
                   'ER_ACCESS_DENIED_ERROR', 'ER_BAD_DB_ERROR'];
  if (err && (DB_DOWN.includes(err.code) || err.fatal)) {
    console.error('Dependency unavailable:', err.code);
    return res.status(503).json({ error: 'A required service is unavailable.' });
  }
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Last-resort net. Express 4 does not catch rejections thrown by async
// handlers, and a single one must not be able to kill a service holding
// patient records. Log loudly, stay up.
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled promise rejection:', reason);
});
process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err);
});

const PORT = Number(process.env.PORT || 3001);
const HOST = process.env.HOST || '127.0.0.1';
app.listen(PORT, HOST, () =>
  console.log(`DiaFact backend running on http://${HOST}:${PORT}`));
