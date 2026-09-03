'use strict';

const axios = require('axios');

// Loopback by default: the ML service has no authentication of its own.
const ML_BASE = process.env.ML_URL || 'http://127.0.0.1:5001';

const client = axios.create({
  baseURL: ML_BASE,
  timeout: Number(process.env.ML_TIMEOUT_MS || 60000),
  maxContentLength: 2 * 1024 * 1024,
  maxBodyLength: 256 * 1024,
  headers: { 'Content-Type': 'application/json' },
});

async function predict(payload) {
  const res = await client.post('/predict', payload);
  return res.data;
}

async function health() {
  const res = await client.get('/health', { timeout: 5000 });
  return res.data;
}

module.exports = { predict, health };
