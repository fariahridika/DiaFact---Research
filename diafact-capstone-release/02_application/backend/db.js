'use strict';

const mysql = require('mysql2/promise');

const password = process.env.DB_PASS || '';

if (!password && process.env.NODE_ENV === 'production') {
  console.error('Refusing to start: DB_PASS is empty in production. '
              + 'A blank MySQL root password exposes every patient record.');
  process.exit(1);
}
if (!password) {
  console.warn('DB password is empty (XAMPP default). Acceptable for a local '
             + 'demo only -- set DB_PASS before any shared deployment.');
}

const pool = mysql.createPool({
  host:     process.env.DB_HOST || '127.0.0.1',
  port:     Number(process.env.DB_PORT || 3306),
  user:     process.env.DB_USER || 'root',
  password,
  database: process.env.DB_NAME || 'diafact',
  waitForConnections: true,
  connectionLimit: Number(process.env.DB_POOL || 10),
  queueLimit: 0,
  // Never let a caller-supplied string be interpreted as several statements.
  multipleStatements: false,
  charset: 'utf8mb4_general_ci',
  timezone: 'Z',
});

module.exports = pool;
