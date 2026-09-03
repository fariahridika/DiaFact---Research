import axios from 'axios';

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:3001/api',
  timeout: 90000,
});

// Optional shared secret, injected at build time when the API requires one.
const KEY = import.meta.env.VITE_API_KEY;
if (KEY) API.defaults.headers.common['x-api-key'] = KEY;

export const getPatients   = (params) => API.get('/patients', { params });
export const getPatient    = (id)     => API.get(`/patients/${encodeURIComponent(id)}`);
export const createPatient = (data)   => API.post('/patients', data);
export const runPredict    = (data)   => API.post('/predict', data);
export const getVisit      = (id)     => API.get(`/predict/visits/${encodeURIComponent(id)}`);
export const getHealth     = ()       => API.get('/health');
