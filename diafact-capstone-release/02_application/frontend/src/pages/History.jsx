import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getPatients, getPatient } from '../api';

export default function History() {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading]   = useState(true);
  const [search,  setSearch]    = useState('');
  const [expanded, setExpanded] = useState(null);
  const [detail,   setDetail]   = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    getPatients().then(r => setPatients(r.data)).finally(() => setLoading(false));
  }, []);

  const toggleExpand = async (id) => {
    if (expanded === id) { setExpanded(null); setDetail(null); return; }
    setExpanded(id);
    const { data } = await getPatient(id);
    setDetail(data);
  };

  const filtered = patients.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    String(p.id).includes(search)
  );

  if (loading) return <div className="spinner" />;

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 800 }}>Patient History</h1>
        <input
          placeholder="Search by name or ID..."
          value={search} onChange={e => setSearch(e.target.value)}
          style={{
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: '8px', padding: '0.5rem 1rem', color: 'var(--text)',
            fontSize: '0.9rem', width: '220px',
          }}
        />
      </div>

      {filtered.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '3rem' }}>
          No patients found. Run an assessment first.
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table className="history-table">
            <thead>
              <tr>
                <th>ID</th><th>Name</th><th>Age</th><th>Gender</th>
                <th>Visits</th><th>Last Risk</th><th>Last Seen</th><th></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(p => (
                <>
                  <tr key={p.id} onClick={() => toggleExpand(p.id)}>
                    <td><span style={{ color: 'var(--primary)', fontWeight: 700 }}>#{p.id}</span></td>
                    <td style={{ fontWeight: 600 }}>{p.name}</td>
                    <td>{p.age}</td>
                    <td>{p.gender}</td>
                    <td>{p.visit_count}</td>
                    <td>
                      {p.last_risk_label
                        ? <span className={`badge badge-${p.last_risk_label.toLowerCase()}`}>
                            {p.last_risk_label} ({p.last_risk_percent?.toFixed(1)}%)
                          </span>
                        : '—'}
                    </td>
                    <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                      {p.last_visit ? new Date(p.last_visit).toLocaleDateString() : '—'}
                    </td>
                    <td>
                      <button className="btn btn-outline" style={{ padding: '0.3rem 0.8rem', fontSize: '0.82rem' }}
                        onClick={e => { e.stopPropagation(); navigate('/', { state: { prefill: p } }); }}>
                        + New Visit
                      </button>
                    </td>
                  </tr>

                  {/* Expanded visits */}
                  {expanded === p.id && detail && (
                    <tr key={`${p.id}-detail`}>
                      <td colSpan={8} style={{ padding: '0', background: 'var(--bg)' }}>
                        <div style={{ padding: '1rem 1.5rem' }}>
                          <div style={{ fontWeight: 700, marginBottom: '0.8rem', color: 'var(--text-muted)', fontSize: '0.82rem', textTransform: 'uppercase' }}>
                            All Visits
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                            {detail.visits.map(v => (
                              <div key={v.id} style={{
                                display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap',
                                background: 'var(--surface)', borderRadius: '8px', padding: '0.7rem 1rem',
                                border: '1px solid var(--border)',
                              }}>
                                <span style={{ fontWeight: 700, color: 'var(--primary)', minWidth: 70 }}>Visit #{v.visit_number}</span>
                                <span className={`badge badge-${v.risk_label.toLowerCase()}`}>{v.risk_label} {v.risk_percent.toFixed(1)}%</span>
                                <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                                  {new Date(v.visited_at).toLocaleString()}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
