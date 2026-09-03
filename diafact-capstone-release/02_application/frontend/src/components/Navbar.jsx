import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const { pathname } = useLocation();
  const nav = [
    { to: '/',        label: 'New Assessment' },
    { to: '/history', label: 'Patient History' },
  ];
  return (
    <header style={{
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '0 2rem',
      display: 'flex', alignItems: 'center', gap: '2rem', height: '60px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
        <span style={{ fontSize: '1.4rem' }}>🩺</span>
        <span style={{ fontWeight: 800, fontSize: '1.1rem', color: 'var(--primary)' }}>DiaFact</span>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>T2D Risk Assessment</span>
      </div>
      <nav style={{ display: 'flex', gap: '0.5rem', marginLeft: 'auto' }}>
        {nav.map(n => (
          <Link key={n.to} to={n.to} style={{
            padding: '0.4rem 1rem',
            borderRadius: '8px',
            textDecoration: 'none',
            fontWeight: 600,
            fontSize: '0.9rem',
            background: pathname === n.to ? 'rgba(5,150,105,0.1)' : 'transparent',
            color: pathname === n.to ? 'var(--primary)' : 'var(--text-muted)',
            transition: 'all 0.2s',
          }}>
            {n.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
