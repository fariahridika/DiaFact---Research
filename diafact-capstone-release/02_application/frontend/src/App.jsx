import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar       from './components/Navbar';
import PatientForm  from './pages/PatientForm';
import Results      from './pages/Results';
import History      from './pages/History';
import './index.css';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/"         element={<PatientForm />} />
            <Route path="/results"  element={<Results />} />
            <Route path="/history"  element={<History />} />
            <Route path="*"         element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
