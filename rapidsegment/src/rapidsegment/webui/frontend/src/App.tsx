import React from 'react';
import { Navigate, Route, Routes, NavLink, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import M1 from './pages/M1DataLoader';
import M2 from './pages/M2Workbench';
import M3 from './pages/M3Execution';
import M4 from './pages/M4Results';
import M5 from './pages/M5Leaderboard';
import M6 from './pages/M6Arena';
import Exit from './pages/Exit';
import { post } from './api';

const NAV = [
  { to: '/', label: 'Home', icon: '⌂', end: true },
  { to: '/m1', label: '1 · Data Loader & Profiling', icon: '⤓' },
  { to: '/m2', label: '2 · Workbench', icon: '⚙' },
  { to: '/m3', label: '3 · Execution & Artifacts', icon: '▲' },
  { to: '/m4', label: '4 · Results Dashboard', icon: '▦' },
  { to: '/m5', label: '5 · Leaderboard', icon: '🏆' },
  { to: '/m6', label: '6 · Arena', icon: '⚔' },
];

function titleFor(path: string): string {
  return NAV.find((n) => (n.end ? path === n.to : path.startsWith(n.to)))?.label ?? 'RapidSegment';
}

export default function App() {
  const loc = useLocation();
  React.useEffect(() => {
    document.title = titleFor(loc.pathname);
  }, [loc.pathname]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">RAPIDSEGMENT</div>
        <div className="brand-sub">No-Code Segmentation Platform</div>
        <div className="nav-group">Modules</div>
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <span className="nav-icon">{n.icon}</span> {n.label}
          </NavLink>
        ))}
        <div className="nav-exit">
          <NavLink to="/exit" className="nav-item">⏻ Exit UI</NavLink>
          <div className="caption">Stops the local app server.</div>
        </div>
      </aside>

      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/m1" element={<M1 />} />
          <Route path="/m2" element={<M2 />} />
          <Route path="/m3" element={<M3 />} />
          <Route path="/m4" element={<M4 />} />
          <Route path="/m5" element={<M5 />} />
          <Route path="/m6" element={<M6 />} />
          <Route path="/exit" element={<Exit />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
