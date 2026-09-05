import React from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';

const MODULES = [
  { to: '/m1', num: '1', name: 'Data Loader & Profiling', icon: '⤓', desc: 'Load a dataset, auto-profile, and materialize it as the active dataset.' },
  { to: '/m2', num: '2', name: 'Workbench', icon: '⚙', desc: 'Assemble segmentation parameters, presets and validation.' },
  { to: '/m3', num: '3', name: 'Execution & Artifacts', icon: '▲', desc: 'Run the segmentation engine and track progress step by step.' },
  { to: '/m4', num: '4', name: 'Results Dashboard', icon: '▦', desc: 'Scorecards, insights and diagnostic charts for the active experiment.' },
  { to: '/m5', num: '5', name: 'Leaderboard', icon: '🏆', desc: 'Compare every experiment by weighted score and segments.' },
  { to: '/m6', num: '6', name: 'Arena', icon: '⚔', desc: 'Head-to-head battle between two experiments.' },
];

export default function Home() {
  const nav = useNavigate();
  const [srv, setSrv] = React.useState<string>('checking…');

  React.useEffect(() => {
    api<{ status: string; suite_dir: string }>('/api/health')
      .then((h) => setSrv(`${h.status} · ${h.suite_dir}`))
      .catch((e) => setSrv(`unreachable (${e.message})`));
  }, []);

  return (
    <div>
      <h1>RapidSegment — No-Code Segmentation Platform</h1>
      <div className="caption">Choose a module from the sidebar.</div>
      <div className="row" style={{ margin: '6px 0 14px' }}>
        <span className="chip">server: {srv}</span>
        <span className="chip">frontend: react + ts + vite</span>
      </div>

      <div className="grid2">
        {MODULES.map((m) => (
          <div key={m.to} className="card" style={{ cursor: 'pointer' }} onClick={() => nav(m.to)}>
            <div className="spread">
              <h3 style={{ margin: 0 }}>{m.icon} · {m.num} · {m.name}</h3>
              <span className="chip">open →</span>
            </div>
            <div className="caption" style={{ marginTop: 6 }}>{m.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
