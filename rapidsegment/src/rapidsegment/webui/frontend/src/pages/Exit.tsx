import React from 'react';
import { post } from '../api';

export default function Exit() {
  const [state, setState] = React.useState<'idle' | 'stopping' | 'done'>('idle');

  const stop = async () => {
    setState('stopping');
    try {
      await post('/api/exit');
    } catch {
      /* server likely stopped mid-request */
    }
    setState('done');
  };

  return (
    <div>
      <h1>Exit UI</h1>
      <div className="card">
        <p>This stops the local app server and closes the session, mirroring the Streamlit "Exit UI" button.</p>
        <button className="primary" onClick={stop} disabled={state !== 'idle'}>
          {state === 'idle' ? 'Stop server' : state === 'stopping' ? 'Stopping…' : 'Server stopped'}
        </button>
        {state === 'done' && (
          <div className="alert alert-warning" style={{ marginTop: 12 }}>
            Server is shutting down. You can safely close this tab — refresh will not bring it back.
          </div>
        )}
      </div>
    </div>
  );
}
