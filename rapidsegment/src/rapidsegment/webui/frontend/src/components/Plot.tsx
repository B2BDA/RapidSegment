import React from 'react';
import factoryNamespace from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';

// react-plotly.js/factory is CommonJS (exports.default = plotComponentFactory).
// Under Vite's ESM interop in a lazy chunk the default import can resolve to the
// namespace object ({ default: fn }) instead of the function itself, so unwrap
// both shapes defensively.
type Factory = (plotly: unknown) => React.ComponentType<any>;
const ns: any = factoryNamespace;
const factory: Factory = typeof ns === 'function' ? ns : ns?.default;

const ReactPlot: React.ComponentType<any> = factory(Plotly);

export type PlotProps = {
  data: any;
  layout?: any;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
  config?: any;
  className?: string;
};

export default function Plot(props: PlotProps) {
  return <ReactPlot {...props} />;
}
