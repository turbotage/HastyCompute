"""dash_orthoslicer.py

Browser-based orthogonal slicer built with Plotly / Dash.
Mirrors the API of HastyOrthoSlicer / image_nd from orthoslicer.py but
renders inside a browser tab instead of a Matplotlib window.
"""

import threading
import time
import webbrowser

import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, ctx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_marks(n, max_ticks=8):
    """Return a {value: label} dict for a Dash Slider with at most *max_ticks*."""
    step = max(1, n // max_ticks)
    return {i: str(i) for i in range(0, n, step)}


# ---------------------------------------------------------------------------
# DashOrthoSlicer
# ---------------------------------------------------------------------------

class DashOrthoSlicer:
    """
    Orthogonal 3-D (+ optional volume dims) slicer rendered in a browser.

    Data layout (same as HastyOrthoSlicer)
    ---------------------------------------
    data[dim0, dim1, dim2, vol3, vol4]
        dim0/1/2  – spatial axes
        vol3/vol4 – optional volume / time axes (size 1 if absent)

    Views
    -----
    Sagittal  (view 0) : dim1–dim2 plane at fixed dim0
    Coronal   (view 1) : dim0–dim2 plane at fixed dim1
    Axial     (view 2) : dim0–dim1 plane at fixed dim2
    Volumes   (panel 3): mean across spatial dims; click to change volume
    """

    # (xax, yax) for each orthogonal view
    _VIEW_AXES   = [(1, 2), (0, 2), (0, 1)]
    _VIEW_LABELS = ['Sagittal', 'Coronal', 'Axial']
    _GRAPH_IDS   = ['sag-graph', 'cor-graph', 'axial-graph', 'vol-graph']
    _SLIDER_IDS  = ['slider-s0', 'slider-s1', 'slider-s2',
                    'slider-vol3', 'slider-vol4']

    def __init__(self, data, title='DashOrthoSlicer', max_clim=False, port=8050):
        # ---- complex handling ------------------------------------------------
        self._is_complex = np.iscomplexobj(data)
        if self._is_complex:
            data = np.asarray(data)
            self._phase     = np.angle(data).astype(np.float32)
            self._abs       = np.abs(data).astype(np.float32)
            self._phase_clim = [-np.pi, np.pi]
            self._abs_clim   = list(map(float, np.percentile(self._abs, (1.0, 99.0))))
            data = self._abs

        data = np.asarray(data, dtype=np.float32)

        # ---- normalise to exactly 5-D ----------------------------------------
        if data.ndim < 3:
            raise ValueError('data must have at least 3 dimensions')
        if data.ndim == 3:
            data = data[:, :, :, np.newaxis, np.newaxis]
        elif data.ndim == 4:
            data = data[:, :, :, :, np.newaxis]
        elif data.ndim > 5:
            raise ValueError('data must not have more than 5 dimensions')

        self._data      = data
        self._sizes     = list(data.shape[:3])         # [size0, size1, size2]
        self._vol_shape = data.shape[3:]               # (vol3, vol4)

        if max_clim:
            self._clim = [float(data.min()), float(data.max())]
        else:
            self._clim = list(map(float, np.percentile(data, (1.0, 99.0))))

        # volume-overview thumbnail: mean over all spatial dims → (vol3, vol4)
        self._vol_mean = np.mean(data, axis=(0, 1, 2))

        self._port  = port
        self._title = title

        self._app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._app.title = title
        self._build_layout()
        self._register_callbacks()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_idx(self):
        return [
            self._sizes[0] // 2,
            self._sizes[1] // 2,
            self._sizes[2] // 2,
            0,
            0,
        ]

    def _vol_data(self, idx3, idx4):
        return self._data[:, :, :, int(idx3), int(idx4)]

    def _make_slice_fig(self, view, idx, vol_data):
        """Return a go.Figure for one orthogonal view."""
        xax, yax = self._VIEW_AXES[view]
        size_x   = self._sizes[xax]
        size_y   = self._sizes[yax]

        # Extract 2-D slice; .T matches HastyOrthoSlicer's transpose rule
        # (order[xax] < order[yax] is always true with identity order [0,1,2])
        slc = np.rollaxis(vol_data, view)[idx[view]].T   # shape: (size_y, size_x)

        ch_x, ch_y = idx[xax], idx[yax]

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=slc,
            x=list(range(size_x)),
            y=list(range(size_y)),
            colorscale='gray',
            zmin=self._clim[0],
            zmax=self._clim[1],
            showscale=False,
            hovertemplate='<extra></extra>',
        ))

        # Crosshairs
        fig.add_shape(
            type='line',
            x0=ch_x, x1=ch_x, y0=-0.5, y1=size_y - 0.5,
            line=dict(color='lime', width=1),
        )
        fig.add_shape(
            type='line',
            x0=-0.5, x1=size_x - 0.5, y0=ch_y, y1=ch_y,
            line=dict(color='lime', width=1),
        )

        fig.update_layout(
            margin=dict(l=5, r=5, t=28, b=5),
            paper_bgcolor='#111',
            plot_bgcolor='#111',
            title=dict(
                text=self._VIEW_LABELS[view],
                font=dict(color='white', size=13),
                x=0.5,
            ),
            uirevision='constant',
            xaxis=dict(
                visible=False,
                range=[-0.5, size_x - 0.5],
                constrain='domain',
                uirevision='constant',
            ),
            yaxis=dict(
                visible=False,
                range=[-0.5, size_y - 0.5],
                scaleanchor='x',
                scaleratio=1,
                uirevision='constant',
            ),
            height=380,
            clickmode='event',
            dragmode=False,
        )
        return fig

    def _make_vol_fig(self, idx):
        """Return a go.Figure for the volume-overview panel."""
        vol3, vol4 = self._vol_shape
        idx3, idx4 = idx[3], idx[4]

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=self._vol_mean,
            x=list(range(vol4)),
            y=list(range(vol3)),
            colorscale='gray',
            showscale=False,
            hovertemplate='<extra></extra>',
        ))
        fig.add_shape(
            type='rect',
            x0=idx4 - 0.5, x1=idx4 + 0.5,
            y0=idx3 - 0.5, y1=idx3 + 0.5,
            line=dict(color='red', width=2),
            fillcolor='rgba(0,255,0,0.3)',
        )
        fig.update_layout(
            margin=dict(l=5, r=5, t=28, b=5),
            paper_bgcolor='#111',
            plot_bgcolor='#111',
            title=dict(text='Volumes', font=dict(color='white', size=13), x=0.5),
            uirevision='constant',
            xaxis=dict(visible=False, range=[-0.5, max(vol4 - 0.5, 0.5)],
                       uirevision='constant'),
            yaxis=dict(visible=False, range=[-0.5, max(vol3 - 0.5, 0.5)],
                       uirevision='constant'),
            height=380,
            clickmode='event',
            dragmode=False,
        )
        return fig

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        init   = self._init_idx()
        vd     = self._vol_data(init[3], init[4])
        config = {'displayModeBar': False, 'scrollZoom': False}
        gs     = {'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}

        # Sliders – always rendered but hidden (display:none) when size == 1
        dim_sizes  = self._sizes + list(self._vol_shape)
        dim_labels = ['Sag (dim 0)', 'Cor (dim 1)', 'Axial (dim 2)',
                      'Vol dim 3', 'Vol dim 4']
        slider_divs = []
        for sid, size, label, init_val in zip(
                self._SLIDER_IDS, dim_sizes, dim_labels, init):
            s_max   = max(size - 1, 1)
            visible = size > 1
            slider_divs.append(html.Div([
                html.Label(
                    label,
                    style={'color': '#aaa', 'fontSize': '11px', 'fontFamily': 'monospace'},
                ),
                dcc.Slider(
                    id=sid,
                    min=0, max=s_max, step=1, value=init_val,
                    marks=_make_marks(s_max + 1),
                    updatemode='drag',
                    tooltip={'placement': 'bottom', 'always_visible': False},
                ),
            ], style={} if visible else {'display': 'none'}))

        self._app.layout = html.Div([
            dcc.Store(id='state', data={'idx': init}),

            # Title
            html.Div(
                self._title,
                style={
                    'color': 'white', 'textAlign': 'center',
                    'padding': '6px 0', 'fontFamily': 'sans-serif',
                    'fontSize': '16px', 'fontWeight': 'bold',
                },
            ),

            # ── 2 × 2 grid ──────────────────────────────────────────────────
            html.Div([
                html.Div(dcc.Graph(
                    id='sag-graph', config=config,
                    figure=self._make_slice_fig(0, init, vd)), style=gs),
                html.Div(dcc.Graph(
                    id='cor-graph', config=config,
                    figure=self._make_slice_fig(1, init, vd)), style=gs),
            ]),
            html.Div([
                html.Div(dcc.Graph(
                    id='axial-graph', config=config,
                    figure=self._make_slice_fig(2, init, vd)), style=gs),
                html.Div(dcc.Graph(
                    id='vol-graph', config=config,
                    figure=self._make_vol_fig(init)), style=gs),
            ]),

            # ── Controls ────────────────────────────────────────────────────
            html.Div([
                html.Div(
                    id='idx-display',
                    style={
                        'color': '#bbb', 'fontFamily': 'monospace',
                        'fontSize': '12px', 'padding': '4px 12px',
                    },
                ),
                html.Div(slider_divs, style={'padding': '4px 16px 10px'}),
            ], style={'backgroundColor': '#1a1a1a', 'borderTop': '1px solid #333'}),

        ], style={'backgroundColor': '#000', 'minHeight': '100vh'})

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self):
        app       = self._app
        sizes     = self._sizes
        vol_shape = self._vol_shape

        # Map each slider id to (dimension-index, max-value)
        slider_dim_map = {
            'slider-s0':   (0, sizes[0] - 1),
            'slider-s1':   (1, sizes[1] - 1),
            'slider-s2':   (2, sizes[2] - 1),
            'slider-vol3': (3, vol_shape[0] - 1),
            'slider-vol4': (4, vol_shape[1] - 1),
        }

        @app.callback(
            Output('state',       'data'),
            Output('sag-graph',   'figure'),
            Output('cor-graph',   'figure'),
            Output('axial-graph', 'figure'),
            Output('vol-graph',   'figure'),
            Output('idx-display', 'children'),
            # sliders (always present)
            Input('slider-s0',   'value'),
            Input('slider-s1',   'value'),
            Input('slider-s2',   'value'),
            Input('slider-vol3', 'value'),
            Input('slider-vol4', 'value'),
            # graph clicks
            Input('sag-graph',   'clickData'),
            Input('cor-graph',   'clickData'),
            Input('axial-graph', 'clickData'),
            Input('vol-graph',   'clickData'),
            State('state',       'data'),
            prevent_initial_call=True,
        )
        def _update(s0, s1, s2, v3, v4,
                    sag_click, cor_click, axial_click, vol_click,
                    state):
            idx       = list(state['idx'])
            triggered = ctx.triggered_id

            def clip(val, lo, hi):
                return int(round(float(np.clip(val, lo, hi))))

            if triggered in slider_dim_map:
                # Only update the ONE dimension whose slider moved.
                # Updating all sliders from DOM values reverts positions set by graph clicks.
                dim, max_val = slider_dim_map[triggered]
                all_vals = [s0, s1, s2, v3, v4]
                val = all_vals[dim]
                if val is not None:
                    idx[dim] = clip(val, 0, max_val)

            elif triggered == 'sag-graph' and sag_click:
                pt = sag_click['points'][0]
                xax, yax = self._VIEW_AXES[0]
                idx[xax] = clip(pt['x'], 0, sizes[xax] - 1)
                idx[yax] = clip(pt['y'], 0, sizes[yax] - 1)

            elif triggered == 'cor-graph' and cor_click:
                pt = cor_click['points'][0]
                xax, yax = self._VIEW_AXES[1]
                idx[xax] = clip(pt['x'], 0, sizes[xax] - 1)
                idx[yax] = clip(pt['y'], 0, sizes[yax] - 1)

            elif triggered == 'axial-graph' and axial_click:
                pt = axial_click['points'][0]
                xax, yax = self._VIEW_AXES[2]
                idx[xax] = clip(pt['x'], 0, sizes[xax] - 1)
                idx[yax] = clip(pt['y'], 0, sizes[yax] - 1)

            elif triggered == 'vol-graph' and vol_click:
                pt = vol_click['points'][0]
                idx[3] = clip(pt['y'], 0, vol_shape[0] - 1)
                idx[4] = clip(pt['x'], 0, vol_shape[1] - 1)

            vd      = self._vol_data(idx[3], idx[4])
            display = (f'dim0:{idx[0]}  dim1:{idx[1]}  dim2:{idx[2]}  '
                       f'vol3:{idx[3]}  vol4:{idx[4]}')

            return (
                {'idx': idx},
                self._make_slice_fig(0, idx, vd),
                self._make_slice_fig(1, idx, vd),
                self._make_slice_fig(2, idx, vd),
                self._make_vol_fig(idx),
                display,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, blocking=True):
        """Launch the Dash server and open the slicer in the default browser.

        Parameters
        ----------
        blocking : bool
            If True (default), block until the server is stopped (Ctrl-C).
            If False, run in a background daemon thread and return it.

        Returns
        -------
        threading.Thread or None
        """
        url = f'http://127.0.0.1:{self._port}/'
        if blocking:
            webbrowser.open(url)
            self._app.run(debug=False, port=self._port, use_reloader=False)
            return None
        else:
            t = threading.Thread(
                target=self._app.run,
                kwargs=dict(debug=False, port=self._port, use_reloader=False),
                daemon=True,
            )
            t.start()
            time.sleep(0.8)   # give the server a moment to bind
            webbrowser.open(url)
            return t


# ---------------------------------------------------------------------------
# Convenience wrapper – same calling convention as HastyOrthoSlicer.image_nd
# ---------------------------------------------------------------------------

def image_nd(img, title='Image', max_clim=False, port=8050, blocking=True):
    """Display an N-D array via DashOrthoSlicer.

    Input layout (identical to the matplotlib ``image_nd`` in orthoslicer.py):

        3-D  →  (z, y, x)
        4-D  →  (vol3, z, y, x)
        5-D  →  (vol3, vol4, z, y, x)

    The array is transposed internally to (dim0=z, dim1=y, dim2=x, vol3, vol4)
    with the z-axis flipped, matching the original behaviour exactly.

    Parameters
    ----------
    img       : array-like
    title     : str
    max_clim  : bool
    port      : int
    blocking  : bool   block until Ctrl-C when True; run in background thread otherwise

    Returns
    -------
    threading.Thread or None
    """
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[np.newaxis, np.newaxis, ...]
    elif img.ndim == 4:
        img = img[np.newaxis, ...]
    if img.ndim != 5:
        raise ValueError('image_nd expects 3-5 dimensional data')

    # (vol3, vol4, z, y, x) → (z, y, x, vol3, vol4), z-axis flipped
    dataf = np.flip(img.transpose((2, 3, 4, 0, 1)), axis=2)

    slicer = DashOrthoSlicer(dataf, title=title, max_clim=max_clim, port=port)
    return slicer.show(blocking=blocking)


if __name__ == '__main__':
    # Quick smoke-test with synthetic data
    data = np.random.rand(64, 64, 64).astype(np.float32)
    image_nd(data, title='Smoke test', blocking=True)