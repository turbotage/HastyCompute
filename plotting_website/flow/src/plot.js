// Minimal canvas line plot: several series over frames, a frame cursor, axes with ticks.

export class Plot {
    constructor(canvas) {
        this.canvas = canvas;
        this.series = [];
        this.cursor = null;
        this.xlabel = '';
        this.ylabel = '';
    }

    /** series: [{name, x: number[], y: number[], color}] */
    set(series, { xlabel = '', ylabel = '' } = {}) {
        this.series = series;
        this.xlabel = xlabel;
        this.ylabel = ylabel;
        this.draw();
    }

    draw() {
        const c = this.canvas;
        const dpr = window.devicePixelRatio || 1;
        const W = c.clientWidth, H = c.clientHeight;
        if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
        const g = c.getContext('2d');
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        if (!this.series.length) return;
        const xs = this.series.flatMap((s) => s.x), ys = this.series.flatMap((s) => s.y);
        let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
        let [y0, y1] = [Math.min(0, ...ys), Math.max(0, ...ys)];
        if (y1 - y0 < 1e-9) y1 = y0 + 1;
        const pad = 0.08 * (y1 - y0);
        y0 -= pad; y1 += pad;
        const L = 46, R = 8, T = 8, B = 26;
        const px = (x) => L + ((x - x0) / Math.max(x1 - x0, 1e-9)) * (W - L - R);
        const py = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
        g.strokeStyle = '#333'; g.fillStyle = '#888'; g.font = '10px sans-serif'; g.lineWidth = 1;
        const step = niceStep((y1 - y0) / 4);
        for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
            g.beginPath(); g.moveTo(L, py(y)); g.lineTo(W - R, py(y)); g.stroke();
            g.fillText(fmt(y), 4, py(y) + 3);
        }
        g.strokeStyle = '#666';
        g.beginPath(); g.moveTo(L, py(0)); g.lineTo(W - R, py(0)); g.stroke();
        const xstep = niceStep((x1 - x0) / 5);
        for (let x = Math.ceil(x0 / xstep) * xstep; x <= x1 + 1e-9; x += xstep) g.fillText(fmt(x), px(x) - 8, H - B + 12);
        g.fillText(this.xlabel, W / 2 - 20, H - 3);
        g.save(); g.translate(10, T + 40); g.rotate(-Math.PI / 2); g.restore();
        g.fillText(this.ylabel, L + 4, T + 10);
        for (const s of this.series) {
            g.strokeStyle = s.color || '#4fc3f7'; g.lineWidth = 1.6;
            g.beginPath();
            s.x.forEach((x, i) => (i ? g.lineTo(px(x), py(s.y[i])) : g.moveTo(px(x), py(s.y[i]))));
            g.stroke();
        }
        if (this.cursor !== null) {
            g.strokeStyle = '#ffd23f'; g.lineWidth = 1;
            g.beginPath(); g.moveTo(px(this.cursor), T); g.lineTo(px(this.cursor), H - B); g.stroke();
        }
    }

    setCursor(x) { this.cursor = x; this.draw(); }
}

function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-12))));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

function fmt(v) {
    const a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
}
