// Colormaps as 256-entry RGBA8 lookup tables (polynomial fits: viridis / inferno by M. Zucker, turbo by Google;
// coolwarm: Moreland's diverging map, piecewise linear through its three anchors).

const POLY = {
    viridis: [
        [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
        [0.1050930431085774, 1.404613529898575, 1.384590162594685],
        [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
        [-4.634230498983486, -5.799100973351585, -19.33244095627987],
        [6.228269936347081, 14.17993336680509, 56.69055260068105],
        [4.776384997670288, -13.74514537774601, -65.35303263337234],
        [-5.435455855934631, 4.645852612178535, 26.3124352495832],
    ],
    inferno: [
        [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
        [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
        [11.60249308247187, -3.972853965665698, -15.9423941062914],
        [-41.70399613139459, 17.43639888205313, 44.35414519872813],
        [77.162935699427, -33.40235894210092, -81.80730925738993],
        [-71.31942824499214, 32.62606426397723, 73.20951985803202],
        [25.13112622477341, -12.24266895238567, -23.07032500287172],
    ],
};

function turbo(x) {
    const r4 = [0.13572138, 4.6153926, -42.66032258, 132.13108234], r2 = [-152.94239396, 59.28637943];
    const g4 = [0.09140261, 2.19418839, 4.84296658, -14.18503333], g2 = [4.27729857, 2.82956604];
    const b4 = [0.1066733, 12.64194608, -60.58204836, 110.36276771], b2 = [-89.90310912, 27.34824973];
    const v4 = [1, x, x * x, x * x * x], v2 = [v4[2] * v4[2], v4[3] * v4[2]];
    const d = (a, b) => a.reduce((s, ai, i) => s + ai * b[i], 0);
    return [d(r4, v4) + d(r2, v2), d(g4, v4) + d(g2, v2), d(b4, v4) + d(b2, v2)];
}

function coolwarm(x) {
    const a = [0.230, 0.299, 0.754], m = [0.865, 0.865, 0.865], b = [0.706, 0.016, 0.150];
    return x < 0.5 ? a.map((v, i) => v + (m[i] - v) * 2 * x) : m.map((v, i) => v + (b[i] - v) * (2 * x - 1));
}

function gray(x) { return [x, x, x]; }

export const COLORMAPS = ['turbo', 'viridis', 'inferno', 'coolwarm', 'gray'];

/** Uint8Array(256 * 4) */
export function lut(name) {
    const out = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
        const x = i / 255;
        let c;
        if (POLY[name]) {
            const p = POLY[name];
            c = [0, 1, 2].map((k) => p.reduceRight((acc, row) => acc * x + row[k], 0));
        } else if (name === 'turbo') c = turbo(x);
        else if (name === 'coolwarm') c = coolwarm(x);
        else c = gray(x);
        for (let k = 0; k < 3; k++) out[4 * i + k] = Math.round(255 * Math.min(1, Math.max(0, c[k])));
        out[4 * i + 3] = 255;
    }
    return out;
}

/** CSS gradient for a legend */
export function cssGradient(name) {
    const t = lut(name);
    const stops = [];
    for (let i = 0; i <= 8; i++) {
        const j = Math.round(i * 255 / 8) * 4;
        stops.push(`rgb(${t[j]},${t[j + 1]},${t[j + 2]}) ${(i * 12.5).toFixed(1)}%`);
    }
    return `linear-gradient(to right, ${stops.join(', ')})`;
}
