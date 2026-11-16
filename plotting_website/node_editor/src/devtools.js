// Dev-only checks, loaded by main.js when the URL has ?gallery=1 or ?perf=N (never in normal use).
//
//   ?gallery=1   one node of every registered type laid out in a grid, plus a numeric layout report in <pre id="selftest">:
//                  - size_lt_min: node.size smaller than node.computeSize() (widgets / slots spill out of the body)
//                  - status_overlap: execution status bar (bottom 22 px, drawn when _execStatus is set) overlaps the widget stack
//                  - info_overlap: the 10 px info line some nodes draw at size[1]-42 overlaps the widget stack
//   ?perf=N      N nodes (mixed types) wired in chains, then redraw / pan / drag timings in <pre id="selftest">

const WIDGET_H = 20, WIDGET_GAP = 4, STATUS_H = 22;

/** y (node-body coordinates) where the bottom-most widget ends: slots first, then widgets, as litegraph lays them out */
function widgetsBottom(node) {
    const LG = window.LiteGraph;
    const slotRows = Math.max(node.inputs?.length ?? 0, node.outputs?.length ?? 0);
    let y = slotRows * (LG.NODE_SLOT_HEIGHT ?? 20) + 4;
    if (node.widgets_start_y != null) y = node.widgets_start_y;
    return y + (node.widgets?.length ?? 0) * (WIDGET_H + WIDGET_GAP);
}

export function layoutReport(graph) {
    const rows = [];
    for (const n of graph._nodes) {
        if (n.type === 'basic/note' || n.type === 'Note') continue;
        const min = n.computeSize();
        const wb = widgetsBottom(n);
        const hasStatus = typeof n.executeAsync === 'function';
        const r = { node: n.type, size: n.size.slice(), min: min.map(Math.round), widgetsBottom: Math.round(wb), problems: [] };
        if (n.size[0] + 0.5 < min[0] || n.size[1] + 0.5 < min[1]) r.problems.push('size_lt_min');
        if (hasStatus && n.size[1] - STATUS_H < wb - 2) r.problems.push('status_overlap');
        if (hasStatus && /_info/.test(String(n.onDrawForeground)) && n.size[1] - 42 < wb) r.problems.push('info_overlap');
        rows.push(r);
    }
    return rows;
}

export function gallery(graph, canvas) {
    const LG = window.LiteGraph;
    const types = Object.keys(LG.registered_node_types);
    let x = 40, y = 40, rowH = 0;
    for (const t of types) {
        const n = LG.createNode(t);
        if (!n) continue;
        graph.add(n);   // adding fits the node (onAdded), so size is final only now
        if (x + n.size[0] > 1500) { x = 40; y += rowH + 50; rowH = 0; }
        n.pos = [x, y];
        x += n.size[0] + 40;
        rowH = Math.max(rowH, n.size[1]);
    }
    canvas.setDirty(true, true);
    return layoutReport(graph);
}

/** show every node in each execution state, to look at the status bar / info line against the widgets */
export function gallery_states(graph, canvas) {
    const states = ['running', 'done', 'error'];
    graph._nodes.forEach((n, i) => {
        if (typeof n.executeAsync !== 'function') return;
        n._execStatus = states[i % 3];
        n._execResult = n._execStatus === 'done' ? ['0123456789abcdef0123456789abcdef'] : null;
        n._execMeta = { dtype: 'f32', shape: [64, 256, 256], device: 'cpu' };
        n._execError = 'something went wrong here';
        n._info = n.title === 'Run Model' ? 'onnx/cpu 12.3 s' : n._info;
    });
    canvas.setDirty(true, true);
}

/** mean time of a synchronous full canvas draw (rAF-paced timing would only show the 60 Hz cap) */
const frames = async (canvas, n) => {
    const t0 = performance.now();
    for (let i = 0; i < n; i++) canvas.draw(true, true);
    return (performance.now() - t0) / n;
};

export async function perf(graph, canvas, count) {
    const LG = window.LiteGraph;
    const out = { nodes: count };
    const types = ['Math/Add', 'Math/Neg', 'FFT/FFT', 'Create/Rand Tensor', 'Math/Sub', 'Viz/Compress UI16', 'Models/Run Model'];
    const t0 = performance.now();
    const cols = Math.ceil(Math.sqrt(count * 1.5));
    const list = [];
    for (let i = 0; i < count; i++) {
        const n = LG.createNode(types[i % types.length]);
        n.pos = [(i % cols) * 300, Math.floor(i / cols) * 320];
        graph.add(n);
        list.push(n);
    }
    for (let i = 1; i < list.length; i++) { // chain: out 0 of the previous node into input 0, where both exist
        if (list[i].inputs?.length && list[i - 1].outputs?.length) list[i - 1].connect(0, list[i], 0);
    }
    out.add_ms = Math.round(performance.now() - t0);
    out.build_ms = out.add_ms;
    out.links = Object.keys(graph.links).length;
    // fit the whole graph on screen, then in the middle (zoomed in)
    const w = canvas.canvas.width, h = canvas.canvas.height;
    const maxX = cols * 300, maxY = Math.ceil(count / cols) * 320;
    const fit = Math.min(w / maxX, h / maxY);
    canvas.ds.scale = fit; canvas.ds.offset = [0, 0];
    await frames(canvas, 3);
    out.frame_ms_all_visible = +(await frames(canvas, 30)).toFixed(2);
    canvas.ds.scale = 1; canvas.ds.offset = [-maxX / 2 + w / 2, -maxY / 2 + h / 2];
    await frames(canvas, 3);
    out.frame_ms_zoom1 = +(await frames(canvas, 30)).toFixed(2);
    // with all nodes marked done + meta, as after a run
    for (const n of list) { n._execStatus = 'done'; n._execResult = ['0123456789abcdef0123456789abcdef']; n._execMeta = { dtype: 'f32', shape: [8, 8], device: 'cpu' }; }
    canvas.ds.scale = fit; canvas.ds.offset = [0, 0];
    out.frame_ms_all_done = +(await frames(canvas, 30)).toFixed(2);
    // graph.serialize / configure (Save / Load button)
    let t = performance.now();
    const ser = JSON.stringify(graph.serialize());
    out.serialize_ms = Math.round(performance.now() - t);
    out.serialized_kb = Math.round(ser.length / 1024);
    t = performance.now();
    const g2 = new LG.LGraph();
    g2.configure(JSON.parse(ser));
    out.configure_ms = Math.round(performance.now() - t);
    // the idle cost: graph.start() runs onExecute of every node each tick, the loop below counts how long a tick takes
    t = performance.now();
    for (let i = 0; i < 20; i++) graph.runStep(1);
    out.runStep_ms = +((performance.now() - t) / 20).toFixed(2);
    out.heap_mb = performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null;
    return out;
}

/** ?flowtest=1: Rand tensors -> Flow Make -> Flow Info / PC-MRA / Flow Plane / Flow Viewer, run against the server */
export async function flowtest(graph, canvas, runGraph, client) {
    const LG = window.LiteGraph;
    const add = (type, props = {}, pos = [0, 0]) => { const n = LG.createNode(type); Object.assign(n.properties, props); n.pos = pos; graph.add(n); return n; };
    const vel = add('Create/Rand Tensor', { shape: '6,3,24,24,24', dtype: 'f32' }, [40, 40]);
    const mag = add('Create/Ones Tensor', { shape: '6,24,24,24', dtype: 'f32' }, [40, 200]);
    const make = add('Flow/Flow Make', { spacing: '2,2,2', dt: 0.05, venc: 1.5 }, [320, 60]);
    const info = add('Flow/Flow Info', {}, [640, 40]);
    const pc = add('Flow/PC-MRA', {}, [640, 140]);
    const plane = add('Flow/Flow Plane', { normal: '0,0,1', radius: 20, grid: 32 }, [640, 240]);
    const view = add('Flow/Flow Viewer', {}, [960, 40]);
    const slicer = add('Viz/Ortho Slicer', {}, [960, 200]);
    vel.connect(0, make, 0); mag.connect(0, make, 1);
    make.connect(0, info, 0); info.connect(0, view, 0); info.connect(0, plane, 0); make.connect(0, pc, 0); pc.connect(0, slicer, 0);
    await runGraph(graph, canvas);
    const nodes = [vel, mag, make, info, pc, plane, view, slicer];
    const errors = nodes.filter((n) => n._execStatus !== 'done').map((n) => `${n.title}: ${n._execError}`);
    const loader = add('Flow/Flow Load', {}, [320, 260]);
    await runGraph(graph, canvas);
    const pcMeta = pc._execResult?.[0] ? await client.info(pc._execResult[0]) : null;
    canvas.setDirty(true, true);
    return { ok: errors.length === 0 && loader._execStatus === 'done', errors, loader: [loader._execStatus, loader._execError, loader._info],
             info: info._info, plane: plane._info, plane_summary_keys: Object.keys(plane._execResult?.[2] ?? {}), viewer: view._uuid, pc_meta: pcMeta?.metadata };
}

/**
 * ?runtest=1: build a graph with a browser-placed node in the middle, save it, submit a run and stop at "waiting for the
 * browser" without serving it (the tab "closes"). Reports the graph and run ids. ?graph=<id>&runtest=2 (a fresh page):
 * re-attaches through the normal open path, presses Continue and waits for the end.
 */
export async function runtest1(graph, canvas, h) {
    const LG = window.LiteGraph;
    const add = (type, props = {}, pos = [0, 0]) => { const n = LG.createNode(type); Object.assign(n.properties, props); n.pos = pos; graph.add(n); return n; };
    const vel = add('Create/Rand Tensor', { shape: '6,3,24,24,24', dtype: 'f32' }, [40, 40]);
    const mag = add('Create/Ones Tensor', { shape: '6,24,24,24', dtype: 'f32' }, [40, 220]);
    const make = add('Flow/Flow Make', { spacing: '2,2,2', dt: 0.05, venc: 1.5 }, [320, 60]);
    const info = add('Flow/Flow Info', {}, [640, 40]);
    const plane = add('Flow/Flow Plane', { normal: '0,0,1', radius: 20, grid: 32 }, [640, 140]);
    const rnd = add('Create/Rand Tensor', { shape: '4,8,8', dtype: 'c32' }, [40, 400]);
    const fft = add('FFT/FFT', { norm: 'ortho', axes: '-1,-2' }, [320, 400]);
    const neg = add('Math/Neg', { placement: 'client' }, [640, 400]);          // runs in the browser
    const after = add('Math/Abs', {}, [900, 400]);                              // server again, after the browser step
    vel.connect(0, make, 0); mag.connect(0, make, 1); make.connect(0, info, 0); info.connect(0, plane, 0);
    rnd.connect(0, fft, 0); fft.connect(0, neg, 0); neg.connect(0, after, 0);
    for (const n of graph._nodes) n.tint?.();
    document.getElementById('graph-name').value = 'runtest';
    await h.save(graph);
    const { plan, errors } = await h.compile(graph);
    if (errors.length) return { ok: false, errors: errors.map((e) => `${e.node.title}: ${e.error}`) };
    const { run } = await h.call('run_submit', { graph: h.doc.id, name: 'runtest', plan });
    const ctl = new h.RunController(graph, canvas, { banner() {}, onStatus() {} });
    await ctl.attach(run, { auto: false });
    for (let i = 0; i < 100 && ctl.last?.status === 'running'; i++) { await new Promise((r) => setTimeout(r, 200)); }
    const nodes = Object.fromEntries(Object.entries(ctl.last.nodes).map(([k, v]) => [v.title + '#' + k, v.status]));
    const wl = await h.call('graph_list', {});
    return { ok: ctl.last.status === 'waiting_client', status: ctl.last.status, graph: h.doc.id, run, nodes, listed: wl.graphs.find((g) => g.id === h.doc.id) };
}

export async function runtest2(graph, canvas) {
    const ctl = window._runCtl;
    const wait = async (f, n = 100) => { for (let i = 0; i < n; i++) { if (f()) return true; await new Promise((r) => setTimeout(r, 200)); } return false; };
    if (!await wait(() => ctl.last?.status === 'waiting_client')) return { ok: false, stage: 'attach', status: ctl.last?.status };
    const painted = graph._nodes.map((n) => `${n.title}:${n._execStatus}`);
    ctl.proceed();
    if (!await wait(() => ctl.last?.status === 'done')) return { ok: false, stage: 'continue', status: ctl.last?.status, nodes: Object.values(ctl.last?.nodes ?? {}).map((n) => `${n.title}:${n.status}:${n.error ?? ''}`) };
    await new Promise((r) => setTimeout(r, 800));
    canvas.setDirty(true, true);
    return { ok: true, painted_while_waiting: painted, final: graph._nodes.map((n) => `${n.title}:${n._execStatus}${n._info ? ' [' + n._info + ']' : ''}`) };
}

/** ?graph=<id>&runtest=3: cancel while waiting for the browser, then two identical server-only runs (second one is cached) */
export async function runtest3(graph, canvas, h) {
    const wait = async (ctl, f, n = 150) => { for (let i = 0; i < n; i++) { if (f()) return true; await new Promise((r) => setTimeout(r, 200)); } return false; };
    const out = {};
    let { plan } = await h.compile(graph);
    const ctl = new h.RunController(graph, canvas, { banner() {}, onStatus() {} });
    let { run } = await h.call('run_submit', { graph: h.doc.id, plan });
    await ctl.attach(run);
    await wait(ctl, () => ctl.last?.status === 'waiting_client');
    await ctl.cancel();
    await wait(ctl, () => ctl.last?.status === 'cancelled');
    out.cancel = { status: ctl.last.status, nodes: Object.values(ctl.last.nodes).map((n) => `${n.title}:${n.status}`) };
    for (const n of graph._nodes) if (n.properties.placement === 'client') n.properties.placement = 'server';
    ({ plan } = await h.compile(graph));
    const runs = [];
    for (let k = 0; k < 2; k++) {
        ({ run } = await h.call('run_submit', { graph: h.doc.id, plan }));
        await ctl.attach(run);
        await wait(ctl, () => ctl.last?.status === 'done');
        runs.push({ status: ctl.last.status, cached: Object.values(ctl.last.nodes).filter((n) => n.cached).length, total: Object.keys(ctl.last.nodes).length });
    }
    out.runs = runs;
    out.ok = out.cancel.status === 'cancelled' && runs[0].status === 'done' && runs[1].cached > 0;
    return out;
}

/** ?meshtest=1: synthetic Y-shaped vessel tree mask → Mesh From Mask → Smooth → Sample (a gradient volume) → Info / Viewer, on the server */
export async function meshtest(graph, canvas, h, client, HTensor) {
    const LG = window.LiteGraph;
    const [nx, ny, nz] = [64, 64, 64];
    const mask = new Uint8Array(nx * ny * nz), grad = new Float32Array(nx * ny * nz);
    const seg = (a, b, r) => {                      // capsule a→b, radius r (voxels)
        const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], L2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
        for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
            const t = Math.min(1, Math.max(0, ((i - a[0]) * d[0] + (j - a[1]) * d[1] + (k - a[2]) * d[2]) / L2));
            const q = [a[0] + t * d[0] - i, a[1] + t * d[1] - j, a[2] + t * d[2] - k];
            if (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 <= r * r) mask[(k * ny + j) * nx + i] = 1;
        }
    };
    seg([32, 32, 4], [32, 32, 32], 5); seg([32, 32, 32], [14, 20, 58], 3.5); seg([32, 32, 32], [50, 44, 58], 3.5);
    seg([8, 8, 8], [10, 10, 12], 2);                // a separate small blob: removed by "largest"
    for (let q = 0; q < grad.length; q++) grad[q] = (q % nx);
    const affine = [[0.8, 0, 0, -20], [0, 0.8, 0, -20], [0, 0, 1.0, -30], [0, 0, 0, 1]];
    const meta = { spacing_mm: [0.8, 0.8, 1.0], affine, axes: ['x', 'y', 'z'] };
    const idMask = await client.put(new HTensor('uint8', [nz, ny, nx], mask), { name: 'tree mask', metadata: meta });
    const idGrad = await client.put(new HTensor('float32', [nz, ny, nx], grad), { name: 'x gradient', metadata: meta });
    const add = (type, props = {}, pos = [0, 0]) => { const n = LG.createNode(type); Object.assign(n.properties, props); n.pos = pos; graph.add(n); return n; };
    const lm = add('IO/Load Tensor', { uuid: idMask }, [40, 40]), lg = add('IO/Load Tensor', { uuid: idGrad }, [40, 160]);
    const mm = add('Geometry/Mesh From Mask', { blur: 1, smooth: 6, keep: 'largest' }, [330, 40]);
    const sm = add('Geometry/Mesh Smooth', { iterations: 4 }, [680, 40]);
    const sa = add('Geometry/Mesh Sample', { name: 'x_voxel' }, [680, 140]);
    const info = add('Geometry/Mesh Info', {}, [960, 140]);
    const view = add('Geometry/Mesh Viewer', { scalar: 'x_voxel' }, [960, 40]);
    lm.connect(0, mm, 0); mm.connect(0, sm, 0); sm.connect(0, sa, 0); lg.connect(0, sa, 1); sa.connect(0, info, 0); info.connect(0, view, 0);
    const { plan, errors: perr } = await h.compile(graph);
    if (perr.length) return { ok: false, errors: perr.map((e) => `${e.node.title}: ${e.error}`) };
    const { run } = await h.call('run_submit', { plan });
    const ctl = new h.RunController(graph, canvas, { banner() {}, onStatus() {} });
    await ctl.attach(run);
    for (let i = 0; i < 200 && ctl.last?.status === 'running'; i++) await new Promise((r) => setTimeout(r, 200));
    await new Promise((r) => setTimeout(r, 400));
    const nodes = [lm, lg, mm, sm, sa, info, view];
    const errors = nodes.filter((n) => n._execStatus !== 'done').map((n) => `${n.title}: ${n._execError}`);
    return { ok: errors.length === 0 && ctl.last?.status === 'done', errors, mask_info: mm._info, final_info: info._info, viewer_ids: view._ids, viewer_scalar: view.properties.scalar };
}

/** ?geomtest=1: decimate / cut / centerline / sequence nodes, Mesh Load + Save steps in the browser, a flow value with centerlines */
export async function geomtest(graph, canvas, h, client, HTensor, parseMesh) {
    const LG = window.LiteGraph;
    const out = { errors: [] };
    const add = (type, props = {}, pos = [0, 0]) => { const n = LG.createNode(type); Object.assign(n.properties, props); n.pos = pos; graph.add(n); return n; };
    const runAll = async (name) => {
        const { plan, errors } = await h.compile(graph);
        if (errors.length) { out.errors.push(...errors.map((e) => `${name}: ${e.node.title}: ${e.error}`)); return null; }
        const { run } = await h.call('run_submit', { plan });
        const ctl = new h.RunController(graph, canvas, { banner() {}, onStatus() {} });
        await ctl.attach(run, { auto: true });
        for (let i = 0; i < 300 && ['running', 'waiting_client'].includes(ctl.last?.status); i++) await new Promise((r) => setTimeout(r, 200));
        await new Promise((r) => setTimeout(r, 400));
        if (ctl.last?.status !== 'done') out.errors.push(`${name}: run ${ctl.last?.status}: ` + Object.values(ctl.last?.nodes ?? {}).filter((n) => n.error).map((n) => `${n.title}: ${n.error}`).join('; '));
        return ctl;
    };
    // data: Y tree (mask) and a growing sphere (4-D mask) with a header
    const [nx, ny, nz] = [64, 64, 64];
    const mask = new Uint8Array(nx * ny * nz);
    const seg = (a, b, r) => {
        const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], L2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
        for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
            const t = Math.min(1, Math.max(0, ((i - a[0]) * d[0] + (j - a[1]) * d[1] + (k - a[2]) * d[2]) / L2));
            const q = [a[0] + t * d[0] - i, a[1] + t * d[1] - j, a[2] + t * d[2] - k];
            if (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 <= r * r) mask[(k * ny + j) * nx + i] = 1;
        }
    };
    seg([32, 32, 4], [32, 32, 32], 5); seg([32, 32, 32], [14, 20, 58], 3.5); seg([32, 32, 32], [50, 44, 58], 3.5);
    const meta = { spacing_mm: [0.8, 0.8, 1.0], affine: [[0.8, 0, 0, -20], [0, 0.8, 0, -20], [0, 0, 1.0, -30], [0, 0, 0, 1]], axes: ['x', 'y', 'z'] };
    const idMask = await client.put(new HTensor('uint8', [nz, ny, nx], mask), { name: 'tree mask', metadata: meta });
    const T = 4, n4 = 32, seq = new Uint8Array(T * n4 ** 3);
    for (let f = 0; f < T; f++) for (let q = 0; q < n4 ** 3; q++) {
        const i = q % n4, j = Math.floor(q / n4) % n4, k = Math.floor(q / n4 ** 2), c = 15.5, r = 6 + 2 * f;
        seq[f * n4 ** 3 + q] = (i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2 <= r * r ? 1 : 0;
    }
    const idSeq = await client.put(new HTensor('uint8', [T, n4, n4, n4], seq), { name: 'moving wall mask', metadata: { spacing_mm: [1, 1, 1], axes: ['x', 'y', 'z', 't'] } });

    // 1. decimate -> cut (capped) -> info, centerline, sequence
    const lm = add('IO/Load Tensor', { uuid: idMask }, [40, 40]);
    const mm = add('Geometry/Mesh From Mask', { smooth: 6 }, [330, 40]);
    const dec = add('Geometry/Mesh Decimate', { ratio: 0.3 }, [620, 40]);
    const cut = add('Geometry/Mesh Cut', { normal: '0,0,1', keep: 'positive', cap: true }, [900, 40]);
    const info = add('Geometry/Mesh Info', {}, [1180, 40]);
    const cl = add('Geometry/Centerline', { min_length: 4 }, [330, 330]);
    const ls = add('IO/Load Tensor', { uuid: idSeq }, [40, 520]);
    const sq = add('Geometry/Mesh Sequence', { dt: 0.05 }, [330, 520]);
    const view = add('Geometry/Mesh Viewer', {}, [900, 330]);
    lm.connect(0, mm, 0); mm.connect(0, dec, 0); dec.connect(0, cut, 0); cut.connect(0, info, 0); lm.connect(0, cl, 0);
    ls.connect(0, sq, 0);
    out.connect_lines_to_viewer = cl.connect(0, view, 0) != null;
    out.connect_seq_to_viewer = sq.connect(0, view, 1) != null;
    out.connect_mesh_to_viewer = info.connect(0, view, 2) != null;
    const ctl = await runAll('geometry');
    out.decimate = dec._info; out.cut_info = info._info; out.centerline = cl._info; out.sequence = sq._info;
    out.viewer_ids = view._ids;
    if (ctl) {
        const r = ctl.last.nodes;
        const lines = await client.get(cl._execResult[0]);
        out.centerline_value = { kind: lines.get('kind'), points: lines.get('points').shape, radius: lines.get('radius').shape, space: lines.get('space') };
    }

    // 2. Mesh Load (browser step, file in cm) -> Mesh Info; Mesh Save formats
    graph.clear();
    const obj = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n';
    const load = add('Geometry/Mesh Load', { units: 'cm' }, [40, 40]);
    load._pendingFile = new File([obj], 'tetra.obj');
    const info2 = add('Geometry/Mesh Info', {}, [330, 40]);
    const save = add('Geometry/Mesh Save', { format: 'obj', filename: 'tetra' }, [640, 40]);
    load.connect(0, info2, 0); info2.connect(0, save, 0);
    await runAll('mesh load');
    const loaded = load._uuid ? await client.get(load._uuid) : null;
    out.load = { info: info2._info, units_cm_to_mm: loaded ? Array.from(loaded.get('vertices').data.slice(3, 6)) : null };
    const blobs = [], origCreate = URL.createObjectURL, origClick = HTMLAnchorElement.prototype.click, names = [];
    URL.createObjectURL = (b) => { blobs.push(b); return 'blob:test'; };
    HTMLAnchorElement.prototype.click = function () { names.push(this.download); };
    out.save = {};
    try {
        for (const fmt of ['obj', 'stl', 'ply']) {
            save.properties.format = fmt;
            blobs.length = 0; names.length = 0;
            await save._save();
            const m = parseMesh(await blobs[0].arrayBuffer(), names[0]);
            out.save[fmt] = { file: names[0], vertices: m.vertices.length / 3, faces: m.faces.length / 3 };
        }
    } finally { URL.createObjectURL = origCreate; HTMLAnchorElement.prototype.click = origClick; }

    // 3. a flow value carrying centerlines, for the viewer
    const pts = new Float32Array([5, 5, 5, 10, 10, 10, 15, 12, 20, 20, 12, 30, 15, 12, 20, 12, 25, 25]);
    const flow = { kind: 'flow4d', vel: new HTensor('float32', [1, 3, 2, 2, 2], new Float32Array(24)), spacing: [0.002, 0.002, 0.002], dt: 0.04,
        centerlines: { points: new HTensor('float32', [6, 3], pts), offsets: new HTensor('uint32', [3], new Uint32Array([0, 4, 6])), radius: new HTensor('float32', [6], new Float32Array([3, 3, 2.5, 2, 2, 1.5])) } };
    out.flow_id = await client.put(flow, { name: 'flow with centerlines' });
    out.seq_id = sq._execResult?.[0] ?? null;
    out.ok = out.errors.length === 0;
    return out;
}

/** ?geomtest=2: fill holes, boolean, voxelize, coherent sequence, centerline table, run on the server */
export async function geomtest2(graph, canvas, h, client, HTensor) {
    const LG = window.LiteGraph;
    const out = { errors: [] };
    const add = (type, props = {}, pos = [0, 0]) => { const n = LG.createNode(type); Object.assign(n.properties, props); n.pos = pos; graph.add(n); return n; };
    const [nx, ny, nz] = [64, 64, 64];
    const tree = new Uint8Array(nx * ny * nz), blob = new Uint8Array(nx * ny * nz);
    const seg = (m, a, b, r) => {
        const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], L2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2 || 1;
        for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
            const t = Math.min(1, Math.max(0, ((i - a[0]) * d[0] + (j - a[1]) * d[1] + (k - a[2]) * d[2]) / L2));
            const q = [a[0] + t * d[0] - i, a[1] + t * d[1] - j, a[2] + t * d[2] - k];
            if (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 <= r * r) m[(k * ny + j) * nx + i] = 1;
        }
    };
    seg(tree, [32, 32, 4], [32, 32, 32], 5); seg(tree, [32, 32, 32], [14, 20, 58], 3.5); seg(tree, [32, 32, 32], [50, 44, 58], 3.5);
    seg(blob, [32, 32, 34], [32, 32, 34], 9);                       // ball around the junction
    const meta = { spacing_mm: [0.8, 0.8, 1.0], affine: [[0.8, 0, 0, -20], [0, 0.8, 0, -20], [0, 0, 1.0, -30], [0, 0, 0, 1]], axes: ['x', 'y', 'z'] };
    const idTree = await client.put(new HTensor('uint8', [nz, ny, nx], tree), { name: 'tree', metadata: meta });
    const idBlob = await client.put(new HTensor('uint8', [nz, ny, nx], blob), { name: 'ball', metadata: meta });
    const n4 = 32, T = 4, seq = new Uint8Array(T * n4 ** 3);
    for (let f = 0; f < T; f++) for (let q = 0; q < n4 ** 3; q++) {
        const i = q % n4, j = Math.floor(q / n4) % n4, k = Math.floor(q / n4 ** 2), c = 15.5, r = 6 + 2 * f;
        seq[f * n4 ** 3 + q] = (i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2 <= r * r ? 1 : 0;
    }
    const idSeq = await client.put(new HTensor('uint8', [T, n4, n4, n4], seq), { name: 'moving wall', metadata: { spacing_mm: [1, 1, 1] } });

    const lt = add('IO/Load Tensor', { uuid: idTree }, [40, 40]), lb = add('IO/Load Tensor', { uuid: idBlob }, [40, 200]), ls = add('IO/Load Tensor', { uuid: idSeq }, [40, 520]);
    const mt = add('Geometry/Mesh From Mask', {}, [330, 40]), mb = add('Geometry/Mesh From Mask', {}, [330, 200]);
    const diff = add('Geometry/Mesh Boolean', { op: 'difference' }, [620, 40]);
    const exact = add('Geometry/Mesh Boolean', { op: 'difference', method: 'exact' }, [620, 600]);
    const exinfo = add('Geometry/Mesh Info', {}, [900, 600]);
    const info = add('Geometry/Mesh Info', {}, [900, 40]);
    const cut = add('Geometry/Mesh Cut', { normal: '1,0,0', keep: 'positive', cap: false }, [620, 200]);   // opens the surface
    const fill = add('Geometry/Mesh Fill Holes', { max_edges: 2000 }, [900, 200]);
    const info2 = add('Geometry/Mesh Info', {}, [1180, 200]);
    const vox = add('Geometry/Mesh Voxelize', { voxel_mm: 0.8 }, [620, 360]);
    const back = add('Geometry/Mesh From Mask', { keep: 'all' }, [900, 360]);
    const cl = add('Geometry/Centerline', {}, [330, 360]);
    const cm = add('Geometry/Centerline Measure', {}, [620, 480]);
    const sq = add('Geometry/Mesh Sequence', { coherent: true, dt: 0.05 }, [330, 520]);
    const view = add('Geometry/Mesh Viewer', {}, [900, 520]);
    lt.connect(0, mt, 0); lb.connect(0, mb, 0); mt.connect(0, diff, 0); mb.connect(0, diff, 1); diff.connect(0, info, 0);
    mt.connect(0, exact, 0); mb.connect(0, exact, 1); exact.connect(0, exinfo, 0);
    mt.connect(0, cut, 0); cut.connect(0, fill, 0); fill.connect(0, info2, 0);
    mt.connect(0, vox, 0); vox.connect(0, back, 0); lt.connect(0, cl, 0); ls.connect(0, sq, 0);
    cl.connect(0, cm, 0); lt.connect(0, cm, 1);
    cm.connect(0, view, 0); sq.connect(0, view, 1); diff.connect(0, view, 2);
    const { plan, errors } = await h.compile(graph);
    if (errors.length) return { ok: false, errors: errors.map((e) => `${e.node.title}: ${e.error}`) };
    const { run } = await h.call('run_submit', { plan });
    const ctl = new h.RunController(graph, canvas, { banner() {}, onStatus() {} });
    await ctl.attach(run, { auto: true });
    for (let i = 0; i < 400 && ['running', 'waiting_client'].includes(ctl.last?.status); i++) await new Promise((r) => setTimeout(r, 200));
    await new Promise((r) => setTimeout(r, 500));
    if (ctl.last?.status !== 'done') out.errors.push(`run ${ctl.last?.status}: ` + Object.values(ctl.last?.nodes ?? {}).filter((n) => n.error).map((n) => `${n.title}: ${n.error}`).join('; '));
    out.tree = mt._info; out.difference = info._info; out.opened_then_filled = info2._info; out.fill = fill._info; out.voxelize = vox._info; out.remeshed = back._info;
    out.exact_boolean = exact._info; out.exact_info = exinfo._info; out.voxel_boolean = diff._info;
    out.centerline = cl._info; out.measure = cm._info; out.sequence = sq._info; out.boolean = diff._info;
    out.branch_table = ctl.last?.nodes?.[cl.id]?.message?.branch_info?.map((b) => ({ id: b.id, parent: b.parent, length: +b.length_mm.toFixed(1), tip_radius: +b.tip_radius_mm.toFixed(1) }));
    out.junctions = ctl.last?.nodes?.[cm.id]?.message?.junctions;
    out.ids = { lines: cm._execResult?.[0], raw_lines: cl._execResult?.[0], tree_mesh: mt._execResult?.[0], seq: sq._execResult?.[0], mesh: diff._execResult?.[0], filled: fill._execResult?.[0] };
    out.ok = out.errors.length === 0;
    return out;
}

/** ?geomtest=3: wall shear stress and pulse wave velocity of a tube carrying a travelling wave, on the server */
export async function geomtest3(graph, canvas, h, client, HTensor) {
    const LG = window.LiteGraph;
    const out = { errors: [] };
    const add = (type, props = {}, pos = [0, 0]) => { const n = LG.createNode(type); Object.assign(n.properties, props); n.pos = pos; graph.add(n); return n; };
    const [nx, ny, nz, nf] = [24, 24, 100, 50], R = 6, c = 4.0, period = 1.0, dt = period / nf, nvx = nx * ny * nz;
    const g = (t, t0, w) => Math.exp(-Math.pow((((((t - t0 + 1.5) % 1) + 1) % 1) - 0.5) / w, 2));
    const pulse = (t) => 0.3 + 1.2 * g(t, 0.25, 0.06) + 0.35 * g(t, 0.55, 0.05);       // systolic pulse with a dicrotic bump
    const vel = new Float32Array(nf * 3 * nvx), mask = new Uint8Array(nvx);
    for (let q = 0; q < nvx; q++) { const i = (q % nx) - 11.5, j = (Math.floor(q / nx) % ny) - 11.5; mask[q] = i * i + j * j <= R * R ? 1 : 0; }
    for (let f = 0; f < nf; f++) for (let q = 0; q < nvx; q++) {
        const i = (q % nx) - 11.5, j = (Math.floor(q / nx) % ny) - 11.5, k = Math.floor(q / (nx * ny)), r2 = (i * i + j * j) / (R * R);
        if (r2 < 1) vel[(f * 3 + 2) * nvx + q] = (1 - r2) * pulse(f * dt - k * 1e-3 / c) + 0.03 * (Math.random() - 0.5);
    }
    const meta = { spacing_mm: [1, 1, 1], axes: ['x', 'y', 'z'] };
    const idVel = await client.put(new HTensor('float32', [nf, 3, nz, ny, nx], vel), { name: 'travelling wave velocity', metadata: meta });
    const idMask = await client.put(new HTensor('uint8', [nz, ny, nx], mask), { name: 'tube mask', metadata: meta });
    const lv = add('IO/Load Tensor', { uuid: idVel }, [40, 40]), lm = add('IO/Load Tensor', { uuid: idMask }, [40, 200]);
    const mk = add('Flow/Flow Make', { spacing: '1,1,1', dt }, [330, 40]);
    const mm = add('Geometry/Mesh From Mask', { keep: 'all' }, [330, 200]);
    const wss = add('Geometry/Mesh Wall Shear', {}, [640, 120]);
    const cl = add('Geometry/Centerline', { coords: 'mm' }, [330, 360]);
    const pwv = add('Geometry/Pulse Wave Velocity', { points: 8 }, [640, 360]);
    const view = add('Geometry/Mesh Viewer', { scalar: 'tawss' }, [940, 120]);
    lv.connect(0, mk, 0); lm.connect(0, mm, 0); mm.connect(0, wss, 0); mk.connect(0, wss, 1); lm.connect(0, cl, 0); cl.connect(0, pwv, 0); mk.connect(0, pwv, 1); wss.connect(0, view, 0);
    const { plan, errors } = await h.compile(graph);
    if (errors.length) return { ok: false, errors: errors.map((e) => `${e.node.title}: ${e.error}`) };
    const { run } = await h.call('run_submit', { plan });
    const ctl = new h.RunController(graph, canvas, { banner() {}, onStatus() {} });
    await ctl.attach(run, { auto: true });
    for (let i = 0; i < 400 && ['running', 'waiting_client'].includes(ctl.last?.status); i++) await new Promise((r) => setTimeout(r, 200));
    await new Promise((r) => setTimeout(r, 500));
    if (ctl.last?.status !== 'done') out.errors.push(`run ${ctl.last?.status}: ` + Object.values(ctl.last?.nodes ?? {}).filter((n) => n.error).map((n) => `${n.title}: ${n.error}`).join('; '));
    out.wss = wss._info; out.pwv = pwv._info; out.pwv_lines = pwv._execResult?.[0]; out.pwv_warnings = ctl.last?.nodes?.[pwv.id]?.message?.warnings; out.pwv_methods = Object.fromEntries(Object.entries(ctl.last?.nodes?.[pwv.id]?.message?.methods ?? {}).map(([k, m]) => [k, +m.pwv_m_s.toFixed(2)])); out.expected = { tawss_pa: 3.5e-3 * 2 / (R * 1e-3), pwv_m_s: c };
    out.mesh_id = wss._execResult?.[0];
    out.ok = out.errors.length === 0;
    return out;
}
