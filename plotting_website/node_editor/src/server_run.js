/**
 * server_run.js — server-driven runs for the node editor.
 *
 * The editor compiles the graph into a plan (a DAG of command steps, see rustlib/crates/hasty-server/src/jobs.rs) and submits
 * it; the server drives the run. A node runs on the server when it has a `plan()` method and its placement allows it,
 * otherwise in the browser: the run then waits at that node until a browser claims it (run_client_claim), executes
 * `executeAsync` and reports the outputs (run_client_done). The tab that submitted a run does this automatically; a tab that
 * attaches to a run later (graph reopened) shows a Continue button.
 *
 * Node contract:
 *   async plan() -> { cmd, options?, inputs?: [input slot indices], name?, meta_in?, meta_out?, outs? }   (server steps)
 *   async executeAsync(inputs) -> outputs[]                                                              (browser steps)
 *   onRunResult(outs, message)       called whenever a finished node's result is (re)painted
 * node.properties.placement: 'auto' (server when it has plan()) | 'server' | 'client'
 */
import { fetchMeta, isId, run as exec } from './hasty.js';

export async function call(cmd, options = {}) {
    const { message } = await exec(cmd, [], options);
    return message ? JSON.parse(message) : null;
}

export const placementOf = (node) => node.properties?.placement ?? 'auto';

/** { plan, errors: [{ node, error }] } */
export async function compilePlan(graph) {
    const planned = new Map();
    const errors = [];
    for (const node of graph._nodes) {
        const hasPlan = typeof node.plan === 'function', hasExec = typeof node.executeAsync === 'function';
        if (!hasPlan && !hasExec) continue;
        const p = placementOf(node);
        const client = p === 'client' || (p === 'auto' && !hasPlan);
        if (client && !hasExec) { errors.push({ node, error: 'this node cannot run in the browser' }); continue; }
        if (!client && !hasPlan) { errors.push({ node, error: 'this node cannot run on the server' }); continue; }
        planned.set(node.id, { node, client });
    }
    const plan = [];
    for (const [id, { node, client }] of planned) {
        const refs = (node.inputs ?? []).map((inp) => {
            if (inp.link == null) return null;
            const l = graph.links[inp.link];
            return l && planned.has(l.origin_id) ? { node: l.origin_id, slot: l.origin_slot } : null;
        });
        try {
            if (client) { plan.push({ id, title: node.title, client: true, inputs: refs }); continue; }
            const sp = await node.plan();
            if (sp.client) { plan.push({ id, title: node.title, client: true, inputs: refs }); continue; }
            const idx = sp.inputs ?? refs.map((_, i) => i);
            const inputs = sp.cmd.startsWith('_') ? idx.map((i) => refs[i]) : idx.filter((i) => refs[i]).map((i) => refs[i]);
            plan.push({ id, title: node.title, cmd: sp.cmd, options: sp.options ?? {}, inputs, name: sp.name ?? '', meta_in: sp.meta_in ?? [], meta_out: sp.meta_out ?? [], ...(sp.outs ? { outs: sp.outs } : {}) });
        } catch (e) {
            errors.push({ node, error: String(e.message ?? e) });
        }
    }
    return { plan, errors };
}

const UI_STATUS = { pending: 'queued', queued: 'queued', running: 'running', waiting_client: 'waiting', done: 'done', error: 'error', skipped: 'skipped', cancelled: 'skipped' };
const BOX = { queued: '#667788', running: '#cc7700', waiting: '#cc9900', done: '#00aa44', error: '#cc2200', skipped: '#555555' };
const ACTIVE = new Set(['running', 'waiting_client']);

/** follows one run, paints its node states on the canvas and serves the steps that wait for this browser */
export class RunController {
    /** ui: { banner(text, {continue: bool}|null), onStatus(status) } */
    constructor(graph, canvas, ui) {
        this.graph = graph; this.canvas = canvas; this.ui = ui;
        this.run = null; this.timer = null; this.auto = false; this.go = false; this.busy = new Set(); this.last = null; this.seen = new Map();
    }

    async attach(runId, { auto = false } = {}) {
        this.run = runId; this.auto = auto; this.go = auto; this.seen.clear();
        clearTimeout(this.timer);
        await this.poll();
    }

    async latestFor(graphId) {
        const { runs } = await call('run_list', { graph: graphId });
        return runs[0] ?? null;
    }

    async poll() {
        if (!this.run) return;
        let r;
        try { r = await call('run_get', { run: this.run }); } catch (e) { this.ui.banner(`run lost: ${String(e.message ?? e).slice(0, 80)}`, null); this.run = null; return; }
        this.last = r;
        this.paint(r);
        const waiting = Object.entries(r.nodes).filter(([, n]) => n.status === 'waiting_client');
        if (waiting.length && this.go) for (const [id] of waiting) this.clientStep(Number(id));
        const n = Object.keys(r.nodes).length, d = Object.values(r.nodes).filter((x) => x.status === 'done').length;
        const text = { running: `running ${d}/${n}`, waiting_client: `waiting for this browser: ${waiting.length} step(s)`, done: `done ${d}/${n}`, error: 'finished with errors', cancelled: 'cancelled' }[r.status] ?? r.status;
        this.ui.banner(text, { continue: waiting.length > 0 && !this.go, cancel: ACTIVE.has(r.status) });
        this.ui.onStatus(r.status);
        if (ACTIVE.has(r.status)) this.timer = setTimeout(() => this.poll(), r.status === 'running' ? 600 : 1200);
    }

    paint(r) {
        for (const [sid, st] of Object.entries(r.nodes)) {
            const node = this.graph.getNodeById(Number(sid));
            if (!node) continue;
            const status = UI_STATUS[st.status] ?? st.status;
            node._execStatus = status; node._execError = st.error; node._execProgress = st.progress; node._execNote = st.note; node._execResult = st.outs;
            node.boxcolor = BOX[status] ?? node.boxcolor;
            const key = `${st.status}|${JSON.stringify(st.outs)}|${st.message ? JSON.stringify(st.message).length : 0}`;
            if (st.status === 'done' && this.seen.get(sid) !== key) {
                this.seen.set(sid, key);
                node.onRunResult?.(st.outs, st.message);
                const first = st.outs?.[0];
                if (isId(first)) fetchMeta(first).then((m) => { node._execMeta = m; this.canvas.setDirty(true, true); }).catch(() => {});
            }
        }
        this.canvas.setDirty(true, true);
    }

    async clientStep(id) {
        if (this.busy.has(id)) return;
        this.busy.add(id);
        const node = this.graph.getNodeById(id), run = this.run;
        try {
            let inputs;
            try { ({ inputs } = await call('run_client_claim', { run, node: id })); } catch { return; }   // another browser took it
            let outs, error;
            try { outs = await node.executeAsync(inputs); } catch (e) { error = String(e.message ?? e); }
            await call('run_client_done', error ? { run, node: id, error } : { run, node: id, outs: outs ?? [] });
        } finally { this.busy.delete(id); }
    }

    /** Continue button: this browser may run the waiting steps of this run */
    proceed() { this.go = true; this.poll(); }

    async cancel() { if (this.run) { await call('run_cancel', { run: this.run }); this.poll(); } }
}
