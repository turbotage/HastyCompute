/**
 * node_runner.js — async graph execution engine for the HastyEdit node editor.
 *
 * Runs all nodes that implement `executeAsync(inputValues)` in topological
 * (dependency) order, passing values (UUIDs, config objects, etc.) between
 * connected outputs and inputs.  Provides live visual feedback by setting
 * node._execStatus and node.boxcolor while running.
 *
 * Fingerprint caching: nodes that have already executed successfully and
 * whose inputs + widget settings haven't changed are skipped.  Opt out by
 * setting `static _skipCache = true` on the node class.
 */

import { fetchMeta } from './grpc_client.js';

// ── Topological sort ──────────────────────────────────────────────────────────

/**
 * Kahn's algorithm on the litegraph node set.
 * Uses graph.links (object keyed by link_id) to build the DAG.
 *
 * @param {LGraph} graph
 * @returns {LGraphNode[]} nodes in dependency order (sources first)
 */
function topoSort(graph) {
    const nodeById = {};
    const inDeg    = {};
    const succs    = {};   // nodeId → [targetId, ...]

    for (const n of graph._nodes) {
        nodeById[n.id] = n;
        inDeg[n.id]    = 0;
        succs[n.id]    = [];
    }

    for (const link of Object.values(graph.links)) {
        inDeg[link.target_id] = (inDeg[link.target_id] || 0) + 1;
        (succs[link.origin_id] ??= []).push(link.target_id);
    }

    const queue = graph._nodes
        .filter(n => inDeg[n.id] === 0)
        .map(n => n.id);
    const order = [];

    while (queue.length) {
        const id = queue.shift();
        order.push(nodeById[id]);
        for (const sid of (succs[id] ?? [])) {
            if (--inDeg[sid] === 0) queue.push(sid);
        }
    }

    return order;
}

// ── Fingerprint helpers ───────────────────────────────────────────────────────

/**
 * Stable string representation of a single input/output value.
 *   - null/undefined → ''
 *   - Uint8Array (UUID) → hex string
 *   - object → JSON with sorted keys (stable across V8 property insertion order)
 *   - anything else → JSON.stringify
 */
function _fingerprintValue(v) {
    if (v == null) return '';
    if (v instanceof Uint8Array) {
        let s = '';
        for (let i = 0; i < v.length; i++) s += v[i].toString(16).padStart(2, '0');
        return s;
    }
    if (typeof v === 'object') {
        const keys = Object.keys(v).sort();
        return JSON.stringify(v, keys);
    }
    return JSON.stringify(v);
}

/**
 * Compute a fingerprint string for a node given its current input values and
 * widget settings.  Used to decide whether executeAsync can be skipped.
 *
 * @param {LGraphNode} node
 * @param {Array}      inputValues  — values passed to executeAsync
 * @returns {string}
 */
function _nodeFingerprint(node, inputValues) {
    const inp = inputValues.map(_fingerprintValue);
    // Exclude button widgets (type='button') — they carry no data
    const wid = (node.widgets ?? [])
        .filter(w => w.type !== 'button')
        .map(w => JSON.stringify(w.value));
    return inp.join('|') + '||' + wid.join('|');
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Execute all nodes in the graph that implement `executeAsync`.
 *
 * For each such node the runner:
 *  1. Collects output values from upstream connected nodes.
 *  2. Checks the fingerprint cache; skips the node if inputs/settings unchanged.
 *  3. Calls `await node.executeAsync(inputValues)`.
 *  4. Stores the result values for downstream nodes.
 *  5. Fetches tensor metadata asynchronously and triggers a canvas redraw.
 *
 * Visual state is communicated via:
 *   node._execStatus  'running' | 'done' | 'error'
 *   node._execResult  any[] | null
 *   node._execError   string | null
 *   node._execMeta    { device, dtype, shape } | null
 *   node.boxcolor     colour string (orange → green / red)
 *
 * Cache opt-out: set `static _skipCache = true` on the node class to always
 * re-execute regardless of fingerprint (useful for side-effecting nodes like
 * NiftiLoad where the node manages its own internal caching).
 *
 * @param {LGraph}       graph
 * @param {LGraphCanvas} canvas   — used to trigger redraws during execution
 */
export async function runGraph(graph, canvas) {
    // nodeId → { [slotIndex]: value } — output values available downstream
    const results = {};

    for (const node of topoSort(graph)) {
        if (typeof node.executeAsync !== 'function') continue;

        // ── Collect input values ─────────────────────────────────────────────
        const inputValues = [];
        if (node.inputs) {
            for (const inp of node.inputs) {
                if (inp.link == null) { inputValues.push(null); continue; }
                const link = graph.links[inp.link];
                if (!link)            { inputValues.push(null); continue; }
                inputValues.push(results[link.origin_id]?.[link.origin_slot] ?? null);
            }
        }

        // ── Fingerprint cache check ──────────────────────────────────────────
        if (!node.constructor._skipCache) {
            const fp = _nodeFingerprint(node, inputValues);
            if (fp === node._execFingerprint &&
                node._execStatus === 'done' &&
                node._execResult != null) {
                // Inputs and settings unchanged — restore results without re-running
                results[node.id] = {};
                node._execResult.forEach((v, i) => { results[node.id][i] = v; });
                canvas.setDirty(true, true);
                continue;
            }
        }

        // ── Mark running ─────────────────────────────────────────────────────
        node._execStatus = 'running';
        node._execError  = null;
        node._execResult = null;
        node._execMeta   = null;
        node.boxcolor    = '#cc7700';
        canvas.setDirty(true, true);

        // ── Execute ──────────────────────────────────────────────────────────
        try {
            const values = await node.executeAsync(inputValues);

            node._execResult = values ?? [];
            node._execStatus = 'done';
            node.boxcolor    = '#00aa44';

            // Store slot-keyed values for downstream nodes
            results[node.id] = {};
            if (values) values.forEach((v, i) => { results[node.id][i] = v; });

            // Save fingerprint so next run can skip if nothing changed
            if (!node.constructor._skipCache) {
                node._execFingerprint = _nodeFingerprint(node, inputValues);
            }

            // Fetch metadata in the background — triggers redraw when ready
            const firstVal = values?.[0];
            if (firstVal instanceof Uint8Array) {
                fetchMeta(firstVal).then(meta => {
                    node._execMeta = meta;
                    canvas.setDirty(true, true);
                });
            }
        } catch (e) {
            node._execStatus = 'error';
            node._execError  = String(e.message ?? e);
            node.boxcolor    = '#cc2200';
            results[node.id] = {};
            console.error(`[runGraph] "${node.title}" failed:`, e);
        }

        canvas.setDirty(true, true);
    }
}
