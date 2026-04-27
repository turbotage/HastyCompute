/**
 * node_runner.js — async graph execution engine for the HastyEdit node editor.
 *
 * Runs all nodes that implement `executeAsync(inputUuids)` in topological
 * (dependency) order, passing Uint8Array UUIDs between connected outputs and
 * inputs.  Provides live visual feedback by setting node._execStatus and
 * node.boxcolor while running.
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

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Execute all nodes in the graph that implement `executeAsync`.
 *
 * For each such node the runner:
 *  1. Collects output UUIDs from upstream connected nodes.
 *  2. Calls `await node.executeAsync(inputUuids)`.
 *  3. Stores the result UUIDs for downstream nodes.
 *  4. Fetches tensor metadata asynchronously and triggers a canvas redraw.
 *
 * Visual state is communicated via:
 *   node._execStatus  'running' | 'done' | 'error'
 *   node._execResult  Uint8Array[] | null
 *   node._execError   string | null
 *   node._execMeta    { device, dtype, shape } | null
 *   node.boxcolor     colour string (orange → green / red)
 *
 * @param {LGraph}       graph
 * @param {LGraphCanvas} canvas   — used to trigger redraws during execution
 */
export async function runGraph(graph, canvas) {
    // nodeId → { [slotIndex]: Uint8Array } — output UUIDs available downstream
    const results = {};

    for (const node of topoSort(graph)) {
        if (typeof node.executeAsync !== 'function') continue;

        // ── Collect input UUIDs ──────────────────────────────────────────────
        const inputUuids = [];
        if (node.inputs) {
            for (const inp of node.inputs) {
                if (inp.link == null) { inputUuids.push(null); continue; }
                const link = graph.links[inp.link];
                if (!link)            { inputUuids.push(null); continue; }
                inputUuids.push(results[link.origin_id]?.[link.origin_slot] ?? null);
            }
        }

        // ── Mark running ────────────────────────────────────────────────────
        node._execStatus = 'running';
        node._execError  = null;
        node._execResult = null;
        node._execMeta   = null;
        node.boxcolor    = '#cc7700';
        canvas.setDirty(true, true);

        // ── Execute ─────────────────────────────────────────────────────────
        try {
            const uuids = await node.executeAsync(inputUuids);

            node._execResult = uuids ?? [];
            node._execStatus = 'done';
            node.boxcolor    = '#00aa44';

            // Store slot-keyed UUIDs for downstream nodes
            results[node.id] = {};
            if (uuids) uuids.forEach((u, i) => { results[node.id][i] = u; });

            // Fetch metadata in the background — triggers redraw when ready
            if (uuids?.length && uuids[0]) {
                fetchMeta(uuids[0]).then(meta => {
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
