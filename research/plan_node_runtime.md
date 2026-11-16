# Node editor runtime: client/server placement and persistent graphs (design, 2026-11-14)

Status: design only, nothing built. Based on a read of `plotting_website/node_editor/src/{main,node_runner,hasty}.js` and
`rustlib/crates/hasty-server/src/{service,bank,commands}.rs`.

## What exists today (the constraints)

- The **browser is the executor**. `runGraph` (node_runner.js) walks a topological order and awaits one `execute` RPC per node,
  strictly one at a time (independent branches do not overlap). Close the tab and the run stops being driven.
- `Execute` is a **unary, blocking RPC**: the call stays open for the whole command. Outputs are put in the bank only when the
  command returns, so a client that left never learns the output ids (they sit in the bank under the command name).
  Commands get `Ctx { bank }` only: no progress callback, no cancel flag.
- The **bank is in memory** (`RwLock<HashMap>`); a server restart loses every value. "Clear Bank" deletes everything,
  including values a running graph still needs.
- `Watch` is a broadcast stream of `put/delete/show` events: a ready-made channel for run events.
- Graph documents exist only as the Save/Load JSON download. No graph id, no server copy.
- Client-side compute already exists elsewhere (flow viewer: WebGPU tracing, worker pool in `common/pool.js`; orthoslicer pulls
  volumes client-side), so "run in the browser" is not new, only its integration in the node graph is.

## 1. Placement: where a node runs

Each node class declares `static placement = 'server' | 'client' | 'either'`; each instance gets a combo widget
`run on: auto | client | server` (stored in `properties.placement`, saved with the graph). `auto` resolves per class default and
input size.

Values get a location: `{ loc: 'server', id }` (what flows today) or `{ loc: 'client', handle }` (an in-memory typed array in the
tab). The runner inserts a **transfer edge** wherever producer and consumer differ: upload = `client.put`, download =
`client.get(codec lz4)`. Transfer size is known from `info` (shape × dtype) before running, so the editor can show the bytes
on the edge and refuse silent multi-GB moves (warn above a threshold, default 256 MB).

Suggested defaults:

| Node kind | Placement | Why |
|---|---|---|
| model_run, NUFFT, recon, FFT on volumes, big arithmetic | server | data and GPU live there |
| NIfTI parse / save, Scalar, Compress preview, Ortho Slicer, flow tracing | client | data originates or ends in the tab; WebGPU already there |
| small arithmetic, reductions, stats | either (auto: server if any input is server-resident and > 16 MB, else client) | avoid transfers |

"either" nodes need one op definition with two backends (server command, client JS/WebGPU). Keep the set tiny (add/sub/mul/div,
neg/abs, scale, slicing); every pair needs a numeric parity test (same approach as the ONNX parity tests).

Rules that keep this tractable: a client-placed node never feeds a persistent server run mid-graph (section 2); the **graph
is partitioned into maximal same-placement segments** and only segment boundaries move data.

## 2. Persistent graphs and runs

Move the **driver** to the server; the browser becomes a viewer/editor of server state.

Server objects (stored under `data/hasty/graphs/<graph_id>/`, plain JSON, small):
- `graph.json`: the LiteGraph document plus a `version` (content hash).
- `runs/<run_id>.json`: pinned graph version, per-node state `queued | running | done | error | cancelled | blocked_on_client`,
  progress 0..1 and a message, start/end times, output ids, error text. Written on every state change (atomic rename).

New RPCs (hasty.v2 additions): `SaveGraph`, `ListGraphs`, `GetGraph`, `DeleteGraph`; `Submit(graph_id, version)` → `run_id`;
`GetRun(run_id)` (snapshot); `WatchRun(run_id)` (stream: snapshot first, then deltas; this is what a returning tab attaches to);
`Cancel(run_id)`; `ListRuns(graph_id)`.

Scheduler (server): topological order over the **server segment**, independent branches run concurrently up to a resource
budget (GPU slots, host RAM; matches the "no swap / cap heavy jobs" rule), one run queue per GPU. A node that needs a
client-placed input or output pauses as `blocked_on_client`; it resumes when a client attaches and supplies/receives it.
Everything between such boundaries runs unattended.

Commands that run long take a `Ctx.progress(frac, msg)` callback and a `Ctx.cancelled()` check, and long ones write their own
checkpoint (iteration) so a restart can resume at node granularity (same STOP/resume convention as the recon drivers).

The client flow: editor opens `?graph=<id>`; Run = `Submit`; the tab subscribes `WatchRun`; closing the tab changes nothing;
opening it later attaches to the same run and repaints node status bars from the snapshot. Edits during a run apply to the
graph, not to the pinned run version; changed nodes show "stale".

Bank lifetime: values referenced by a run are **pinned** (excluded from Clear Bank and any future GC) while the run exists.
The bank being memory-only is the real limit: a server restart loses pinned values. Options: spill pinned values to
`data/hasty/bank/` (hasty-value already has a file codec) and reload on start. Without that, "persistent" means "survives the
browser", not "survives a server restart". Decide how far to go.

Multi-user, auth and per-user graph ownership: out of scope until the server is shared.

## 3. Suggested order (each step is useful alone)

1. `Execute` fix: stop being unary-blocking for long commands. Smallest step: `ExecuteAsync` returns a job id at once,
   `WatchJob` streams progress and the final output ids (also fixes orphaned results). Add `Ctx.progress/cancelled`.
2. Graph store + `Submit/GetRun/WatchRun` with the scheduler for all-server graphs. The editor Run button becomes Submit.
3. Placement widget + transfer edges + segment partition; `blocked_on_client`.
4. Bank spill for pinned values (restart survival).
5. Parallel branches and per-GPU queues.

## 4. Decisions (user, 2026-11-14)

- **No users** for now; one shared namespace.
- **Graph list on the base website** (plotting_website/index.html): every saved graph with a status (idle / running / blocked on
  client / done / error, last run time, node counts).
- **Client-placed node in a graph run: the run waits** for a client to connect and press Continue. No forbidding.
- **A run need not survive a server restart; the graph must.** Persisted: nodes, placement, widget settings, links, positions.
  Not persisted: values and evaluations. So step 4 (bank spill) is dropped, and run state is held in memory (finished run
  summaries may still be written, but are not required). After a restart a graph comes back "idle" with all nodes stale.

Consequences: `graphs/<id>/graph.json` is the only on-disk object; `GetRun`/`WatchRun` read in-memory state; pinned values
only need to survive client disconnects, not restarts.

## 5. Flow nodes in the editor (built 2026-11-14, first version)

Category `Flow`, connection type `flow_uuid` (a flow value: dict kind flow4d, see hasty-server/src/flow.rs):

| Node | Server command | Notes |
|---|---|---|
| Flow Load | (bank list) | picks an existing flow value (recon output, show_flow); label re-matched by name after a restart |
| Flow Make | flow_make | vel [F,3,Z,Y,X] + optional mag, mask; spacing from vel metadata or widget (mm), dt, venc |
| Flow Info | flow_info | passes the flow through; `info` output is the summary object |
| PC-MRA | flow_pcmra | tensor [Z,Y,X] with spacing_mm: feeds Ortho Slicer / NIfTI Save |
| Flow Plane | flow_plane | disc at a voxel centre + normal: through-plane velocity [F,n,n], lumen mask, summary (net / forward / backward ml, flow_ml_s per frame) |
| Flow Viewer | (opens flow/index.html) | start frame; particles on server (default) or in the browser (loads the whole volume into the tab) |

All server commands are CPU. Tested end to end headless on CPU (`?flowtest=1`).

Not built, needs viewer changes: the **back-channel** from the viewer into the graph. Proposed: the viewer gets a "send to editor"
button that writes a small `flow_selection` value (picked plane: center, normal, radius, label; vessel tool result: plane stack
and flow curves) to the bank with metadata `{flow: <id>}`; a `Flow Selection` node lists the selections of the connected flow
and outputs the plane parameters (feeding Flow Plane) or the curves. The other direction is simple: pass plane parameters from
the graph to the viewer in the URL. Streamlines/pathlines as graph values (Flow Lines node) have no consumer yet except the
viewer, which computes its own; skipped.

## 6. Open questions

- Graph list: the status "running" needs the server-side runs from step 2, so the list can only show saved graphs until then.
- Flow viewer back-channel as proposed above, or something else (e.g. embed the viewer inside the editor as a node panel)?

## 7. Built 2026-11-14 (steps 1-3 in one: server-driven runs, graph store, placement)

- Server `hasty-server/src/jobs.rs` (commands via Execute, no proto change): `graph_save/list/get/delete` (one JSON file per graph in
  `data/hasty/graphs`, env HASTY_GRAPHS), `run_submit/get/list/cancel`, `run_client_claim/done`. Plan = DAG of command steps compiled
  by the editor (`node.plan()`), run by the server: independent branches in parallel (HASTY_RUN_PARALLEL, default 2 heavy steps at a
  time), per-node status and progress (`Ctx.progress`, `Ctx.cancelled`), unchanged steps reused across runs of the same graph, steps
  placed on the client wait (`waiting_client`) until a browser claims them. `fft/ifft` got `norm`; `flow_make` takes `spacing_mm`;
  `flow_plane` defaults to the volume centre; `model_run` reports progress and honours cancel.
- Editor: Run = compile + submit + follow (`server_run.js`); the submitting tab serves its own browser steps, a re-opened tab shows
  Continue. Graph name + server Save + autosave (3 s) + `?graph=<id>` open + re-attach to the latest run. Export/Import keep the file
  route. Right-click "Run on: auto / server / client" for nodes that can do both; title tint shows an explicit placement.
- Home page: Graphs table with status (idle / running / waiting for browser / done / error), polled every 3 s.
- Not done: bank values are not pinned (Clear Bank during a run breaks it), no `Execute` streaming for non-run callers, no per-GPU queues,
  runs are in memory only (as decided), model_run / NIfTI-upload client steps / flow viewer not exercised through a server run.
- Checks: `?runtest=1|2|3` (src/devtools.js) against a local `hasty-serve` (HASTY_TLS=0): plan runs, stops at a browser step, a fresh
  page re-attaches and continues, cancel, cache hit on re-run; Rust unit tests for plan validation and fft norm conventions.
