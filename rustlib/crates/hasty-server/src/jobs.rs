//! Persistent graphs and server-driven runs for the node editor.
//!
//! - **Graphs** (`graph_save | graph_list | graph_get | graph_delete`): the node editor's document (opaque JSON) as one file
//!   `<graphs dir>/<id>.json` (default `data/hasty/graphs`, env HASTY_GRAPHS). Nodes, links, widget settings and placement
//!   survive restarts; values and run state do not.
//! - **Runs** (`run_submit | run_get | run_list | run_cancel | run_client_claim | run_client_done`): the editor compiles the
//!   graph into a *plan* (a DAG of command steps) and submits it. The server drives it: independent branches run in
//!   parallel (at most `HASTY_RUN_PARALLEL`, default 2, steps at a time), progress is reported per node, and a step marked
//!   `client` waits until a browser claims it, runs it, and reports the outputs. A closed browser changes nothing; a
//!   returning one reads `run_get` and repaints. Runs live in memory only.
//!
//! Plan node (JSON): `{id, title, client?, cmd, options?, inputs: [ref|null], name?, meta_in?, meta_out?, outs?}`
//! - `ref`: `{"node": id, "slot": k}` (output slot k of another node) or `{"lit": json}`. Ids of bank values are 32-hex strings;
//!   any other JSON (numbers, config objects) is passed along as is.
//! - `cmd`: a registry command, or `_const` (outputs `options.value`) / `_pass` (outputs its inputs).
//! - `meta_in`: `[{input: k, keys: {option: metadata_key}}]`: fill `options[option]` from the metadata of input k when absent.
//! - `meta_out`: `[{output: k, from_input: {input: i, keys: [..]}, from_message: {dst: src}, set: {..}, set_input_id: {key: i}}]`:
//!   metadata written to output k (bank value) after the command.
//! - `outs`: node output slots, each `{"output": k}` | `{"input": k}` | `{"message": true}` | `{"message_key": "x"}`; default: the
//!   command outputs in order.
//! Unchanged steps (same command, options and inputs as the previous run of the same graph) are reused when their output
//! values are still in the bank.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use hasty_value::Value;
use serde_json::{json, Map, Value as Json};
use tokio::sync::{mpsc, oneshot, Notify, Semaphore};
use uuid::Uuid;

use crate::bank::Bank;
use crate::commands::{Ctx, Registry};
use crate::Error;

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

fn as_id(v: &Json) -> Option<Uuid> {
    let s = v.as_str()?;
    (s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit())).then(|| Uuid::parse_str(s).ok()).flatten()
}

const KEEP_RUNS: usize = 40;

pub const COMMANDS: &[(&str, &str)] = &[
    ("graph_save", "save a node editor graph: options {id? (new when absent), name, graph: JSON}; message {id, name}"),
    ("graph_list", "saved graphs: message {graphs: [{id, name, updated_ms, nodes, status (idle | running | waiting_client | done | error | cancelled), run, run_started_ms}]}"),
    ("graph_get", "options {id}; message {id, name, updated_ms, graph}"),
    ("graph_delete", "options {id}"),
    ("run_submit", "start a run: options {graph? (graph id, for the list), name?, plan: [plan nodes, see hasty_server::jobs]}; message {run}"),
    ("run_get", "options {run}; message: the run with per-node status, progress, outputs"),
    ("run_list", "options {graph?}; message {runs: [{run, graph, name, status, started_ms, ended_ms}]}, newest first"),
    ("run_cancel", "options {run}: stop launching steps, wake client waits; running commands that check for cancellation stop"),
    ("run_client_claim", "options {run, node}: take a waiting client step; message {inputs: [..]} (fails when someone else has it)"),
    ("run_client_done", "options {run, node, outs: [..], error?, message?}: report a client step"),
];

pub struct Jobs {
    dir: PathBuf,
    bank: Arc<Bank>,
    registry: Arc<Registry>,
    runs: Mutex<Vec<Arc<Run>>>,
    cache: Mutex<HashMap<String, (String, Vec<Json>)>>,
    sem: Arc<Semaphore>,
}

struct NodeSt {
    title: String,
    client: bool,
    status: String, // pending | queued | running | waiting_client | done | error | cancelled | skipped
    progress: f64,
    note: String,
    error: Option<String>,
    outs: Vec<Json>,
    message: Json,
    started_ms: Option<i64>,
    ended_ms: Option<i64>,
    cached: bool,
    pending_inputs: Option<Vec<Json>>,
}

struct RunSt {
    status: String,
    ended_ms: Option<i64>,
    nodes: BTreeMap<i64, NodeSt>,
}

struct Run {
    id: String,
    graph: Option<String>,
    name: String,
    started_ms: i64,
    cancel: Arc<AtomicBool>,
    cancel_n: Notify,
    st: Mutex<RunSt>,
    waiters: Mutex<HashMap<i64, oneshot::Sender<Result<(Vec<Json>, Json), String>>>>,
}

impl Run {
    fn set(&self, id: i64, f: impl FnOnce(&mut NodeSt)) {
        if let Some(n) = self.st.lock().unwrap().nodes.get_mut(&id) {
            f(n);
        }
    }

    fn to_json(&self) -> Json {
        let st = self.st.lock().unwrap();
        let nodes: Map<String, Json> = st
            .nodes
            .iter()
            .map(|(id, n)| {
                (id.to_string(), json!({"title": n.title, "client": n.client, "status": n.status, "progress": n.progress, "note": n.note, "error": n.error,
                    "outs": n.outs, "message": n.message, "started_ms": n.started_ms, "ended_ms": n.ended_ms, "cached": n.cached,
                    "inputs": n.pending_inputs.as_ref().filter(|_| n.status == "waiting_client")}))
            })
            .collect();
        json!({"run": self.id, "graph": self.graph, "name": self.name, "status": st.status, "started_ms": self.started_ms, "ended_ms": st.ended_ms, "nodes": nodes})
    }

    fn summary(&self) -> Json {
        let st = self.st.lock().unwrap();
        json!({"run": self.id, "graph": self.graph, "name": self.name, "status": st.status, "started_ms": self.started_ms, "ended_ms": st.ended_ms})
    }

    fn status(&self) -> String {
        self.st.lock().unwrap().status.clone()
    }
}

#[derive(Debug)]
struct PNode {
    id: i64,
    title: String,
    client: bool,
    cmd: String,
    options: Json,
    inputs: Vec<Json>,
    name: String,
    meta_in: Vec<Json>,
    meta_out: Vec<Json>,
    outs: Option<Vec<Json>>,
    deps: Vec<i64>,
}

fn parse_plan(p: &Json) -> Result<Vec<PNode>, Error> {
    let a = p.as_array().ok_or_else(|| Error::Invalid("plan must be an array".into()))?;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for n in a {
        let id = n["id"].as_i64().ok_or_else(|| Error::Invalid("plan node: id (integer) missing".into()))?;
        if !seen.insert(id) {
            return inv(format!("plan: duplicate node id {id}"));
        }
        let inputs: Vec<Json> = n["inputs"].as_array().cloned().unwrap_or_default();
        let deps: Vec<i64> = inputs.iter().filter_map(|r| r["node"].as_i64()).collect();
        out.push(PNode {
            id,
            title: n["title"].as_str().unwrap_or("").into(),
            client: n["client"].as_bool().unwrap_or(false),
            cmd: n["cmd"].as_str().unwrap_or("").into(),
            options: if n["options"].is_null() { json!({}) } else { n["options"].clone() },
            inputs,
            name: n["name"].as_str().unwrap_or("").into(),
            meta_in: n["meta_in"].as_array().cloned().unwrap_or_default(),
            meta_out: n["meta_out"].as_array().cloned().unwrap_or_default(),
            outs: n["outs"].as_array().cloned(),
            deps,
        });
    }
    for n in &out {
        if !n.client && n.cmd.is_empty() {
            return inv(format!("plan node {}: cmd missing", n.id));
        }
        if let Some(d) = n.deps.iter().find(|d| !seen.contains(d)) {
            return inv(format!("plan node {}: input from unknown node {d}", n.id));
        }
    }
    // cycle check (Kahn)
    let mut indeg: HashMap<i64, usize> = out.iter().map(|n| (n.id, n.deps.iter().collect::<HashSet<_>>().len())).collect();
    let mut q: Vec<i64> = indeg.iter().filter(|(_, &d)| d == 0).map(|(&i, _)| i).collect();
    let mut done = 0;
    while let Some(i) = q.pop() {
        done += 1;
        for n in &out {
            if n.deps.contains(&i) {
                let d = indeg.get_mut(&n.id).unwrap();
                *d -= 1;
                if *d == 0 {
                    q.push(n.id);
                }
            }
        }
    }
    if done != out.len() {
        return inv("plan has a cycle");
    }
    Ok(out)
}

fn resolve(r: &Json, results: &HashMap<i64, Vec<Json>>) -> Json {
    if let Some(n) = r["node"].as_i64() {
        return results.get(&n).and_then(|v| v.get(r["slot"].as_u64().unwrap_or(0) as usize)).cloned().unwrap_or(Json::Null);
    }
    r.get("lit").cloned().unwrap_or(Json::Null)
}

fn opt_str(o: &Json, k: &str) -> Result<String, Error> {
    o[k].as_str().map(String::from).ok_or_else(|| Error::Invalid(format!("options.{k} missing")))
}

fn safe_id(s: &str) -> Result<String, Error> {
    if !s.is_empty() && s.len() <= 64 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
        Ok(s.to_string())
    } else {
        inv("graph id: letters, digits, _ and - only")
    }
}

impl Jobs {
    pub fn new(bank: Arc<Bank>, registry: Arc<Registry>) -> Arc<Self> {
        let dir = std::env::var_os("HASTY_GRAPHS").map(PathBuf::from).unwrap_or_else(|| hasty_site::data().join("graphs"));
        let par = std::env::var("HASTY_RUN_PARALLEL").ok().and_then(|s| s.parse::<usize>().ok()).filter(|&p| p > 0).unwrap_or(2);
        Arc::new(Self { dir, bank, registry, runs: Mutex::new(Vec::new()), cache: Mutex::new(HashMap::new()), sem: Arc::new(Semaphore::new(par)) })
    }

    /// handles the graph_* / run_* commands; `None` for any other command. Must be called inside a tokio runtime.
    pub fn handle(self: &Arc<Self>, cmd: &str, options: &str) -> Option<Result<String, Error>> {
        if !COMMANDS.iter().any(|(n, _)| *n == cmd) {
            return None;
        }
        let o: Json = match if options.trim().is_empty() { Ok(json!({})) } else { serde_json::from_str(options) } {
            Ok(o) => o,
            Err(e) => return Some(inv(format!("{cmd}: options are not JSON: {e}"))),
        };
        Some(self.dispatch(cmd, &o).map(|j| j.to_string()))
    }

    fn dispatch(self: &Arc<Self>, cmd: &str, o: &Json) -> Result<Json, Error> {
        match cmd {
            "graph_save" => self.graph_save(o),
            "graph_list" => self.graph_list(),
            "graph_get" => {
                let id = safe_id(&opt_str(o, "id")?)?;
                let txt = std::fs::read_to_string(self.dir.join(format!("{id}.json"))).map_err(|_| Error::Invalid(format!("no graph {id}")))?;
                serde_json::from_str(&txt).map_err(|e| Error::Invalid(format!("graph {id}: {e}")))
            }
            "graph_delete" => {
                let id = safe_id(&opt_str(o, "id")?)?;
                let _ = std::fs::remove_file(self.dir.join(format!("{id}.json")));
                Ok(json!({"deleted": id}))
            }
            "run_submit" => self.submit(o),
            "run_get" => {
                let r = self.run(&opt_str(o, "run")?)?;
                Ok(r.to_json())
            }
            "run_list" => {
                let g = o["graph"].as_str();
                let v: Vec<Json> = self.runs.lock().unwrap().iter().rev().filter(|r| g.is_none() || r.graph.as_deref() == g).map(|r| r.summary()).collect();
                Ok(json!({"runs": v}))
            }
            "run_cancel" => {
                let r = self.run(&opt_str(o, "run")?)?;
                r.cancel.store(true, Ordering::SeqCst);
                r.cancel_n.notify_waiters();
                Ok(json!({"cancelled": r.id}))
            }
            "run_client_claim" => {
                let r = self.run(&opt_str(o, "run")?)?;
                let id = o["node"].as_i64().ok_or_else(|| Error::Invalid("options.node".into()))?;
                let res = {
                    let mut st = r.st.lock().unwrap();
                    let n = st.nodes.get_mut(&id).ok_or_else(|| Error::Invalid(format!("no node {id}")))?;
                    if n.status != "waiting_client" {
                        return inv(format!("node {id} is {} (not waiting for a client)", n.status));
                    }
                    n.status = "running".into();
                    n.started_ms = Some(now_ms());
                    json!({"inputs": n.pending_inputs.clone().unwrap_or_default()})
                };
                Self::refresh_status(&r);
                Ok(res)
            }
            "run_client_done" => {
                let r = self.run(&opt_str(o, "run")?)?;
                let id = o["node"].as_i64().ok_or_else(|| Error::Invalid("options.node".into()))?;
                let res = match o["error"].as_str() {
                    Some(e) => Err(e.to_string()),
                    None => Ok((o["outs"].as_array().cloned().unwrap_or_default(), o["message"].clone())),
                };
                let waiter = r.waiters.lock().unwrap().remove(&id);
                match waiter {
                    Some(tx) => {
                        let _ = tx.send(res);
                        Ok(json!({"ok": true}))
                    }
                    None => inv(format!("node {id} is not waiting for a client result")),
                }
            }
            _ => unreachable!(),
        }
    }

    fn run(&self, id: &str) -> Result<Arc<Run>, Error> {
        self.runs.lock().unwrap().iter().find(|r| r.id == id).cloned().ok_or_else(|| Error::Invalid(format!("no run {id} (runs are not kept across server restarts)")))
    }

    // ── graphs ──────────────────────────────────────────────────────────────────────────────────────────────────

    fn graph_save(&self, o: &Json) -> Result<Json, Error> {
        std::fs::create_dir_all(&self.dir)?;
        let id = match o["id"].as_str() {
            Some(s) if !s.is_empty() => safe_id(s)?,
            _ => Uuid::new_v4().simple().to_string()[..8].to_string(),
        };
        if !o["graph"].is_object() {
            return inv("graph_save: options.graph (object) missing");
        }
        let name = o["name"].as_str().filter(|s| !s.is_empty()).unwrap_or("untitled").to_string();
        let doc = json!({"id": id, "name": name, "updated_ms": now_ms(), "graph": o["graph"]});
        let tmp = self.dir.join(format!(".{id}.tmp"));
        std::fs::write(&tmp, doc.to_string())?;
        std::fs::rename(&tmp, self.dir.join(format!("{id}.json")))?;
        Ok(json!({"id": id, "name": name}))
    }

    fn graph_list(&self) -> Result<Json, Error> {
        let mut v = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                let Ok(txt) = std::fs::read_to_string(&p) else { continue };
                let Ok(d) = serde_json::from_str::<Json>(&txt) else { continue };
                let id = d["id"].as_str().unwrap_or("").to_string();
                let last = self.runs.lock().unwrap().iter().rev().find(|r| r.graph.as_deref() == Some(id.as_str())).cloned();
                v.push(json!({"id": id, "name": d["name"], "updated_ms": d["updated_ms"], "nodes": d["graph"]["nodes"].as_array().map(|a| a.len()).unwrap_or(0),
                    "status": last.as_ref().map(|r| r.status()).unwrap_or_else(|| "idle".into()), "run": last.as_ref().map(|r| r.id.clone()), "run_started_ms": last.as_ref().map(|r| r.started_ms)}));
            }
        }
        v.sort_by_key(|g| std::cmp::Reverse(g["updated_ms"].as_i64().unwrap_or(0)));
        Ok(json!({"graphs": v}))
    }

    // ── runs ────────────────────────────────────────────────────────────────────────────────────────────────────

    fn submit(self: &Arc<Self>, o: &Json) -> Result<Json, Error> {
        let plan = parse_plan(&o["plan"])?;
        if plan.is_empty() {
            return inv("run_submit: empty plan");
        }
        let id = Uuid::new_v4().simple().to_string()[..12].to_string();
        let nodes = plan
            .iter()
            .map(|n| {
                (n.id, NodeSt { title: n.title.clone(), client: n.client, status: "pending".into(), progress: 0.0, note: String::new(), error: None, outs: vec![], message: Json::Null, started_ms: None, ended_ms: None, cached: false, pending_inputs: None })
            })
            .collect();
        let run = Arc::new(Run {
            id: id.clone(),
            graph: o["graph"].as_str().map(String::from),
            name: o["name"].as_str().unwrap_or("").into(),
            started_ms: now_ms(),
            cancel: Arc::new(AtomicBool::new(false)),
            cancel_n: Notify::new(),
            st: Mutex::new(RunSt { status: "running".into(), ended_ms: None, nodes }),
            waiters: Mutex::new(HashMap::new()),
        });
        {
            let mut runs = self.runs.lock().unwrap();
            runs.push(run.clone());
            let n = runs.len();
            if n > KEEP_RUNS {
                runs.drain(..n - KEEP_RUNS);
            }
        }
        let me = self.clone();
        tokio::spawn(async move { me.drive(run, plan).await });
        Ok(json!({"run": id}))
    }

    fn refresh_status(run: &Run) {
        let mut st = run.st.lock().unwrap();
        let any = |s: &str| st.nodes.values().any(|n| n.status == s);
        // pending nodes behind a waiting client step do not make the run "running"
        let st_new = if any("running") || any("queued") {
            "running"
        } else if any("waiting_client") {
            "waiting_client"
        } else if any("pending") {
            "running"
        } else if any("error") {
            "error"
        } else if any("cancelled") {
            "cancelled"
        } else {
            "done"
        };
        let active = st_new == "running" || st_new == "waiting_client";
        st.status = st_new.into();
        if !active && st.ended_ms.is_none() {
            st.ended_ms = Some(now_ms());
        }
    }

    async fn drive(self: Arc<Self>, run: Arc<Run>, plan: Vec<PNode>) {
        let nodes: HashMap<i64, Arc<PNode>> = plan.into_iter().map(|n| (n.id, Arc::new(n))).collect();
        let mut order: Vec<i64> = nodes.keys().copied().collect();
        order.sort_unstable();
        let results: Arc<Mutex<HashMap<i64, Vec<Json>>>> = Arc::new(Mutex::new(HashMap::new()));
        let (tx, mut rx) = mpsc::unbounded_channel::<(i64, Result<(), String>)>();
        let mut pending: Vec<i64> = order;
        let mut running = 0usize;
        let mut ok: HashSet<i64> = HashSet::new();
        let mut bad: HashSet<i64> = HashSet::new();
        loop {
            // launch / skip until nothing changes
            loop {
                let mut changed = false;
                let cancelled = run.cancel.load(Ordering::SeqCst);
                let mut keep = Vec::new();
                for id in std::mem::take(&mut pending) {
                    let n = &nodes[&id];
                    if cancelled {
                        run.set(id, |s| s.status = "cancelled".into());
                        bad.insert(id);
                        changed = true;
                    } else if n.deps.iter().any(|d| bad.contains(d)) {
                        run.set(id, |s| {
                            s.status = "skipped".into();
                            s.note = "an input failed".into();
                        });
                        bad.insert(id);
                        changed = true;
                    } else if n.deps.iter().all(|d| ok.contains(d)) {
                        run.set(id, |s| s.status = "queued".into());
                        let (me, run2, n2, res, tx2) = (self.clone(), run.clone(), n.clone(), results.clone(), tx.clone());
                        tokio::spawn(async move {
                            let r = me.exec_node(&run2, &n2, &res).await;
                            let _ = tx2.send((n2.id, r));
                        });
                        running += 1;
                        changed = true;
                    } else {
                        keep.push(id);
                    }
                }
                pending = keep;
                if !changed {
                    break;
                }
            }
            Self::refresh_status(&run);
            if running == 0 {
                break;
            }
            let Some((id, r)) = rx.recv().await else { break };
            running -= 1;
            match r {
                Ok(()) => {
                    ok.insert(id);
                }
                Err(_) => {
                    bad.insert(id);
                }
            }
        }
        Self::refresh_status(&run);
    }

    async fn exec_node(self: &Arc<Self>, run: &Arc<Run>, n: &Arc<PNode>, results: &Arc<Mutex<HashMap<i64, Vec<Json>>>>) -> Result<(), String> {
        let inputs: Vec<Json> = {
            let r = results.lock().unwrap();
            n.inputs.iter().map(|x| resolve(x, &r)).collect()
        };
        let fail = |run: &Run, e: String| {
            run.set(n.id, |s| {
                s.status = if e == "cancelled" { "cancelled".into() } else { "error".into() };
                s.error = Some(e.clone());
                s.ended_ms = Some(now_ms());
            });
            Self::refresh_status(run);
            Err(e)
        };
        if n.client {
            let (tx, rx) = oneshot::channel();
            run.waiters.lock().unwrap().insert(n.id, tx);
            run.set(n.id, |s| {
                s.status = "waiting_client".into();
                s.pending_inputs = Some(inputs.clone());
            });
            Self::refresh_status(run);
            let got = tokio::select! {
                r = rx => r.map_err(|_| "client step dropped".to_string()).and_then(|x| x),
                _ = run.cancel_n.notified() => Err("cancelled".to_string()),
            };
            run.waiters.lock().unwrap().remove(&n.id);
            return match got {
                Ok((outs, msg)) => {
                    results.lock().unwrap().insert(n.id, outs.clone());
                    run.set(n.id, |s| {
                        s.status = "done".into();
                        s.outs = outs;
                        s.message = msg;
                        s.progress = 1.0;
                        s.pending_inputs = None;
                        s.ended_ms = Some(now_ms());
                    });
                    Self::refresh_status(run);
                    Ok(())
                }
                Err(e) => fail(run, e),
            };
        }
        // server step: wait for a slot (cheap steps skip the queue)
        let cheap = n.cmd.starts_with('_');
        let _permit = if cheap {
            None
        } else {
            tokio::select! {
                p = self.sem.clone().acquire_owned() => p.ok(),
                _ = run.cancel_n.notified() => return fail(run, "cancelled".into()),
            }
        };
        if run.cancel.load(Ordering::SeqCst) {
            return fail(run, "cancelled".into());
        }
        run.set(n.id, |s| {
            s.status = "running".into();
            s.started_ms = Some(now_ms());
        });
        Self::refresh_status(run);
        let fp = json!([n.cmd, n.options, inputs, n.meta_in, n.meta_out, n.outs, n.name]).to_string();
        let key = run.graph.as_ref().map(|g| format!("{g}:{}", n.id));
        if let (false, Some(k)) = (cheap, &key) {
            let hit = self.cache.lock().unwrap().get(k).filter(|(f, _)| *f == fp).map(|(_, o)| o.clone());
            if let Some(outs) = hit {
                if outs.iter().filter_map(as_id).all(|u| self.bank.info(&u).is_ok()) {
                    results.lock().unwrap().insert(n.id, outs.clone());
                    run.set(n.id, |s| {
                        s.status = "done".into();
                        s.outs = outs;
                        s.cached = true;
                        s.progress = 1.0;
                        s.ended_ms = Some(now_ms());
                    });
                    Self::refresh_status(run);
                    return Ok(());
                }
            }
        }
        let (me, run2, n2, inputs2) = (self.clone(), run.clone(), n.clone(), inputs.clone());
        let res = tokio::task::spawn_blocking(move || me.run_step(&run2, &n2, &inputs2)).await.map_err(|e| e.to_string()).and_then(|r| r.map_err(|e| e.to_string()));
        match res {
            Ok((outs, msg)) => {
                if let Some(k) = key {
                    if !cheap {
                        self.cache.lock().unwrap().insert(k, (fp, outs.clone()));
                    }
                }
                results.lock().unwrap().insert(n.id, outs.clone());
                run.set(n.id, |s| {
                    s.status = "done".into();
                    s.outs = outs;
                    s.message = msg;
                    s.progress = 1.0;
                    s.ended_ms = Some(now_ms());
                });
                Self::refresh_status(run);
                Ok(())
            }
            Err(e) => fail(run, e),
        }
    }

    /// blocking part of a server step: resolve inputs, run the command, store outputs, write metadata, pick node outputs
    fn run_step(&self, run: &Arc<Run>, n: &PNode, inputs: &[Json]) -> Result<(Vec<Json>, Json), Error> {
        if n.cmd == "_const" {
            return Ok((vec![n.options["value"].clone()], Json::Null));
        }
        if n.cmd == "_pass" {
            return Ok((inputs.to_vec(), Json::Null));
        }
        let mut values: Vec<Arc<Value>> = Vec::new();
        let mut in_meta: Vec<Json> = Vec::new();
        let mut in_ids: Vec<Option<Uuid>> = Vec::new();
        for (i, j) in inputs.iter().enumerate() {
            let u = as_id(j).ok_or_else(|| Error::Invalid(format!("{}: input {i} is not a value id ({j})", n.cmd)))?;
            values.push(self.bank.value(&u)?);
            let m = self.bank.info(&u)?.metadata;
            in_meta.push(serde_json::from_str(&m).unwrap_or(Json::Null));
            in_ids.push(Some(u));
        }
        let mut opts = n.options.clone();
        for m in &n.meta_in {
            let k = m["input"].as_u64().unwrap_or(0) as usize;
            if let (Some(keys), Some(meta)) = (m["keys"].as_object(), in_meta.get(k)) {
                for (dst, src) in keys {
                    let missing = opts.get(dst).map(|v| v.is_null()).unwrap_or(true);
                    if let (true, Some(v)) = (missing, src.as_str().and_then(|s| meta.get(s))) {
                        opts[dst] = v.clone();
                    }
                }
            }
        }
        let (r2, nid) = (run.clone(), n.id);
        let ctx = Ctx {
            bank: self.bank.clone(),
            input_meta: in_meta.clone(),
            progress: Arc::new(move |f, note| r2.set(nid, |s| {
                s.progress = f.clamp(0.0, 1.0);
                s.note = note.to_string();
            })),
            cancel: run.cancel.clone(),
        };
        let (msg, outs) = self.registry.run(&n.cmd, &ctx, &opts.to_string(), &values)?;
        let name = if n.name.is_empty() { n.cmd.clone() } else { n.name.clone() };
        let k = outs.len();
        let out_ids: Vec<Uuid> = outs.into_iter().enumerate().map(|(i, v)| self.bank.put(&if k > 1 { format!("{name}[{i}]") } else { name.clone() }, v, "{}")).collect();
        let message: Json = if msg.is_empty() { Json::Null } else { serde_json::from_str(&msg).unwrap_or(Json::String(msg)) };
        for m in &n.meta_out {
            let Some(id) = m["output"].as_u64().and_then(|i| out_ids.get(i as usize)) else { continue };
            let mut meta = Map::new();
            if let Some(fi) = m.get("from_input") {
                let src = in_meta.get(fi["input"].as_u64().unwrap_or(0) as usize);
                for key in fi["keys"].as_array().into_iter().flatten().filter_map(|k| k.as_str()) {
                    if let Some(v) = src.and_then(|s| s.get(key)) {
                        meta.insert(key.into(), v.clone());
                    }
                }
            }
            for (dst, src) in m["from_message"].as_object().into_iter().flatten() {
                if let Some(v) = src.as_str().and_then(|s| message.get(s)) {
                    meta.insert(dst.clone(), v.clone());
                }
            }
            for (key, i) in m["set_input_id"].as_object().into_iter().flatten() {
                if let Some(u) = i.as_u64().and_then(|i| in_ids.get(i as usize)).copied().flatten() {
                    meta.insert(key.clone(), Json::String(u.simple().to_string()));
                }
            }
            for (key, v) in m["set"].as_object().into_iter().flatten() {
                meta.insert(key.clone(), v.clone());
            }
            self.bank.set_metadata(id, &Json::Object(meta).to_string())?;
        }
        let ids: Vec<Json> = out_ids.iter().map(|u| Json::String(u.simple().to_string())).collect();
        let node_outs = match &n.outs {
            None => ids,
            Some(sel) => sel
                .iter()
                .map(|s| {
                    if let Some(k) = s["output"].as_u64() {
                        ids.get(k as usize).cloned().unwrap_or(Json::Null)
                    } else if let Some(k) = s["input"].as_u64() {
                        inputs.get(k as usize).cloned().unwrap_or(Json::Null)
                    } else if let Some(k) = s["message_key"].as_str() {
                        message.get(k).cloned().unwrap_or(Json::Null)
                    } else if s["message"].as_bool() == Some(true) {
                        message.clone()
                    } else {
                        Json::Null
                    }
                })
                .collect(),
        };
        Ok((node_outs, message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hasty_value::Tensor;

    #[test]
    fn plan_rejects_cycles_and_unknown_inputs() {
        let ok = json!([{"id": 1, "cmd": "rand"}, {"id": 2, "cmd": "neg", "inputs": [{"node": 1, "slot": 0}]}]);
        assert!(parse_plan(&ok).is_ok());
        let cyc = json!([{"id": 1, "cmd": "neg", "inputs": [{"node": 2, "slot": 0}]}, {"id": 2, "cmd": "neg", "inputs": [{"node": 1, "slot": 0}]}]);
        assert!(parse_plan(&cyc).unwrap_err().to_string().contains("cycle"));
        let unk = json!([{"id": 1, "cmd": "neg", "inputs": [{"node": 9, "slot": 0}]}]);
        assert!(parse_plan(&unk).unwrap_err().to_string().contains("unknown node"));
        assert!(parse_plan(&json!([{"id": 1, "cmd": "a"}, {"id": 1, "cmd": "b"}])).is_err());
        assert!(parse_plan(&json!([{"id": 1}])).is_err());
    }

    #[test]
    fn fft_norm_conventions() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(Bank::new()));
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7).sin() + 0.3 * i as f32).collect();
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&[4, 4], &x).unwrap()));
        let energy = |v: &Value| -> f64 { let t = v.as_tensor().unwrap(); (0..t.numel()).map(|i| { let (a, b) = t.get_f64(i); a * a + b * b }).sum() };
        let ex = energy(&t);
        let run = |cmd: &str, norm: &str, inp: &Arc<Value>| reg.run(cmd, &ctx, &json!({"norm": norm}).to_string(), std::slice::from_ref(inp)).unwrap().1.remove(0);
        // ortho: unitary (energy preserved), round trip exact
        let f = run("fft", "ortho", &t);
        assert!((energy(&f) - ex).abs() < 1e-3 * ex);
        let back = run("ifft", "ortho", &Arc::new(f));
        let bt = back.as_tensor().unwrap();
        assert!((0..16).all(|i| (bt.get_f64(i).0 - x[i] as f64).abs() < 1e-4 && bt.get_f64(i).1.abs() < 1e-4));
        // forward: the 1/n sits on the forward transform: DC = mean
        let f2 = run("fft", "forward", &t);
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / 16.0;
        assert!((f2.as_tensor().unwrap().get_f64(0).0 - mean).abs() < 1e-4);
        // backward (default) keeps the old behaviour: DC = sum
        let f3 = run("fft", "backward", &t);
        assert!((f3.as_tensor().unwrap().get_f64(0).0 - mean * 16.0).abs() < 1e-3);
        assert!(reg.run("fft", &ctx, r#"{"norm":"bogus"}"#, std::slice::from_ref(&t)).is_err());
    }
}
