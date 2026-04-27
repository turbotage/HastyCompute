/**
 * main.js — HastyEdit node editor entry point.
 *
 * litegraph.js is loaded as a plain <script> tag (sets window.LiteGraph)
 * before this ES module is evaluated.
 *
 * Node categories and their backend mapping:
 *   IO     — LoadTensor, SaveTensor
 *   Create — RandTensor, ZerosTensor, OnesTensor  (cmd 6, 7, 8)
 *   Math   — Add, Sub, Mult, Div, Neg, Abs        (cmd 0-5)
 *            Scalar  (no backend, plain number)
 *   FFT    — FFT, IFFT, NUFFT                     (stubs)
 *   MRI    — CoilSense, SensitivityMaps, Toeplitz (stubs)
 *   Viz    — OrthoSlicer
 */

import { runGraph }                               from './node_runner.js';
import { execute, fetchMeta, uuidToHex,
         hexToUuid, COMMAND_IDS }                 from './grpc_client.js';

const LG = window.LiteGraph;

// -- Theme --------------------------------------------------------------------
LG.NODE_DEFAULT_COLOR        = "#141820";
LG.NODE_DEFAULT_BGCOLOR      = "#0e1218";
LG.NODE_DEFAULT_BOXCOLOR     = "#3a6090";
LG.NODE_SELECTED_TITLE_COLOR = "#7ab4d8";
LG.DEFAULT_SHADOW_COLOR      = "transparent";
LG.CONNECTING_LINK_COLOR     = "#5a9abc";
LG.LINK_COLOR                = "#3a6090";
LG.EVENT_LINK_COLOR          = "#60b070";
LG.RENDER_CONNECTIONS_BORDER = false;

// -- Shared helpers -----------------------------------------------------------

/**
 * Draw an execution status bar in the bottom 22px of the node body.
 * Call from onDrawForeground (coordinate system: node body, top-left origin).
 */
function drawStatusBar(ctx, node) {
    if (!node._execStatus) return;
    const w    = node.size[0];
    const h    = node.size[1];
    const barH = 22;
    const y    = h - barH;

    const BG = { running: '#2a1800', done: '#001810', error: '#1a0006' };
    const FG = { running: '#cc8800', done:  '#44cc88', error:  '#ee4444' };
    ctx.fillStyle = BG[node._execStatus] ?? '#111';
    ctx.fillRect(3, y + 2, w - 6, barH - 4);

    ctx.fillStyle    = FG[node._execStatus] ?? '#888';
    ctx.font         = '10px monospace';
    ctx.textBaseline = 'middle';

    let text = '';
    if (node._execStatus === 'running') {
        text = 'running...';
    } else if (node._execStatus === 'done') {
        const uuid = node._execResult?.[0];
        if (uuid) {
            const hex  = uuidToHex(uuid).slice(0, 8);
            const meta = node._execMeta;
            text = meta
                ? hex + '... ' + meta.dtype + '[' + meta.shape.join('x') + ']'
                : hex + '...';
        } else {
            text = 'done';
        }
    } else if (node._execStatus === 'error') {
        text = 'ERR: ' + (node._execError ?? 'error').slice(0, 28);
    }

    ctx.fillText(text, 6, y + 2 + (barH - 4) / 2);
}

/**
 * Build a tensor-creation options string for from_metadata_string().
 * Format: "Tensor[dtype=f32,device=cpu,shape=(4,4)]"
 */
function makeTensorOptions(dtype, device, shapeStr) {
    const s = shapeStr.trim().replace(/^\(|\)$/g, '').replace(/\s+/g, ',');
    return 'Tensor[dtype=' + dtype + ',device=' + device + ',shape=(' + s + ')]';
}

// -- Node definitions ---------------------------------------------------------

function registerNodes() {

    // IO -----------------------------------------------------------------------

    class LoadTensorNode extends LG.LGraphNode {
        static title    = "Load Tensor";
        static category = "IO";
        constructor() {
            super();
            this.addWidget("text", "uuid (hex)", "", v => { this.properties.uuid = v; });
            this.addOutput("tensor", "tensor_uuid");
            this.size = [240, 60];
        }
        async executeAsync(_inputs) {
            const hex = (this.properties.uuid ?? '').replace(/\s/g, '');
            if (hex.length !== 32) throw new Error('UUID must be 32 hex chars');
            const uuid = hexToUuid(hex);
            this._execMeta = await fetchMeta(uuid);
            return [uuid];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class SaveTensorNode extends LG.LGraphNode {
        static title    = "Save Tensor";
        static category = "IO";
        constructor() {
            super();
            this.addInput("tensor", "tensor_uuid");
            this.addWidget("text", "label", "result", v => { this.properties.label = v; });
            this.size = [200, 60];
        }
        async executeAsync([uuid]) {
            if (!uuid) throw new Error('No tensor connected');
            this._execMeta = await fetchMeta(uuid);
            return [uuid];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    // Tensor creation ----------------------------------------------------------

    class TensorCreationBase extends LG.LGraphNode {
        constructor(cmdId) {
            super();
            this._cmdId = cmdId;
            this.addWidget("combo", "dtype",  "f32", v => { this.properties.dtype  = v; },
                { values: ["f32","f64","c32","c64","i32","i64"] });
            this.addWidget("combo", "device", "cpu", v => { this.properties.device = v; },
                { values: ["cpu","cuda:0"] });
            this.addWidget("text",  "shape",  "4,4", v => { this.properties.shape  = v; });
            this.addOutput("tensor", "tensor_uuid");
            this.size = [200, 102];
        }
        async executeAsync(_inputs) {
            const opts = makeTensorOptions(
                this.properties.dtype  ?? 'f32',
                this.properties.device ?? 'cpu',
                this.properties.shape  ?? '4,4');
            return await execute(this._cmdId, [], opts);
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class RandTensorNode extends TensorCreationBase {
        static title = "Rand Tensor";  static category = "Create";
        constructor() { super(COMMAND_IDS.rand); }
    }
    class ZerosTensorNode extends TensorCreationBase {
        static title = "Zeros Tensor"; static category = "Create";
        constructor() { super(COMMAND_IDS.zeros); }
    }
    class OnesTensorNode extends TensorCreationBase {
        static title = "Ones Tensor";  static category = "Create";
        constructor() { super(COMMAND_IDS.ones); }
    }

    // Binary arithmetic --------------------------------------------------------

    class BinaryArithBase extends LG.LGraphNode {
        constructor(cmdId) {
            super();
            this._cmdId = cmdId;
            this.addInput("A",    "tensor_uuid");
            this.addInput("B",    "tensor_uuid");
            this.addOutput("out", "tensor_uuid");
            this.size = [150, 62];
        }
        async executeAsync([a, b]) {
            if (!a) throw new Error('Input A not connected');
            if (!b) throw new Error('Input B not connected');
            return await execute(this._cmdId, [a, b]);
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class AddNode  extends BinaryArithBase {
        static title = "Add";  static category = "Math";
        constructor() { super(COMMAND_IDS.add); }
    }
    class SubNode  extends BinaryArithBase {
        static title = "Sub";  static category = "Math";
        constructor() { super(COMMAND_IDS.sub); }
    }
    class MultNode extends BinaryArithBase {
        static title = "Mult"; static category = "Math";
        constructor() { super(COMMAND_IDS.mult); }
    }
    class DivNode  extends BinaryArithBase {
        static title = "Div";  static category = "Math";
        constructor() { super(COMMAND_IDS.div); }
    }

    // Unary arithmetic ---------------------------------------------------------

    class UnaryArithBase extends LG.LGraphNode {
        constructor(cmdId) {
            super();
            this._cmdId = cmdId;
            this.addInput("in",   "tensor_uuid");
            this.addOutput("out", "tensor_uuid");
            this.size = [140, 56];
        }
        async executeAsync([a]) {
            if (!a) throw new Error('Input not connected');
            return await execute(this._cmdId, [a]);
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class NegNode extends UnaryArithBase {
        static title = "Neg"; static category = "Math";
        constructor() { super(COMMAND_IDS.neg); }
    }
    class AbsNode extends UnaryArithBase {
        static title = "Abs"; static category = "Math";
        constructor() { super(COMMAND_IDS.abs); }
    }

    // Scalar (no backend) ------------------------------------------------------

    class ScalarNode extends LG.LGraphNode {
        static title = "Scalar"; static category = "Math";
        constructor() {
            super();
            this.addWidget("number", "value", 1.0, v => { this.properties.value = v; });
            this.addOutput("out", "number");
            this.size = [160, 50];
        }
        onExecute() { this.setOutputData(0, this.properties.value ?? 1.0); }
    }

    // FFT stubs ----------------------------------------------------------------

    class FFTNode extends LG.LGraphNode {
        static title = "FFT"; static category = "FFT";
        constructor() {
            super();
            this.addInput("in",      "tensor_uuid");
            this.addOutput("kspace", "tensor_uuid");
            this.addWidget("combo", "norm", "ortho", null,
                { values: ["none","ortho","forward","backward"] });
            this.size = [160, 70];
        }
    }
    class IFFTNode extends LG.LGraphNode {
        static title = "IFFT"; static category = "FFT";
        constructor() {
            super();
            this.addInput("kspace", "tensor_uuid");
            this.addOutput("out",   "tensor_uuid");
            this.addWidget("combo", "norm", "ortho", null,
                { values: ["none","ortho","forward","backward"] });
            this.size = [160, 70];
        }
    }
    class NUFFTNode extends LG.LGraphNode {
        static title = "NUFFT"; static category = "FFT";
        constructor() {
            super();
            this.addInput("image",  "tensor_uuid");
            this.addInput("coords", "tensor_uuid");
            this.addOutput("kspace","tensor_uuid");
            this.size = [160, 70];
        }
    }

    // MRI stubs ----------------------------------------------------------------

    class CoilSenseNode extends LG.LGraphNode {
        static title = "Coil Sense"; static category = "MRI";
        constructor() {
            super();
            this.addInput("image",      "tensor_uuid");
            this.addInput("smaps",      "tensor_uuid");
            this.addOutput("coil_imgs", "tensor_uuid");
            this.size = [170, 70];
        }
    }
    class SensitivityMapNode extends LG.LGraphNode {
        static title = "Sensitivity Maps"; static category = "MRI";
        constructor() {
            super();
            this.addInput("kspace", "tensor_uuid");
            this.addOutput("smaps", "tensor_uuid");
            this.size = [170, 56];
        }
    }
    class ToeplitzNode extends LG.LGraphNode {
        static title = "Toeplitz"; static category = "MRI";
        constructor() {
            super();
            this.addInput("ktraj",   "tensor_uuid");
            this.addInput("weights", "tensor_uuid");
            this.addOutput("kernel", "tensor_uuid");
            this.size = [170, 70];
        }
    }

    // Viz ----------------------------------------------------------------------

    class OrthoSlicerNode extends LG.LGraphNode {
        static title = "Ortho Slicer"; static category = "Viz";
        constructor() {
            super();
            this.addInput("volume", "tensor_uuid");
            this.addWidget("button", "Open Viewer", null,
                () => window.open("../orthoslicer/", "_blank"));
            this.size = [175, 70];
        }
    }

    // Register all -------------------------------------------------------------

    for (const cls of [
        LoadTensorNode, SaveTensorNode,
        RandTensorNode, ZerosTensorNode, OnesTensorNode,
        AddNode, SubNode, MultNode, DivNode, NegNode, AbsNode, ScalarNode,
        FFTNode, IFFTNode, NUFFTNode,
        CoilSenseNode, SensitivityMapNode, ToeplitzNode,
        OrthoSlicerNode,
    ]) {
        LG.registerNodeType(cls.category + '/' + cls.title, cls);
    }
}

// -- Sidebar ------------------------------------------------------------------

let _lgCanvas;

function buildSidebar() {
    const container = document.getElementById("node-list");
    const groups = {};
    for (const [type, cls] of Object.entries(LG.registered_node_types)) {
        const [group] = type.split("/");
        (groups[group] ??= []).push({ type, title: cls.title ?? type.split("/")[1] });
    }
    for (const [group, nodes] of Object.entries(groups)) {
        const label = document.createElement("div");
        label.className   = "node-group-label";
        label.textContent = group;
        container.appendChild(label);
        for (const { type, title } of nodes) {
            const entry       = document.createElement("div");
            entry.className   = "node-entry";
            entry.textContent = title;
            entry.title       = type;
            entry.draggable   = true;
            entry.addEventListener("dragstart", e => {
                e.dataTransfer.setData("nodetype", type);
            });
            entry.addEventListener("dblclick", () => {
                const node = LG.createNode(type);
                const rect = _lgCanvas.canvas.getBoundingClientRect();
                node.pos = _lgCanvas.convertOffsetToCanvas(
                    [rect.width / 2, rect.height / 2]);
                _lgCanvas.graph.add(node);
            });
            container.appendChild(entry);
        }
    }
}

// -- Drop handler -------------------------------------------------------------

function setupDrop(lgCanvas) {
    const el = lgCanvas.canvas;
    el.addEventListener("dragover", e => e.preventDefault());
    el.addEventListener("drop", e => {
        e.preventDefault();
        const type = e.dataTransfer.getData("nodetype");
        if (!type) return;
        const rect = el.getBoundingClientRect();
        const pos  = lgCanvas.convertOffsetToCanvas(
            [e.clientX - rect.left, e.clientY - rect.top]);
        const node = LG.createNode(type);
        node.pos = pos;
        lgCanvas.graph.add(node);
    });
}

// -- Toolbar ------------------------------------------------------------------

function setupToolbar(graph, lgCanvas) {
    const btnRun = document.getElementById("btn-run");

    btnRun.addEventListener("click", async () => {
        btnRun.disabled    = true;
        btnRun.textContent = 'Running...';
        try {
            await runGraph(graph, lgCanvas);
        } finally {
            btnRun.disabled    = false;
            btnRun.textContent = '\u25B6 Run';
        }
    });

    document.getElementById("btn-clear").addEventListener("click", () => {
        if (confirm("Clear graph?")) graph.clear();
    });

    document.getElementById("btn-save").addEventListener("click", () => {
        const json = JSON.stringify(graph.serialize(), null, 2);
        const a    = document.createElement("a");
        a.href     = URL.createObjectURL(new Blob([json], { type: "application/json" }));
        a.download = "graph.json";
        a.click();
    });

    document.getElementById("btn-load").addEventListener("click", () => {
        const inp    = document.createElement("input");
        inp.type     = "file";
        inp.accept   = ".json";
        inp.onchange = async () => {
            const text = await inp.files[0].text();
            graph.configure(JSON.parse(text));
        };
        inp.click();
    });
}

// -- Init ---------------------------------------------------------------------

function init() {
    registerNodes();

    const graph    = new LG.LGraph();
    const canvasEl = document.getElementById("graph-canvas");

    function resize() {
        const main      = canvasEl.parentElement;
        canvasEl.width  = main.clientWidth;
        canvasEl.height = main.clientHeight;
    }
    resize();
    window.addEventListener("resize", () => { resize(); _lgCanvas?.setDirty(true, true); });

    _lgCanvas = new LG.LGraphCanvas(canvasEl, graph);
    _lgCanvas.background_image     = null;
    _lgCanvas.render_shadows        = false;
    _lgCanvas.render_canvas_border  = false;
    _lgCanvas.node_title_color      = "#9ab8d0";
    _lgCanvas.default_connection_color.input_on  = "#5a9abc";
    _lgCanvas.default_connection_color.output_on = "#5a9abc";

    buildSidebar();
    setupDrop(_lgCanvas);
    setupToolbar(graph, _lgCanvas);

    graph.start();

    // Starter note
    const note = LG.createNode("Note");
    if (note) {
        note.pos = [60, 60];
        note.properties.text =
            "Drag nodes from the sidebar or double-click them.\n" +
            "Connect outputs to inputs, then press Run\n" +
            "to execute against the C++ backend.";
        graph.add(note);
    }
}

init();
