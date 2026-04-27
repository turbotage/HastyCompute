// Node editor — litegraph.js integration
// litegraph is loaded as a global script (window.LiteGraph) before this module runs

const LG = window.LiteGraph;

// ── Theme ───────────────────────────────────────────────────────────────────
LG.NODE_DEFAULT_COLOR        = "#141820";
LG.NODE_DEFAULT_BGCOLOR      = "#0e1218";
LG.NODE_DEFAULT_BOXCOLOR     = "#3a6090";
LG.NODE_SELECTED_TITLE_COLOR = "#7ab4d8";
LG.DEFAULT_SHADOW_COLOR      = "transparent";
LG.CONNECTING_LINK_COLOR     = "#5a9abc";
LG.LINK_COLOR                = "#3a6090";
LG.EVENT_LINK_COLOR          = "#60b070";
LG.RENDER_CONNECTIONS_BORDER = false;

// ── Node definitions ────────────────────────────────────────────────────────

function registerNodes() {
    // ── I/O ─────────────────────────────────────────────────────────────────
    class LoadTensorNode extends LG.LGraphNode {
        static title = "Load Tensor";
        static category = "IO";
        constructor() {
            super();
            this.addWidget("text", "uuid", "", "uuid");
            this.addOutput("tensor", "tensor");
            this.size = [180, 50];
        }
        onExecute() { /* future: fetch via gRPC / REST */ }
    }

    class SaveTensorNode extends LG.LGraphNode {
        static title = "Save Tensor";
        static category = "IO";
        constructor() {
            super();
            this.addInput("tensor", "tensor");
            this.addWidget("text", "uuid", "output", "uuid");
            this.size = [180, 50];
        }
        onExecute() {}
    }

    // ── Math ────────────────────────────────────────────────────────────────
    class AddNode extends LG.LGraphNode {
        static title = "Add";
        static category = "Math";
        constructor() {
            super();
            this.addInput("A", "tensor,number");
            this.addInput("B", "tensor,number");
            this.addOutput("out", "tensor,number");
            this.size = [140, 60];
        }
        onExecute() {
            const a = this.getInputData(0), b = this.getInputData(1);
            if (a != null && b != null) this.setOutputData(0, a + b);
        }
    }

    class MultiplyNode extends LG.LGraphNode {
        static title = "Multiply";
        static category = "Math";
        constructor() {
            super();
            this.addInput("A", "tensor,number");
            this.addInput("B", "tensor,number");
            this.addOutput("out", "tensor,number");
            this.size = [140, 60];
        }
        onExecute() {
            const a = this.getInputData(0), b = this.getInputData(1);
            if (a != null && b != null) this.setOutputData(0, a * b);
        }
    }

    class ScalarNode extends LG.LGraphNode {
        static title = "Scalar";
        static category = "Math";
        constructor() {
            super();
            this.addWidget("number", "value", 1.0, "value");
            this.addOutput("out", "number");
            this.size = [150, 50];
        }
        onExecute() { this.setOutputData(0, this.properties.value ?? 1.0); }
    }

    // ── FFT ─────────────────────────────────────────────────────────────────
    class FFTNode extends LG.LGraphNode {
        static title = "FFT";
        static category = "FFT";
        constructor() {
            super();
            this.addInput("in", "tensor");
            this.addOutput("k-space", "tensor");
            this.addWidget("combo", "norm", "ortho", "norm",
                { values: ["none", "ortho", "forward", "backward"] });
            this.size = [160, 70];
        }
        onExecute() {}
    }

    class IFFTNode extends LG.LGraphNode {
        static title = "IFFT";
        static category = "FFT";
        constructor() {
            super();
            this.addInput("k-space", "tensor");
            this.addOutput("out", "tensor");
            this.addWidget("combo", "norm", "ortho", "norm",
                { values: ["none", "ortho", "forward", "backward"] });
            this.size = [160, 70];
        }
        onExecute() {}
    }

    class NUFFTNode extends LG.LGraphNode {
        static title = "NUFFT";
        static category = "FFT";
        constructor() {
            super();
            this.addInput("image", "tensor");
            this.addInput("coords", "tensor");
            this.addOutput("k-space", "tensor");
            this.size = [160, 70];
        }
        onExecute() {}
    }

    // ── MRI ─────────────────────────────────────────────────────────────────
    class CoilSenseNode extends LG.LGraphNode {
        static title = "Coil Sense";
        static category = "MRI";
        constructor() {
            super();
            this.addInput("image", "tensor");
            this.addInput("smaps", "tensor");
            this.addOutput("coil_imgs", "tensor");
            this.size = [170, 70];
        }
        onExecute() {}
    }

    class SensitivityMapNode extends LG.LGraphNode {
        static title = "Sensitivity Maps";
        static category = "MRI";
        constructor() {
            super();
            this.addInput("k-space", "tensor");
            this.addOutput("smaps", "tensor");
            this.size = [170, 50];
        }
        onExecute() {}
    }

    class ToeplitzNode extends LG.LGraphNode {
        static title = "Toeplitz";
        static category = "MRI";
        constructor() {
            super();
            this.addInput("k-traj", "tensor");
            this.addInput("weights", "tensor");
            this.addOutput("kernel", "tensor");
            this.size = [170, 70];
        }
        onExecute() {}
    }

    // ── Viz ─────────────────────────────────────────────────────────────────
    class OrthoSliceNode extends LG.LGraphNode {
        static title = "Ortho Slicer";
        static category = "Viz";
        constructor() {
            super();
            this.addInput("volume", "tensor");
            this.addWidget("button", "Open Viewer", null, () => {
                window.open("../orthoslicer/", "_blank");
            });
            this.size = [175, 70];
        }
        onExecute() {}
    }

    // ── Register ────────────────────────────────────────────────────────────
    for (const cls of [
        LoadTensorNode, SaveTensorNode,
        AddNode, MultiplyNode, ScalarNode,
        FFTNode, IFFTNode, NUFFTNode,
        CoilSenseNode, SensitivityMapNode, ToeplitzNode,
        OrthoSliceNode,
    ]) {
        LG.registerNodeType(`${cls.category}/${cls.title}`, cls);
    }
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
function buildSidebar(canvas) {
    const container = document.getElementById("node-list");
    const groups = {};
    for (const [type, cls] of Object.entries(LG.registered_node_types)) {
        const [group] = type.split("/");
        (groups[group] ??= []).push({ type, title: cls.title ?? type.split("/")[1] });
    }
    for (const [group, nodes] of Object.entries(groups)) {
        const label = document.createElement("div");
        label.className = "node-group-label";
        label.textContent = group;
        container.appendChild(label);
        for (const { type, title } of nodes) {
            const entry = document.createElement("div");
            entry.className = "node-entry";
            entry.textContent = title;
            entry.title = type;
            // Drag-to-canvas: store type in dataTransfer
            entry.draggable = true;
            entry.addEventListener("dragstart", e => {
                e.dataTransfer.setData("nodetype", type);
            });
            // Double-click to add at center
            entry.addEventListener("dblclick", () => {
                const node = LG.createNode(type);
                const rect = canvas.canvas.getBoundingClientRect();
                node.pos = canvas.convertOffsetToCanvas([rect.width / 2, rect.height / 2]);
                canvas.graph.add(node);
            });
            container.appendChild(entry);
        }
    }
}

// ── Drop handler ─────────────────────────────────────────────────────────────
function setupDrop(lgCanvas) {
    const el = lgCanvas.canvas;
    el.addEventListener("dragover", e => e.preventDefault());
    el.addEventListener("drop", e => {
        e.preventDefault();
        const type = e.dataTransfer.getData("nodetype");
        if (!type) return;
        const rect = el.getBoundingClientRect();
        const pos = lgCanvas.convertOffsetToCanvas([e.clientX - rect.left, e.clientY - rect.top]);
        const node = LG.createNode(type);
        node.pos = pos;
        lgCanvas.graph.add(node);
    });
}

// ── Toolbar buttons ──────────────────────────────────────────────────────────
function setupToolbar(graph) {
    document.getElementById("btn-run").addEventListener("click", () => graph.runStep());
    document.getElementById("btn-clear").addEventListener("click", () => {
        if (confirm("Clear graph?")) graph.clear();
    });
    document.getElementById("btn-save").addEventListener("click", () => {
        const json = JSON.stringify(graph.serialize(), null, 2);
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
        a.download = "graph.json";
        a.click();
    });
    document.getElementById("btn-load").addEventListener("click", () => {
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept = ".json";
        inp.onchange = async () => {
            const text = await inp.files[0].text();
            graph.configure(JSON.parse(text));
        };
        inp.click();
    });
}

// ── Init ─────────────────────────────────────────────────────────────────────
function init() {
    registerNodes();

    const graph  = new LG.LGraph();
    const canvas = document.getElementById("graph-canvas");

    // Fit canvas to its CSS-laid-out container
    function resize() {
        const main = canvas.parentElement;
        canvas.width  = main.clientWidth;
        canvas.height = main.clientHeight;
    }
    resize();
    window.addEventListener("resize", () => { resize(); lgCanvas.setDirty(true, true); });

    const lgCanvas = new LG.LGraphCanvas(canvas, graph);
    lgCanvas.background_image = null;

    // Dark canvas theme
    lgCanvas.render_shadows       = false;
    lgCanvas.render_canvas_border = false;
    lgCanvas.node_title_color     = "#9ab8d0";
    lgCanvas.default_connection_color.input_on  = "#5a9abc";
    lgCanvas.default_connection_color.output_on = "#5a9abc";

    buildSidebar(lgCanvas);
    setupDrop(lgCanvas);
    setupToolbar(graph);

    graph.start();

    // Add a welcome comment node
    const note = LG.createNode("Note");
    if (note) {
        note.pos = [80, 80];
        note.properties.text = "Double-click a node in the sidebar\nor drag it onto the canvas.";
        graph.add(note);
    }
}

init();
