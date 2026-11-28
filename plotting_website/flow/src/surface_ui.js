// Wiring of the "Surface" section of the flow viewer: compute the moving blood-pool surface on the server from the data
// (flow_bloodpool) and the wall shell around it (flow_wall_shell, from the magnitude), keep both in the bank, load / show / cut
// them. Opt-in: nothing happens until the user asks (or ?mesh= / ?wall=).
//   flow/index.html?id=<flow value>&mesh=<mesh_seq id>[&wall=<mesh_seq id>]   show stored surface sequences directly
export function initSurfaceUI({ $, client, id, name, surface, wall, status, invalidate, params, ext, tau }) {
    const poolName = `${name} blood pool`, wallName = `${name} wall shell`;
    let poolId = null;
    const msg = (s) => { $('sf-msg').textContent = s; };
    const layers = [surface, wall];
    const clip = () => {
        const axis = $('sf-cut').value;
        if (axis === '') for (const l of layers) l.setClip(null);
        else {
            const a = Number(axis);
            $('sf-cutpos').max = ext[a];
            for (const l of layers) l.setClip(a, Number($('sf-cutpos').value), $('sf-flip').checked);
        }
        invalidate();
    };
    const show = () => {
        surface.group.visible = $('sf-on').checked && surface.loaded;
        wall.group.visible = $('sf-wall-on').checked && wall.loaded;
        for (const l of layers) { l.key = ''; l.setTime(tau()); }
        invalidate();
    };

    async function load(meshId, info) {
        status('loading the surface sequence…');
        const s = surface.set(await client.get(meshId));
        poolId = meshId;
        const verts = s.frames.map((f) => f.pos.length / 3), linked = s.links.reduce((a, b) => a + b, 0);
        $('sf-on').disabled = false;
        $('sf-on').checked = true;
        $('sf-wall-go').disabled = false;
        $('sf-info').textContent = `(${s.n} frames, ${Math.round(Math.max(...verts) / 1e3)} k vertices)`;
        msg(`${linked} of ${s.links.length} frame transitions keep the same vertices (smooth motion); the others start a new surface (a valve opening or closing splits or joins the pool)${info ? ` — ${info}` : ''}`);
        show();
        status('surface loaded');
    }

    async function loadWall(meshId, info) {
        status('loading the wall shell…');
        const s = wall.set(await client.get(meshId));
        $('sf-wall-on').disabled = false;
        $('sf-wall-on').checked = true;
        $('sf-wall-info').textContent = `(${s.n} frames)`;
        if (info) $('sf-wall-msg').textContent = info;
        show();
        status('wall shell loaded');
    }

    async function compute() {
        const opts = {};
        const mt = Number($('sf-mag').value), st = Number($('sf-speed').value);
        if ($('sf-mag').value !== '' && mt > 0) opts.mag_thr = mt;
        if ($('sf-speed').value !== '' && st > 0) opts.speed_thr = st;
        $('sf-go').disabled = true;
        status('computing the blood-pool surfaces on the server (minutes at 1 mm)…');
        try {
            const r = await client.execute('flow_bloodpool', opts, [id], poolName);
            const m = JSON.parse(r.message || '{}');
            const vol = (m.volume_ml ?? []).map((x) => x.toFixed(0));
            await load(r.outputs[0], `blood volume ${vol.join(' / ')} ml per frame, magnitude threshold ${Number(m.mag_thr).toPrecision(3)}, speed threshold ${Number(m.speed_thr).toPrecision(3)} m/s${(m.warnings ?? []).length ? `; ${m.warnings.join('; ')}` : ''}; stored in the bank as "${poolName}" (reuse: &mesh=${r.outputs[0]})`);
        } catch (e) { msg(`failed: ${e.message}`); status(`surface failed: ${e.message}`); console.error(e); }
        $('sf-go').disabled = false;
    }

    async function computeWall() {
        if (!poolId) return;
        $('sf-wall-go').disabled = true;
        status('estimating the wall shell on the server…');
        try {
            const r = await client.execute('flow_wall_shell', {}, [id, poolId], wallName);
            const m = JSON.parse(r.message || '{}');
            const med = (m.frames ?? []).map((f) => f.median_thickness_mm).filter((x) => x != null).sort((a, b) => a - b);
            const share = (m.frames ?? []).map((f) => f.confident_vertices / f.vertices);
            await loadWall(r.outputs[0], `median wall thickness where the data has an edge: ${med.length ? med[Math.floor(med.length / 2)].toFixed(1) : '?'} mm; ${(100 * share.reduce((a, b) => a + b, 0) / Math.max(1, share.length)).toFixed(0)} % of the blood-pool vertices have an edge; stored as "${wallName}" (reuse: &wall=${r.outputs[0]})`);
        } catch (e) { $('sf-wall-msg').textContent = `failed: ${e.message}`; status(`wall shell failed: ${e.message}`); console.error(e); }
        $('sf-wall-go').disabled = false;
    }

    $('sf-go').onclick = compute;
    $('sf-wall-go').onclick = computeWall;
    $('sf-on').onchange = show;
    $('sf-wall-on').onchange = show;
    $('sf-op').oninput = () => { surface.setOpacity(Number($('sf-op').value)); invalidate(); };
    $('sf-col').oninput = () => { surface.setColor($('sf-col').value); invalidate(); };
    $('sf-wall-op').oninput = () => { wall.setOpacity(Number($('sf-wall-op').value)); invalidate(); };
    $('sf-wall-col').oninput = () => { wall.setColor($('sf-wall-col').value); invalidate(); };
    $('sf-wall-sup').oninput = () => { wall.setMinSupport(Number($('sf-wall-sup').value)); invalidate(); };
    wall.setColor($('sf-wall-col').value);
    wall.setOpacity(Number($('sf-wall-op').value));
    wall.setMinSupport(Number($('sf-wall-sup').value));
    for (const k of ['sf-cut', 'sf-flip']) $(k).onchange = clip;
    $('sf-cutpos').oninput = clip;

    // stored sequences of this flow value, or the ones named in the URL
    (async () => {
        const wantMesh = params.get('mesh'), wantWall = params.get('wall');
        if (wantMesh) await load(wantMesh).catch((e) => msg(`could not load ${wantMesh}: ${e.message}`));
        if (wantWall) await loadWall(wantWall).catch((e) => { $('sf-wall-msg').textContent = `could not load ${wantWall}: ${e.message}`; });
        if (wantMesh) return;
        const list = await client.list();
        const pool = list.filter((e) => e.name === poolName).pop(), shell = list.filter((e) => e.name === wallName).pop();
        if (pool) { $('sf-stored').hidden = false; $('sf-stored').onclick = async () => { await load(pool.id).catch((e) => msg(`failed: ${e.message}`)); if (shell) await loadWall(shell.id).catch(() => {}); }; }
    })().catch((e) => console.warn('surface list', e));
}
