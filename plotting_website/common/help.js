// Hover help for settings, shared by all pages.
//
//   import { installHelp } from '../../common/help.js';
//   import { HELP } from './help.js';          // per page: { 'control-id': 'explanation', ... }
//   installHelp(HELP);
//
// Hover the label of a control (or the control, or a button) for about a second and a box with the explanation opens.
// The key is the id of the control; the text may use \n for paragraphs. Elements outside a label can carry the text
// directly: <h3 data-help="...">. Controls with an entry get a dotted underline. `?helpcheck=1` lists in the console
// every control that has no entry yet.
const CSS = `
.has-help { text-decoration: underline dotted rgba(150,170,190,0.55); text-underline-offset: 3px; }
#help-tip { position: fixed; z-index: 10000; max-width: 340px; padding: 8px 11px; border-radius: 6px; background: #1d2228; color: #dfe5ea;
  border: 1px solid #44505c; box-shadow: 0 6px 22px rgba(0,0,0,0.5); font: 12.5px/1.45 system-ui, sans-serif; white-space: pre-line; pointer-events: none;
  opacity: 0; transition: opacity 0.12s; }
#help-tip.show { opacity: 1; }
#help-tip b { color: #fff; }
`;

const CONTROLS = 'label, button, select, input, [data-help], h3, .hdr-label';

function keyOf(el) {
    if (el.dataset?.help) return null;                       // direct text
    if (el.tagName === 'LABEL') {
        const c = el.control ?? el.querySelector('[id]');
        return c?.id ?? null;
    }
    return el.id || null;
}

export function installHelp(help, { delay = 900 } = {}) {
    if (document.getElementById('help-tip')) return;
    const st = document.createElement('style'); st.textContent = CSS; document.head.append(st);
    const tip = document.createElement('div'); tip.id = 'help-tip'; document.body.append(tip);
    let timer = null, cur = null;

    const textOf = (el) => el.dataset?.help ?? help[keyOf(el)];
    // the element that carries the hover: a label for its control, else the element itself
    const anchorOf = (t) => {
        const el = t.closest?.(CONTROLS);
        if (!el) return null;
        if (el.tagName !== 'LABEL') { const l = el.closest('label'); if (l && textOf(l)) return l; }
        return textOf(el) ? el : null;
    };
    const hide = () => { clearTimeout(timer); timer = null; cur = null; tip.classList.remove('show'); };
    function show(el) {
        tip.textContent = textOf(el);
        const r = el.getBoundingClientRect(), W = innerWidth, H = innerHeight;
        tip.style.left = '0px'; tip.style.top = '0px';
        const w = tip.offsetWidth, h = tip.offsetHeight;
        let x = r.right + 10, y = r.top - 4;
        if (x + w > W - 8) x = Math.max(8, r.left - w - 10);                  // no room to the right: left of it
        if (x < 8 || x + w > W - 8) { x = Math.max(8, Math.min(W - w - 8, r.left)); y = r.bottom + 8; }   // narrow: below
        if (y + h > H - 8) y = Math.max(8, H - h - 8);
        tip.style.left = `${x}px`; tip.style.top = `${y}px`;
        tip.classList.add('show');
    }
    document.addEventListener('mouseover', (ev) => {
        const a = anchorOf(ev.target);
        if (a === cur) return;
        hide();
        if (!a) return;
        cur = a;
        timer = setTimeout(() => { try { show(a); } catch (e) { console.log("help err " + e); } }, delay);
    });
    document.addEventListener('mouseleave', hide);
    for (const e of ['pointerdown', 'wheel', 'keydown', 'scroll']) document.addEventListener(e, hide, { capture: true, passive: true });

    // dotted underline on the label text of every control that has help (also on controls added later)
    const mark = () => {
        for (const el of document.querySelectorAll('label, .hdr-label, h3, button')) {
            if (el.classList.contains('has-help') || !textOf(el)) continue;
            if (el.tagName === 'BUTTON' || el.tagName === 'H3') continue;     // buttons/headers: tooltip only
            el.classList.add('has-help');
        }
    };
    mark();
    new MutationObserver(mark).observe(document.body, { childList: true, subtree: true });

    if (new URLSearchParams(location.search).get('helpcheck') === '1') {
        const missing = [...document.querySelectorAll('input[id], select[id], button[id]')].filter((e) => !(e.id in help) && !e.closest('[data-help]')).map((e) => e.id);
        const unused = Object.keys(help).filter((k) => !document.getElementById(k));
        console.log(`help coverage: ${missing.length} controls without text: ${missing.join(', ')}\nentries without a control: ${unused.join(', ')}`);
    }
}
