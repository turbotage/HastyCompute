// Server access for the node editor (hasty server, proto hasty.v2). Values flowing between nodes are value ids
// (32-character hex strings); comprep configs and numbers flow as plain objects.
import { HastyClient, firstTensor } from '../../common/hasty_client.js';

export const client = new HastyClient();

/** editor dtype names (widgets, NIfTI) → value-format dtype names, and back */
export const DTYPE = {
    b8: 'bool', u8: 'uint8', i8: 'int8', u16: 'uint16', i16: 'int16', i32: 'int32', i64: 'int64',
    f16: 'float16', f32: 'float32', f64: 'float64', c32: 'complex64', c64: 'complex128',
};
export const SHORT = Object.fromEntries(Object.entries(DTYPE).map(([k, v]) => [v, k]));

export const isId = (v) => typeof v === 'string' && /^[0-9a-f]{32}$/.test(v);

/** { name, dtype (editor name), shape, device } of the first tensor in a value */
export async function fetchMeta(id) {
    const e = await client.info(id);
    const t = firstTensor(e.summary);
    return { name: e.name, dtype: t ? (SHORT[t[1].dtype] ?? t[1].dtype) : (e.summary?.type ?? typeof e.summary), shape: t ? t[1].shape : [], device: 'cpu' };
}

/** run a server command; returns { outputs: [id], message } */
export function run(command, inputs = [], options = {}, name = '') {
    return client.execute(command, options, inputs, name || command);
}
