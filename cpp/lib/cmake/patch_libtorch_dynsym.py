#!/usr/bin/env python3
"""
Patch libtorch_cuda.so: rename ntsr4__T0 → ntsr3std in .dynsym/.dynstr.

Strategy: rebuild .dynstr with replacements (shorter strings), pad to original
size, update .dynsym st_name AND .dynamic string-index entries using the same
old→new offset mapping. Patching both prevents the corrupted DT_NEEDED issue.
"""

import sys
import struct
from pathlib import Path

OLD = b'ntsr4__T0'
NEW = b'ntsr3std'

# .dynamic tags whose d_val is an index into .dynstr
DYNSTR_TAGS = {
    1,   # DT_NEEDED
    14,  # DT_SONAME
    15,  # DT_RPATH
    29,  # DT_RUNPATH
    0x6000000c,  # DT_SUNW_AUXILIARY
    0x6000000d,  # DT_SUNW_FILTER
    0x6ffffefa,  # DT_CONFIG
    0x6ffffefb,  # DT_DEPAUDIT
    0x6ffffefc,  # DT_AUDIT
}


def patch(path: str) -> int:
    data = bytearray(Path(path).read_bytes())

    assert data[:4] == b'\x7fELF', "Not an ELF file"
    assert data[4] == 2, "Only ELF64 supported"

    def u16(off): return struct.unpack_from('<H', data, off)[0]
    def u32(off): return struct.unpack_from('<I', data, off)[0]
    def u64(off): return struct.unpack_from('<Q', data, off)[0]
    def put32(off, v): struct.pack_into('<I', data, off, v)
    def put64(off, v): struct.pack_into('<Q', data, off, v)

    e_shoff     = u64(0x28)
    e_shentsize = u16(0x3a)
    e_shnum     = u16(0x3c)
    e_shstrndx  = u16(0x3e)

    def shdr(i):
        base = e_shoff + i * e_shentsize
        return {
            'sh_name':    u32(base),
            'sh_type':    u32(base + 4),
            'sh_offset':  u64(base + 24),
            'sh_size':    u64(base + 32),
            'sh_entsize': u64(base + 56),
            'hdr_off':    base,
        }

    sections = [shdr(i) for i in range(e_shnum)]

    shstr = sections[e_shstrndx]
    shstr_bytes = bytes(data[shstr['sh_offset']:shstr['sh_offset'] + shstr['sh_size']])

    def name_of(s):
        off = s['sh_name']
        return shstr_bytes[off:shstr_bytes.index(0, off)].decode()

    dynsym  = next((s for s in sections if name_of(s) == '.dynsym'), None)
    dynstr  = next((s for s in sections if name_of(s) == '.dynstr'), None)
    dynamic = next((s for s in sections if name_of(s) == '.dynamic'), None)
    assert dynsym and dynstr, ".dynsym or .dynstr not found"

    old_dynstr = bytes(data[dynstr['sh_offset']:dynstr['sh_offset'] + dynstr['sh_size']])

    # Build new .dynstr and old→new offset map
    new_dynstr  = bytearray()
    old_to_new  = {}

    i = 0
    while i < len(old_dynstr):
        old_to_new[i] = len(new_dynstr)
        try:
            end = old_dynstr.index(0, i)
        except ValueError:
            end = len(old_dynstr)
        s = old_dynstr[i:end].replace(OLD, NEW)
        new_dynstr += s + b'\x00'
        i = end + 1

    assert len(new_dynstr) <= len(old_dynstr), (
        f"New .dynstr ({len(new_dynstr)}) longer than old ({len(old_dynstr)})"
    )
    new_dynstr += b'\x00' * (len(old_dynstr) - len(new_dynstr))

    data[dynstr['sh_offset']:dynstr['sh_offset'] + dynstr['sh_size']] = new_dynstr

    # Update .dynsym st_name
    sym_size = dynsym['sh_entsize']
    n_syms   = dynsym['sh_size'] // sym_size
    updated_syms = 0
    for j in range(n_syms):
        sym_off = dynsym['sh_offset'] + j * sym_size
        st_name = u32(sym_off)
        new_name = old_to_new.get(st_name, st_name)
        if new_name != st_name:
            put32(sym_off, new_name)
            updated_syms += 1

    # Update .dynamic string-index entries (DT_NEEDED, DT_SONAME, DT_RPATH …)
    updated_dyn = 0
    if dynamic:
        dyn_ent = 16  # sizeof(Elf64_Dyn)
        n_dyn   = dynamic['sh_size'] // dyn_ent
        for j in range(n_dyn):
            dyn_off = dynamic['sh_offset'] + j * dyn_ent
            d_tag = u64(dyn_off)
            if d_tag == 0:   # DT_NULL — end of table
                break
            if d_tag in DYNSTR_TAGS:
                d_val = u64(dyn_off + 8)
                new_val = old_to_new.get(d_val, d_val)
                if new_val != d_val:
                    put64(dyn_off + 8, new_val)
                    updated_dyn += 1

    # Update .gnu.version_r (Elf64_Verneed + Elf64_Vernaux) — vn_file, vna_name
    updated_ver = 0
    gnu_ver_r = next((s for s in sections if name_of(s) == '.gnu.version_r'), None)
    if gnu_ver_r:
        off = gnu_ver_r['sh_offset']
        while off < gnu_ver_r['sh_offset'] + gnu_ver_r['sh_size']:
            vn_file = u32(off + 4)   # index into .dynstr for file name
            vn_cnt  = u16(off + 2)
            vn_aux  = u32(off + 8)   # offset from off to first Vernaux
            vn_next = u32(off + 12)  # offset from off to next Verneed (0=last)

            new_file = old_to_new.get(vn_file, vn_file)
            if new_file != vn_file:
                put32(off + 4, new_file)
                updated_ver += 1

            # Iterate Vernaux entries
            aux_off = off + vn_aux
            for _ in range(vn_cnt):
                vna_name = u32(aux_off + 8)   # index into .dynstr for version name
                vna_next = u32(aux_off + 12)
                new_name = old_to_new.get(vna_name, vna_name)
                if new_name != vna_name:
                    put32(aux_off + 8, new_name)
                    updated_ver += 1
                if vna_next == 0:
                    break
                aux_off += vna_next

            if vn_next == 0:
                break
            off += vn_next

    # Update .gnu.version_d (Elf64_Verdef + Elf64_Verdaux) — vda_name
    gnu_ver_d = next((s for s in sections if name_of(s) == '.gnu.version_d'), None)
    if gnu_ver_d:
        off = gnu_ver_d['sh_offset']
        while off < gnu_ver_d['sh_offset'] + gnu_ver_d['sh_size']:
            vd_aux  = u32(off + 12)
            vd_next = u32(off + 16)
            aux_off = off + vd_aux
            while True:
                vda_name = u32(aux_off)
                vda_next = u32(aux_off + 4)
                new_name = old_to_new.get(vda_name, vda_name)
                if new_name != vda_name:
                    put32(aux_off, new_name)
                    updated_ver += 1
                if vda_next == 0:
                    break
                aux_off += vda_next
            if vd_next == 0:
                break
            off += vd_next

    Path(path).write_bytes(data)

    remaining = data[dynstr['sh_offset']:dynstr['sh_offset'] + dynstr['sh_size']].count(OLD)
    print(f"Updated {updated_syms} .dynsym + {updated_dyn} .dynamic "
          f"+ {updated_ver} version entries; {remaining} OLD left in .dynstr")
    return updated_syms


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <libtorch_cuda.so>")
        sys.exit(1)
    n = patch(sys.argv[1])
    sys.exit(0 if n > 0 else 1)
