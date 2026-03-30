#!/usr/bin/env python3

# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
import json
import re

vl_int_re = re.compile(r"\d+'s*h([\da-fA-F]+)")

def parse_vl_int(text):
    m = vl_int_re.search(text)
    if m:
        return int(m.group(1), 16)
    return int(text, 0)

def build_addr_lookups(json_doc):
    """Build addr→module and addr→type lookups."""
    addr_to_mod = {mod['addr']: mod for mod in json_doc.get('modulesp', [])}
    addr_to_type = {}
    for item in json_doc.get('miscsp', []):
        if item.get('type') == 'TYPETABLE':
            for t in item.get('typesp', []):
                addr_to_type[t['addr']] = t
    return addr_to_mod, addr_to_type

def parse_dtype_width(addr_to_type, dtype_addr):
    """Return bit width of a type identified by its addr."""
    t = addr_to_type.get(dtype_addr)
    if not t:
        return 1
    ttype = t.get('type', '')
    if ttype == 'BASICDTYPE':
        rng = t.get('range')
        if rng:
            left, right = (int(x) for x in rng.split(':'))
            return abs(left - right) + 1
        return 1
    elif ttype in ('PACKARRAYDTYPE', 'UNPACKARRAYDTYPE'):
        sub_addr = t.get('refDTypep')
        base_width = parse_dtype_width(addr_to_type, sub_addr) if sub_addr else 1
        rangep = t.get('rangep', [])
        if rangep:
            rng = rangep[0]
            left_val  = parse_vl_int(rng.get('leftp',  [{'name': '0'}])[0].get('name', '0'))
            right_val = parse_vl_int(rng.get('rightp', [{'name': '0'}])[0].get('name', '0'))
            return base_width * (abs(left_val - right_val) + 1)
        return base_width
    elif ttype == 'STRUCTDTYPE':
        return sum(parse_dtype_width(addr_to_type, m.get('refDTypep'))
                   for m in t.get('membersp', []))
    elif ttype == 'UNIONDTYPE':
        return max((parse_dtype_width(addr_to_type, m.get('refDTypep'))
                    for m in t.get('membersp', [])), default=1)
    else:
        sub_addr = t.get('refDTypep')
        if sub_addr:
            return parse_dtype_width(addr_to_type, sub_addr)
        return 1

def parse_sel_field(addr_to_type, dtype_addr, offset, width):
    """Map (offset, width) within a type to a field suffix string."""
    t = addr_to_type.get(dtype_addr)
    if not t:
        return ''
    ttype = t.get('type', '')
    if ttype == 'STRUCTDTYPE':
        bit_offset = 0
        members = list(t.get('membersp', []))
        members.reverse()
        for member in members:
            sub_addr = member.get('refDTypep')
            member_name = member.get('name', '')
            member_width = parse_dtype_width(addr_to_type, sub_addr)
            if bit_offset <= offset < bit_offset + member_width:
                if width != member_width and sub_addr:
                    sub_field = parse_sel_field(addr_to_type, sub_addr, offset - bit_offset, width)
                    return f".{member_name}{sub_field}"
                return f".{member_name}"
            bit_offset += member_width
        raise ValueError(f"invalid probe entry: offset={offset}, width={width} not in struct")
    elif ttype in ('PACKARRAYDTYPE', 'UNPACKARRAYDTYPE'):
        sub_addr = t.get('refDTypep')
        base_width = parse_dtype_width(addr_to_type, sub_addr)
        if width > base_width:
            return ''
        array_index = offset // base_width
        sub_offset  = offset % base_width
        sub_field = parse_sel_field(addr_to_type, sub_addr, sub_offset, width)
        return f"_{array_index}{sub_field}"
    elif ttype == 'BASICDTYPE':
        if width == 1:
            return f"[{offset}]"
        return f"[{offset + width - 1}:{offset}]"
    else:
        sub_addr = t.get('refDTypep')
        if sub_addr:
            return parse_sel_field(addr_to_type, sub_addr, offset, width)
        raise ValueError(f"invalid probe entry: type={ttype}")

def parse_var_name(node):
    """Extract signal name from a VARREF / VARXREF / ARRAYSEL node."""
    ntype = node.get('type', '')
    if ntype == 'VARREF':
        return node.get('name', '')
    elif ntype == 'VARXREF':
        name   = node.get('name', '')
        dotted = node.get('dotted', '')
        return f"{dotted}.{name}" if dotted else name
    elif ntype == 'ARRAYSEL':
        return parse_arraysel_name(node)
    else:
        raise ValueError(f"invalid probe entry: type={ntype}")

def parse_arraysel_name(node):
    """Flatten an ARRAYSEL chain into a name with _N suffixes."""
    if node.get('type') == 'ARRAYSEL':
        fromp = node.get('fromp', [{}])
        child = fromp[0] if fromp else {}
        name = (parse_arraysel_name(child)
                if child.get('type') == 'ARRAYSEL'
                else parse_var_name(child))
        indexp = node.get('bitp', node.get('rhsp', [{'name': '0'}]))
        offset = parse_vl_int((indexp[0] if indexp else {}).get('name', '0'))
        return f"{name}_{offset}"
    return parse_var_name(node)

def parse_sel_name(addr_to_type, node):
    """Get field-selected signal name from a SEL node."""
    fromp = node.get('fromp', [{}])
    first_child = fromp[0] if fromp else {}
    name = parse_var_name(first_child) if first_child else ''
    dtype_addr = first_child.get('dtypep', '')
    lsbp   = node.get('lsbp', [{'name': '0'}])
    offset = parse_vl_int((lsbp[0] if lsbp else {}).get('name', '0'))
    width  = node.get('widthConst', parse_dtype_width(addr_to_type, node.get('dtypep', '')))
    return name + parse_sel_field(addr_to_type, dtype_addr, offset, width)

def parse_vl_port(addr_to_type, node, signals):
    """Recursively extract (signal_name, width) pairs from a port expression."""
    if node is None:
        return 0
    ntype = node.get('type', '')
    total_width = 0
    if ntype == 'CONCAT':
        for child_list in (node.get('lhsp', []), node.get('rhsp', [])):
            if child_list:
                total_width += parse_vl_port(addr_to_type, child_list[0], signals)
    elif ntype in ('VARREF', 'VARXREF'):
        name = parse_var_name(node)
        signal_width = parse_dtype_width(addr_to_type, node.get('dtypep', ''))
        signals.append([name, signal_width])
        total_width += signal_width
    elif ntype == 'SEL':
        name = parse_sel_name(addr_to_type, node)
        signal_width = parse_dtype_width(addr_to_type, node.get('dtypep', ''))
        signals.append([name, signal_width])
        total_width += signal_width
    elif ntype == 'ARRAYSEL':
        name = parse_arraysel_name(node)
        signal_width = parse_dtype_width(addr_to_type, node.get('dtypep', ''))
        signals.append([name, signal_width])
        total_width += signal_width
    else:
        raise ValueError(f"invalid probe entry: type={ntype}")
    signal_names = [s[0] for s in signals]
    duplicates = {n for n in signal_names if signal_names.count(n) > 1}
    if duplicates:
        raise ValueError("duplicate signal names: " + ", ".join(duplicates))
    return total_width

def get_module_param(mod, param_name):
    """Return integer value of a parameter from a module's stmtsp."""
    for stmt in mod.get('stmtsp', []):
        if stmt.get('type') == 'VAR' and stmt.get('name') == param_name:
            valuep = stmt.get('valuep', [])
            if valuep:
                return parse_vl_int(valuep[0].get('name', '0'))
    return 0

def iter_stmts(stmtsp):
    """Yield all statements, recursing into generate blocks."""
    for stmt in stmtsp:
        yield stmt
        if stmt.get('type') in ('GENBLOCK', 'GENFOR', 'GENIF', 'BEGINBLOCK'):
            yield from iter_stmts(stmt.get('stmtsp', []))

def find_scope_tap_instances(mod, addr_to_mod, scope_tap_addrs, current_path, results, visited):
    """DFS through module hierarchy to locate VX_scope_tap instances."""
    if mod['addr'] in visited:
        return
    visited = visited | {mod['addr']}
    for stmt in iter_stmts(mod.get('stmtsp', [])):
        if stmt.get('type') != 'CELL':
            continue
        inst_name  = stmt.get('name', '')
        modp_addr  = stmt.get('modp')
        child_mod  = addr_to_mod.get(modp_addr)
        if not child_mod:
            continue
        full_path = f"{current_path}.{inst_name}" if current_path else inst_name
        if modp_addr in scope_tap_addrs:
            results.append((stmt, child_mod, full_path))
        else:
            find_scope_tap_instances(child_mod, addr_to_mod, scope_tap_addrs,
                                     full_path, results, visited)

def find_top_module(json_doc, addr_to_mod):
    """Identify the top module as the one never used as a CELL's modp."""
    all_modp = {stmt.get('modp')
                for mod in json_doc.get('modulesp', [])
                for stmt in iter_stmts(mod.get('stmtsp', []))
                if stmt.get('type') == 'CELL'}
    for mod in json_doc.get('modulesp', []):
        if mod['addr'] not in all_modp:
            return mod
    mods = json_doc.get('modulesp', [])
    return mods[0] if mods else None

def parse_json(filename, max_taps):
    with open(filename) as f:
        json_doc = json.load(f)

    addr_to_mod, addr_to_type = build_addr_lookups(json_doc)

    scope_tap_addrs = {mod['addr']
                       for mod in json_doc.get('modulesp', [])
                       if mod.get('origName') == 'VX_scope_tap'}
    if not scope_tap_addrs:
        return {"version": "0.1.0", "taps": []}

    top_mod = find_top_module(json_doc, addr_to_mod)
    if not top_mod:
        return {"version": "0.1.0", "taps": []}

    instances = []
    find_scope_tap_instances(top_mod, addr_to_mod, scope_tap_addrs, '', instances, set())

    taps = []
    for cell, scope_mod, hier_path in instances:
        if max_taps != -1 and len(taps) >= max_taps:
            break
        scope_id  = get_module_param(scope_mod, 'SCOPE_ID')
        xtriggerw = get_module_param(scope_mod, 'XTRIGGERW')
        htriggerw = get_module_param(scope_mod, 'HTRIGGERW')
        probew    = get_module_param(scope_mod, 'PROBEW')

        pin_by_name = {pin.get('name'): pin for pin in cell.get('pinsp', [])}

        xtriggers, htriggers, probes = [], [], []

        if xtriggerw > 0:
            pin = pin_by_name.get('xtriggers', {})
            exprp = pin.get('exprp', [])
            if exprp:
                w = parse_vl_port(addr_to_type, exprp[0], xtriggers)
                if w != xtriggerw:
                    raise ValueError(f"invalid xtriggers width: actual={w}, expected={xtriggerw}")

        if htriggerw > 0:
            pin = pin_by_name.get('htriggers', {})
            exprp = pin.get('exprp', [])
            if exprp:
                w = parse_vl_port(addr_to_type, exprp[0], htriggers)
                if w != htriggerw:
                    raise ValueError(f"invalid htriggers width: actual={w}, expected={htriggerw}")

        pin = pin_by_name.get('probes', {})
        exprp = pin.get('exprp', [])
        if exprp:
            w = parse_vl_port(addr_to_type, exprp[0], probes)
            if w != probew:
                raise ValueError(f"invalid probes width: actual={w}, expected={probew}")

        signals = probes + xtriggers + htriggers
        path = hier_path.rsplit('.', 1)[0] if '.' in hier_path else hier_path

        taps.append({"id":     scope_id,
                     "width":  xtriggerw + htriggerw + probew,
                     "signals": signals,
                     "path":   path})

    return {"version": "0.1.0", "taps": taps}

def main():
    parser = argparse.ArgumentParser(description='Scope headers generator.')
    parser.add_argument('-o', nargs='?', default='scope.json', metavar='o',
                        help='Output JSON manifest')
    parser.add_argument('-n', nargs='?', default=-1, metavar='n', type=int,
                        help='Maximum number of taps to read')
    parser.add_argument('json', help='Design JSON descriptor file')
    args = parser.parse_args()
    scope_taps = parse_json(args.json, args.n)
    with open(args.o, "w") as f:
        json.dump(scope_taps, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
