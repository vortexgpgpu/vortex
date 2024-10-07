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
import xml.etree.ElementTree as ET
import re
import json

vl_int_re = re.compile(r"\d+'s*h([\da-fA-F]+)")

def parse_vl_int(text):
    str_hex = re.sub(vl_int_re, r'\1', text)
    return int(str_hex, 16)

def source_loc(xml_doc, xml_loc):
    loc = xml_loc.split(",")
    file_id = loc[0]
    start_line = loc[1]
    start_col = loc[2]
    end_line = loc[3]
    end_col = loc[4]
    file = xml_doc.find(".//file/[@id='" + file_id + "']").get("filename")
    return f"{file} ({start_line}:{start_col}-{end_line}:{end_col})"

def parse_dtype_width(xml_doc, dtype_id):
    xml_type = xml_doc.find(".//typetable/*[@id='" + dtype_id + "']")
    if xml_type.tag in ["packarraydtype", "unpackarraydtype"]:
        sub_dtype_id = xml_type.get("sub_dtype_id")
        base_width = parse_dtype_width(xml_doc, sub_dtype_id)
        const_iter = xml_type.iter("const")
        first_const = next(const_iter)
        second_const = next(const_iter)
        left  = parse_vl_int(first_const.get("name"))
        right = parse_vl_int(second_const.get("name"))
        return base_width * (left - right + 1)
    elif xml_type.tag == "structdtype":
        width = 0
        for member in xml_type.iter("memberdtype"):
            sub_dtype_id = member.get("sub_dtype_id")
            width = width + parse_dtype_width(xml_doc, sub_dtype_id)
        return width
    elif xml_type.tag == "uniondtype":
        width = 0
        for member in xml_type.iter("memberdtype"):
            sub_dtype_id = member.get("sub_dtype_id")
            width = max(width, parse_dtype_width(xml_doc, sub_dtype_id))
        return width
    else:
        sub_dtype_id = xml_type.get("sub_dtype_id")
        if sub_dtype_id != None:
            return parse_dtype_width(xml_doc, sub_dtype_id)
        left = xml_type.get("left")
        right = xml_type.get("right")
        if left != None and right != None:
            return int(left) - int(right) + 1
        return 1

def parse_var_name(xml_doc, xml_node):
    if xml_node.tag == "varref":
        return xml_node.get("name")
    elif xml_node.tag == "varxref":
        name = xml_node.get("name")
        dotted = xml_node.get("dotted")
        return f"{dotted}.{name}"
    elif xml_node.tag == "arraysel":
        return parse_arraysel_name(xml_doc, xml_node)
    else:
        raise ET.ParseError("invalid probe entry: tag=" + xml_node.tag + ", " + source_loc(xml_doc, xml_node.get("loc")))
    return name

def parse_sel_field(xml_doc, dtype_id, offset, width):
    xml_type = xml_doc.find(".//typetable/*[@id='" + dtype_id + "']")
    name = xml_type.get("name")
    if xml_type.tag == "structdtype":
        bit_offset = 0
        members = list(xml_type.findall("memberdtype"))
        members.reverse()
        for member in members:
            sub_dtype_id = member.get("sub_dtype_id")
            member_name = member.get("name")
            member_width = parse_dtype_width(xml_doc, sub_dtype_id)
            if bit_offset <= offset < bit_offset + member_width:
                if width != member_width and sub_dtype_id:
                    sub_field = parse_sel_field(xml_doc, sub_dtype_id, offset - bit_offset, width)
                    return f".{member_name}{sub_field}"
                else:
                    return f".{member_name}"
            bit_offset += member_width
        raise ET.ParseError("invalid probe entry: " + source_loc(xml_doc, xml_type.get("loc")))
    elif xml_type.tag in ["packarraydtype", "unpackarraydtype"]:
        sub_dtype_id = xml_type.get("sub_dtype_id")
        base_width = parse_dtype_width(xml_doc, sub_dtype_id)
        if width > base_width:
            return ""
        array_index = offset // base_width
        sub_offset = offset % base_width
        array_sel_name = f"_{array_index}" # array indexing is not supported in VCD
        sub_field = parse_sel_field(xml_doc, sub_dtype_id, sub_offset, width)
        return f"{array_sel_name}{sub_field}"
    elif xml_type.tag == "basicdtype":
        if width == 1:
            return F"[{offset}]"
        end = width - 1 + offset
        return F"[{end}:{offset}]"
    else:
        raise ET.ParseError("invalid probe entry: tag=" + xml_type.tag + ", " + source_loc(xml_doc, xml_type.get("loc")))
    return None

def parse_sel_name(xml_doc, xml_node):
    first_child = xml_node.find("*")
    name = parse_var_name(xml_doc, first_child)
    dtype_id = first_child.get("dtype_id")
    const_iter = xml_node.iter("const")
    first_const = next(const_iter)
    second_const = next(const_iter)
    offset = parse_vl_int(first_const.get("name"))
    width = parse_vl_int(second_const.get("name"))
    return name + parse_sel_field(xml_doc, dtype_id, offset, width)

def parse_arraysel_name(xml_doc, xml_node):
    if xml_node.tag == "arraysel":
        first_child = xml_node.find("*")
        name = parse_arraysel_name(xml_doc, first_child)
        const_iter = xml_node.iter("const")
        first_const = next(const_iter)
        offset = parse_vl_int(first_const.get("name"))
        name = f"{name}_{offset}" # array indexing is not supported in VCD
    else:
        name = parse_var_name(xml_doc, xml_node)
    return name

def parse_vl_port(xml_doc, xml_node, signals):
    total_width = 0
    if xml_node.tag == "concat":
        child_nodes = xml_node.findall("*")
        for xml_child in child_nodes:
            total_width = total_width + parse_vl_port(xml_doc, xml_child, signals)
    elif xml_node.tag in ["varref", "varxref"]:
        name = parse_var_name(xml_doc, xml_node)
        dtype_id = xml_node.get("dtype_id")
        signal_width = parse_dtype_width(xml_doc, dtype_id)
        signals.append([name, signal_width])
        total_width = total_width + signal_width
    elif xml_node.tag == "sel":
        name = parse_sel_name(xml_doc, xml_node)
        dtype_id = xml_node.get("dtype_id")
        signal_width = parse_dtype_width(xml_doc, dtype_id)
        signals.append([name, signal_width])
        total_width = total_width + signal_width
    elif xml_node.tag == "arraysel":
        name = parse_arraysel_name(xml_doc, xml_node)
        dtype_id = xml_node.get("dtype_id")
        signal_width = parse_dtype_width(xml_doc, dtype_id)
        signals.append([name, signal_width])
        total_width = total_width + signal_width
    else:
        raise ET.ParseError("invalid probe entry: tag=" + xml_node.tag + ", " + source_loc(xml_doc, xml_node.get("loc")))
    # Check for duplicate signal names
    signal_names = [signal[0] for signal in signals]
    duplicates = set([name for name in signal_names if signal_names.count(name) > 1])
    if len(duplicates) > 0:
        raise ET.ParseError("duplicate signal names: " + ", ".join(duplicates))
    return total_width

def parse_xml(filename, max_taps):
    xml_doc = ET.parse(filename)
    modules = {}
    xml_modules = xml_doc.findall(".//module/[@origName='VX_scope_tap']")
    for xml_module in xml_modules:
        scope_id = parse_vl_int(xml_module.find(".//var/[@name='SCOPE_ID']/const").get("name"))
        xtriggerw = parse_vl_int(xml_module.find(".//var/[@name='XTRIGGERW']/const").get("name"))
        htriggerw = parse_vl_int(xml_module.find(".//var/[@name='HTRIGGERW']/const").get("name"))
        probew = parse_vl_int(xml_module.find(".//var/[@name='PROBEW']/const").get("name"))
        module_name = xml_module.get("name")
        modules[module_name] = [scope_id, xtriggerw, htriggerw, probew]

    taps = []
    xml_instances = xml_doc.iter("instance")
    for xml_instance in xml_instances:
        if (max_taps != -1 and len(taps) >= max_taps):
            break
        defName = xml_instance.get("defName")
        module = modules.get(defName)
        if module is None:
            continue

        xtriggers = []
        htriggers = []
        probes = []

        if module[1] > 0:
            w = parse_vl_port(xml_doc, xml_instance.find(".//port/[@name='xtriggers']/*"), xtriggers)
            if w != module[1]:
                raise ET.ParseError("invalid xtriggers width: actual=" + str(w) + ", expected=" + str(module[1]))

        if module[2] > 0:
            w = parse_vl_port(xml_doc, xml_instance.find(".//port/[@name='htriggers']/*"), htriggers)
            if w != module[2]:
                raise ET.ParseError("invalid htriggers width: actual=" + str(w) + ", expected=" + str(module[2]))

        w = parse_vl_port(xml_doc, xml_instance.find(".//port/[@name='probes']/*"), probes)
        if w != module[3]:
            raise ET.ParseError("invalid probes width: actual=" + str(w) + ", expected=" + str(module[3]))

        signals = probes
        for xtrigger in xtriggers:
            signals.append(xtrigger)
        for htrigger in htriggers:
            signals.append(htrigger)

        loc = xml_instance.get("loc")
        hier = xml_doc.find(".//cell/[@loc='" + loc + "']").get("hier")
        path = hier.rsplit(".", 1)[0]
        taps.append({"id":module[0],
                     "width":module[1] + module[2] + module[3],
                     "signals":signals,
                     "path":path})

    return {"version":"0.1.0", "taps":taps}

def main():
    parser = argparse.ArgumentParser(description='Scope headers generator.')
    parser.add_argument('-o', nargs='?', default='scope.json', metavar='o', help='Output JSON manifest')
    parser.add_argument('-n', nargs='?', default=-1, metavar='n', type=int, help='Maximum number of taps to read')
    parser.add_argument('xml', help='Design XML descriptor file')
    args = parser.parse_args()
    #print("args=", args)
    scope_taps = parse_xml(args.xml, args.n)
    with open(args.o, "w") as f:
        json.dump(scope_taps, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
