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
    return file + " (" + start_line + ":" + start_col + "-" + end_line + ":" + end_col + ")"
    
def parse_dtype_width(xml_doc, dtype_id):
    xml_type = xml_doc.find(".//typetable/*[@id='" + dtype_id + "']")
    if xml_type.tag == "packarraydtype" or xml_type.tag == "unpackarraydtype":
        sub_dtype_id = xml_type.get("sub_dtype_id")
        base_width = parse_dtype_width(xml_doc, sub_dtype_id)
        const = xml_type.iter("const")
        left  = parse_vl_int(next(const).get("name"))
        right = parse_vl_int(next(const).get("name"))
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
        return dotted + '.' + name
    else:
        raise ET.ParseError("invalid probe entry" + source_loc(xml_doc, xml_node.get("loc")))
    return name

def parse_sel_name(xml_doc, xml_node):
    name = parse_var_name(xml_doc, xml_node.find("*"))
    const = xml_node.iter("const")
    offset = parse_vl_int(next(const).get("name"))
    #size = parse_vl_int(next(const).get("name"))
    return name + '_' + str(offset)

def parse_array_name(xml_doc, xml_node):
    if xml_node.tag == "arraysel":
        name = parse_array_name(xml_doc, xml_node.find("*"))
        xml_size = xml_node.find("const").get("name")
        array_size = parse_vl_int(xml_size)
        name = name + '_' + str(array_size)
    else:
        name = parse_var_name(xml_doc, xml_node)
    return name

def parse_vl_port(xml_doc, xml_node, signals):
    total_width = 0
    if xml_node.tag == "concat":
        for xml_child in xml_node.findall("*"):
            total_width = total_width + parse_vl_port(xml_doc, xml_child, signals)
    elif xml_node.tag == "varref" or xml_node.tag == "varxref":
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
        name = parse_array_name(xml_doc, xml_node)
        dtype_id = xml_node.get("dtype_id")
        signal_width = parse_dtype_width(xml_doc, dtype_id)
        signals.append([name, signal_width])
        total_width = total_width + signal_width
    else:
        raise ET.ParseError("invalid probe entry: " + source_loc(xml_doc, xml_node.get("loc")))
    return total_width

def parse_xml(filename, max_taps):
    xml_doc = ET.parse(filename)
    modules = {}
    xml_modules = xml_doc.findall(".//module/[@origName='VX_scope_tap']")
    for xml_module in xml_modules:        
        scope_id = parse_vl_int(xml_module.find(".//var/[@name='SCOPE_ID']/const").get("name"))
        triggerw = parse_vl_int(xml_module.find(".//var/[@name='TRIGGERW']/const").get("name"))
        probew = parse_vl_int(xml_module.find(".//var/[@name='PROBEW']/const").get("name"))
        module_name = xml_module.get("name")
        modules[module_name] = [scope_id, triggerw, probew]

    taps = []
    xml_instances = xml_doc.iter("instance")    
    for xml_instance in xml_instances:      
        if (max_taps != -1 and len(taps) >= max_taps):
            break      
        defName = xml_instance.get("defName")
        module = modules.get(defName)
        if module is None:
            continue
        triggers = []
        probes = []           
        w = parse_vl_port(xml_doc, xml_instance.find(".//port/[@name='triggers']/*"), triggers)
        if w != module[1]:
            raise ET.ParseError("invalid triggers width: actual=" + str(w) + ", expected=" + str(module[1]))
        w = parse_vl_port(xml_doc, xml_instance.find(".//port/[@name='probes']/*"), probes)
        if w != module[2]:
            raise ET.ParseError("invalid probes width: actual=" + str(w) + ", expected=" + str(module[2]))
        signals = probes
        for trigger in triggers:
            signals.append(trigger)
        loc = xml_instance.get("loc")
        hier = xml_doc.find(".//cell/[@loc='" + loc + "']").get("hier")
        path = hier.rsplit(".", 1)[0]
        taps.append({"id":module[0],
                     "width":module[1] + module[2],
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
