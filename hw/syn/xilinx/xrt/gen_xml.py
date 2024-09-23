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

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generate_xml(numbanks, datawidth, addresswidth, offset, output_file):
    root = ET.Element("root", versionMajor="1", versionMinor="6")
    kernel = ET.SubElement(root, "kernel", name="vortex_afu", language="ip_c",
                           vlnv="mycompany.com:kernel:vortex_afu:1.0",
                           attributes="", preferredWorkGroupSizeMultiple="0",
                           workGroupSize="1", interrupt="true")

    ports = ET.SubElement(kernel, "ports")

    # control ports
    ET.SubElement(ports, "port", name="s_axi_ctrl", mode="slave", range="0x1000", dataWidth="32", portType="addressable", base="0x0")

    # memory ports
    for i in range(numbanks):
        port_name = f"m_axi_mem_{i}"
        ET.SubElement(ports, "port", name=port_name, mode="master", range=f"0x{(1 << addresswidth) - 1:X}", dataWidth=str(datawidth), portType="addressable", base=f"0x0")

    args = ET.SubElement(kernel, "args")

    # control args
    ET.SubElement(args, "arg", name="dev", addressQualifier="0", id="0", port="s_axi_ctrl", size="0x4", offset="0x010", type="uint", hostOffset="0x0", hostSize="0x4")
    ET.SubElement(args, "arg", name="isa", addressQualifier="0", id="1", port="s_axi_ctrl", size="0x4", offset="0x018", type="uint", hostOffset="0x0", hostSize="0x4")
    ET.SubElement(args, "arg", name="dcr", addressQualifier="0", id="2", port="s_axi_ctrl", size="0x4", offset="0x020", type="uint", hostOffset="0x0", hostSize="0x4")
    ET.SubElement(args, "arg", name="scp", addressQualifier="0", id="3", port="s_axi_ctrl", size="0x4", offset="0x028", type="uint", hostOffset="0x0", hostSize="0x4")

    # memory args
    for i in range(numbanks):
        arg_name = f"mem_{i}"
        ET.SubElement(args, "arg", name=arg_name, addressQualifier="1", id=str(4 + i),
                      port=f"m_axi_mem_{i}", size="0x8", offset=f"0x{offset + (i * 8):X}",
                      type="int*", hostOffset="0x0", hostSize="0x8")

    # Pretty-print and write the XML to file
    with open(output_file, "w") as f:
        f.write(prettify(root))

def main():
    parser = argparse.ArgumentParser(description="Kernel Configuration File Generator")
    parser.add_argument("-n", "--numbanks", type=int, default=1, help="Number of AXI memory banks")
    parser.add_argument("-d", "--datawidth", type=int, default=512, help="Data width of the AXI memory ports")
    parser.add_argument("-a", "--addresswidth", type=int, default=28, help="Address width of the AXI memory ports")
    parser.add_argument("-x", "--offset", type=lambda x: int(x, 0), default=0x30, help="Starting offset for kernel args (hex)")
    parser.add_argument("-o", "--output", type=str, default="kernel.xml", help="Output XML file name")
    args = parser.parse_args()

    # Call the generate function
    generate_xml(args.numbanks, args.datawidth, args.addresswidth, args.offset, args.output)

if __name__ == "__main__":
    main()
