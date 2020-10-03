#!/usr/bin/env python3
import os
import re
import json
import argparse
import math

vl_include_re = re.compile(r"^\s*`include\s+\"(.+)\"")
vl_define_re = re.compile(r"^\s*`define\s+(\w+)(\([\w\s,]*\))?(.*)")
vl_ifdef_re = re.compile(r"^\s*`(ifdef|ifndef|elsif)\s+(\w+)\s*$")
vl_endif_re = re.compile(r"^\s*`(endif|else)\s*$")
vl_expand_re = re.compile(r"`([0-9a-zA-Z_]+)")

parameters = []
exclude_files = []
include_dirs = []
macros = []
br_stack = []

def parse_func_args(text):
    args = []
    arg = ''
    l = len(text)
    if text[0] != '(':
        raise Exception("missing leading parenthesis: " + text)
    paren = 1
    for i in range(1, l):
        c = text[i]

        if c == '(':
            paren += 1
        elif c == ')':
            if paren == 0:
                raise Exception("mismatched parenthesis: (" + i + ") " + text)
            paren -= 1
            if paren == 0:
                l = i
                break

        if c == ',' and paren == 1:
            if arg.strip():
                args.append(arg)
            arg = ''
        else:
            arg += c

    if paren != 0:
        raise Exception("missing closing parenthesis: " + text)

    if arg.strip():
        args.append(arg)

    return (args, l)

def resolve_include_path(filename, parent_dir):    
    if os.path.basename(filename) in exclude_files:
        return None
    if os.path.isfile(filename):
        return os.path.abspath(filename)
    search_dirs = include_dirs
    if parent_dir:
        search_dirs.append(parent_dir)
    for dir in search_dirs:
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            return os.path.abspath(filepath)    
    raise Exception("couldn't find include file: " + filename)

def remove_comments(text):
    text = re.sub(re.compile("/\*.*?\*/",re.DOTALL ), "", text) # multiline
    text = re.sub(re.compile("//.*?\n" ), "\n", text) # singleline
    return text

def add_macro(name, args, value):
    macro = (name, args, value)
    macros.append(macro)
    if not args is None:
        print("*** token: " + name + "(", end='')        
        for i in range(len(args)):
            if i > 0:
                print(', ', end='')
            print(args[i], end='')
        print(")=" + value)
    else:
        print("*** token: " + name + "=" + value)

def find_macro(name):
    for macro in macros:
        if macro[0] == name:
            return macro
    return None

def expand_text(text):

    class DoRepl(object):
        def __init__(self):            
            self.expanded = False
            self.has_func = False
        def __call__(self, match):
            name = match.group(1)
            macro = find_macro(name)
            if macro:
                if not macro[1] is None:
                    self.has_func = True
                else:
                    self.expanded = True
                    return macro[2]
            return "`" + name

    class DoRepl2(object):
        def __init__(self, args, f_args):            
            map = {}
            for i in range(len(args)):
                map[args[i]] = f_args[i]
            self.map = map
        def __call__(self, match):
            for key in match.groups():
                return self.map[key]
            return group

    def repl_func_macro(text):
        expanded = False
        match = re.search(vl_expand_re, text)        
        if match:
            name = match.group(1)
            macro = find_macro(name)
            if macro:
                args = macro[1]
                value = macro[2]
                if not args is None:
                    str_args = text[match.end():].strip()                    
                    f_args = parse_func_args(str_args)
                    if len(args) == 0:
                        if len(f_args[0]) != 0:
                            raise Exception("invalid argments for macro '" + name + "': value=" + text)
                    else:                                         
                        if len(args) != len(f_args[0]):
                            raise Exception("mismatch number of argments for macro '" + name + "': actual=" + len(f_args[0]) + ", expected=" + len(args))
                        
                        pattern = "(?<![0-9a-zA-Z_])("
                        for i in range(len(args)):
                            if i > 0:
                                pattern += "|"    
                            pattern += args[i]
                        pattern += ")(?![0-9a-zA-Z_])"
                        
                        dorepl = DoRepl2(args, f_args[0])
                        value = re.sub(pattern, dorepl, value)

                    str_head = text[0:match.start()]
                    str_tail = text[match.end() + f_args[1]+1:]
                    text = str_head + value + str_tail
                    expanded = True
        if expanded:
            return text
        return None

    changed = False
    iter = 0

    while True:
        if iter > 99:
            raise Exception("Macro recursion!")    
        has_func = False
        while True:
            do_repl = DoRepl()
            new_text = re.sub(vl_expand_re, do_repl, text)
            has_func = do_repl.has_func
            if not do_repl.expanded:
                break
            text = new_text            
            changed = True
        if not has_func:
            break
        expanded = repl_func_macro(text)
        if not expanded:
            break
        text = expanded
        changed = True
        iter += 1            

    if changed:
        return text
    return None

def parse_include(filename, nesting):    
    if nesting > 99:
        raise Exception("include recursion!")    
    print("*** parsing '" + filename + "'...")    
    content = None
    with open(filename, "r") as f:        
        content = f.read()
    # remove comments
    content = remove_comments(content)
    # parse content
    prev_line = None
    for line in content.splitlines(False):
        # skip empty lines
        if re.match(re.compile(r'^\s*$'), line):
            continue
        # merge multi-line lines
        if line.endswith('\\'):
            if prev_line:
                prev_line += line[:len(line) - 1]
            else:
                prev_line = line[:len(line) - 1]
            continue
        if prev_line:
            line = prev_line + line
            prev_line = None      
        # parse ifdef
        m = re.match(vl_ifdef_re, line)
        if m:
            key = m.group(1)
            cond = m.group(2)
            taken = find_macro(cond) is not None
            if key == 'ifndef':
                taken = not taken
            elif key == '"elsif':
                br_stack.pop()
            br_stack.append(taken)
            print("*** " + key + "(" + cond + ") => " + str(taken))
            continue  
        # parse endif
        m = re.match(vl_endif_re, line)
        if m:
            key = m.group(1)
            top = br_stack.pop()
            if key == 'else':                
                br_stack.append(not top)
            print("*** " + key)
            continue
        # skip disabled blocks
        if not all(br_stack):
            continue
        
        # parse include
        m = re.match(vl_include_re, line)
        if m:
            include = m.group(1)
            include = resolve_include_path(include, os.path.dirname(filename))    
            if include:
                parse_include(include, nesting + 1)
            continue
        # parse define
        m = re.match(vl_define_re, line)
        if m:
            name = m.group(1)
            args = m.group(2)
            if args:                
                args = args[1:len(args)-1].strip()
                if args != '':
                    args = args.split(',')
                    for i in range(len(args)):
                        args[i] = args[i].strip()
                else:
                    args = []
            value = m.group(3)
            add_macro(name, args, value.strip())
            continue

def parse_includes(includes):
    # change current directory to include directory
    old_dir = os.getcwd()    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    for include in includes:
        parse_include(include, 0)

    # restore current directory
    os.chdir(old_dir)      

def load_include_dirs(dirs):
    for dir in dirs:
        print("*** include dir: " + dir)
        include_dirs.append(dir)

def load_defines(defines):
    for define in defines:
        key_value = define.split('=', 2)
        name = key_value[0]
        value = ''
        if len(key_value) == 2:
            value = key_value[1]
        add_macro(name, None, value)

def load_config(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    print("condfig=", config)
    return config

def gen_cc_header(file, ports):

    header = '''
#pragma once\n
struct scope_signal_t {
    int width;
    const char* name;
};\n
inline constexpr int __clog2(int n) { return (n > 1) ? 1 + __clog2((n + 1) >> 1) : 0; }\n
static constexpr scope_signal_t scope_signals[] = {'''

    footer = "};"

    def eval_macro(text):
        expanded = expand_text(text)
        if expanded:
            text = expanded
        text = text.replace('$clog2', '__clog2')
        return text

    def asize_name(asize):
        def Q(arr, ss, asize, idx, N):
            for i in range(asize[idx]):  
                tmp = ss + "_" + str(i)                  
                if (idx + 1) < N:
                    Q(arr, tmp, asize, idx + 1, N)
                else:
                    arr.append(tmp)            

        l = len(asize)   
        if l == 0:     
            return [""]
        arr = []
        Q(arr, "", asize, 0, l)
        return arr                  

    with open(file, 'w') as f:
        print(header, file=f)
        i = 0
        for port in ports:                   
            name = port[0]             
            size = eval_macro(str(port[1]))
            for ss in asize_name(port[2]):                
                if i > 0:
                    print(",", file=f)      
                print("\t{" + size + ", \"" + name + ss + "\"}", file=f, end='')    
                i += 1
        print("", file=f)
        print(footer, file=f)

def gen_vl_header(file, taps, triggers):

    header = '''
`ifndef VX_SCOPE_DEFS
`define VX_SCOPE_DEFS
'''
    footer = '`endif'

    def signal_size(size, asize):
        str_asize = ""
        for s in asize:
            if type(s) == int:
                str_asize += "[" + str(s-1) + ":0]"
            else:                
                str_asize += "[" + str(s) + "-1:0]"

        if type(size) == int:
            size1 = (size-1)
            if size1 != 0:
                return str_asize + "[" + str(size1) + ":0]"
            else:
                return str_asize
        else:
            return str_asize + "[(" + size + ")-1:0]"

    def generate_ports(tclass, tap, ports, new_taps):

        def emit_io(tap, ports, prefix, asize, return_list, new_taps, is_enabled):
            stap = tap + "_IO"
            new_taps.append(stap)        
            print("`define " + stap + " \\", file=f)                
            if is_enabled:
                for key in ports:
                    size = ports[key]
                    name = key
                    is_trigger = False
                    if name[0] == '!':
                        name = name[1:]
                        is_trigger = True
                    if not return_list is None:
                        return_list.append((name + prefix, size, asize, is_trigger))
                    print("\toutput wire" + signal_size(size, asize) + " " + name + prefix + ", \\", file=f)
            print("", file=f)            
            emit_bind(tap, ports, prefix, prefix, new_taps, is_enabled)

        def emit_bind(tap, ports, from_prefix, to_prefix, new_taps, is_enabled):
            stap = tap + "_BIND"
            new_taps.append(stap)       
            print("`define " + stap + " \\", file=f)            
            for key in ports:
                name = key
                if name[0] == '!':
                    name = name[1:]
                if is_enabled:
                    print("\t." + name + to_prefix + " (" + name + from_prefix + "), \\", file=f)
                else:
                    if (from_prefix != to_prefix):
                        print("\t`UNUSED_PIN (" + name + to_prefix + "), \\", file=f)
            print("", file=f)

        def emit_select(tap, ports, from_prefix, to_prefix, new_taps, is_enabled):
            stap = tap + "_SELECT(__i__)"
            new_taps.append(stap)      
            print("`define " + stap + " \\", file=f)
            if is_enabled:
                for key in ports:
                    name = key
                    if name[0] == '!':
                        name = name[1:]
                    print("\t." + name + to_prefix + " (" + name + from_prefix + "[__i__]), \\", file=f)    
            print("", file=f)

        def do_top(tap, ports, new_taps):
            out_ports = []
            for p in ports:
                name = p
                is_trigger = False
                if name[0] == '!':
                    name = name[1:]
                    is_trigger = True
                out_ports.append((name, ports[p], [], is_trigger))
            return out_ports

        def do_core(tap, ports, new_taps):
            out_ports = []
            nclusters = parameters["NUM_CLUSTERS"]
            ncores = parameters["NUM_CORES"]
            emit_io(tap + "_TOP", ports, "_top", [nclusters, ncores], out_ports, new_taps, True)
            emit_io(tap + "_CLUSTER", ports, "_cluster", [ncores], None, new_taps, True)
            emit_io(tap + "", ports, "", [], None, new_taps, True)
            emit_select(tap + "_CLUSTER", ports, "_top", "_cluster", new_taps, True)
            emit_select(tap + "", ports, "_cluster", "", new_taps, True)
            return out_ports         

        def do_bank(tap, ports, new_taps):
            out_ports = []

            nclusters = parameters["NUM_CLUSTERS"]
            ncores = parameters["NUM_CORES"]
            has_l3 = (parameters["L3_ENABLE"] != 0)
            has_l2 = (parameters["L2_ENABLE"] != 0)

            emit_io(tap + "_L3_TOP", ports, "_l3_cache", [parameters["L3NUM_BANKS"]], out_ports, new_taps, has_l3)
            emit_io(tap + "_L2_TOP", ports, "_l2_top", [nclusters, parameters["L2NUM_BANKS"]], out_ports, new_taps, has_l2)
            emit_io(tap + "_L1D_TOP", ports, "_l1d_top", [nclusters, ncores, parameters["DNUM_BANKS"]], out_ports, new_taps, True)
            emit_io(tap + "_L1I_TOP", ports, "_l1i_top", [nclusters, ncores, parameters["INUM_BANKS"]], out_ports, new_taps, True)
            emit_io(tap + "_L1S_TOP", ports, "_l1s_top", [nclusters, ncores, parameters["SNUM_BANKS"]], out_ports, new_taps, True)

            emit_io(tap + "_L2_CLUSTER", ports, "_l2_cache", [parameters["L2NUM_BANKS"]], None, new_taps, has_l2)            
            emit_io(tap + "_L1D_CLUSTER", ports, "_l1d_cluster", [ncores, parameters["DNUM_BANKS"]], None, new_taps, True)
            emit_io(tap + "_L1I_CLUSTER", ports, "_l1i_cluster", [ncores, parameters["INUM_BANKS"]], None, new_taps, True)
            emit_io(tap + "_L1S_CLUSTER", ports, "_l1s_cluster", [ncores, parameters["SNUM_BANKS"]], None, new_taps, True)

            emit_io(tap + "_L1D_CORE", ports, "_l1d_cache", [parameters["DNUM_BANKS"]], None, new_taps, True)
            emit_io(tap + "_L1I_CORE", ports, "_l1i_cache", [parameters["INUM_BANKS"]], None, new_taps, True)
            emit_io(tap + "_L1S_CORE", ports, "_l1s_cache", [parameters["SNUM_BANKS"]], None, new_taps, True)

            emit_io(tap + "_CACHE", ports, "_cache", ["NUM_BANKS"], None, new_taps, True)
            emit_io(tap + "", ports, "", [], None, new_taps, True)

            emit_select(tap + "_L2_CLUSTER", ports, "_l2_top", "_l2_cache", new_taps, has_l2)
            emit_select(tap + "_L1D_CLUSTER", ports, "_l1d_top", "_l1d_cluster", new_taps, True)
            emit_select(tap + "_L1I_CLUSTER", ports, "_l1i_top", "_l1i_cluster", new_taps, True)
            emit_select(tap + "_L1S_CLUSTER", ports, "_l1s_top", "_l1s_cluster", new_taps, True)  

            emit_select(tap + "_L1D_CORE", ports, "_l1d_cluster", "_l1d_cache", new_taps, True)
            emit_select(tap + "_L1I_CORE", ports, "_l1i_cluster", "_l1i_cache", new_taps, True)
            emit_select(tap + "_L1S_CORE", ports, "_l1s_cluster", "_l1s_cache", new_taps, True)         

            emit_bind(tap + "_L3_CACHE", ports, "_l3_cache", "_cache", new_taps, has_l3)
            emit_bind(tap + "_L2_CACHE", ports, "_l2_cache", "_cache", new_taps, has_l2)
            emit_bind(tap + "_L1D_CACHE", ports, "_l1d_cache", "_cache", new_taps, True)
            emit_bind(tap + "_L1I_CACHE", ports, "_l1i_cache", "_cache", new_taps, True)
            emit_bind(tap + "_L1S_CACHE", ports, "_l1s_cache", "_cache", new_taps, True)         
            
            emit_select(tap + "", ports, "_cache", "", new_taps, True)

            return out_ports   

        callbacks = {
            "top":  do_top, 
            "core": do_core,
            "bank": do_bank
        }

        return callbacks[tclass](tap, ports, new_taps)

    def trigger_size(name, ports):
        for port in ports:
            if port[0] == name:
                return (port[1], port[2])
        return None

    def trigger_prefices(asize):
        def Q(arr, ss, asize, idx, N):
            for i in range(asize[idx]):  
                tmp = ss + '[' + str(i) + ']'                  
                if (idx + 1) < N:
                    Q(arr, tmp, asize, idx + 1, N)
                else:
                    arr.append(tmp)            

        l = len(asize)   
        if l == 0:     
            return [""]
        arr = []
        Q(arr, "", asize, 0, l)
        return arr         

    def trigger_name(name, size):
        if type(size) == int:
            size1 = (size-1)
            if size1 != 0:
                return "(| " + name + ")"
            else:
                return name
        else:
            return "(| " + name + ")"

    with open(file, 'w') as f:
        print(header, file=f)

        all_ports = []
        new_taps = []

        for key in taps:            
            [tclass, tap] = key.split('::')
            ports = generate_ports(tclass, tap, taps[key], new_taps)
            for port in ports:
                all_ports.append(port)

        print("`define SCOPE_SIGNALS_DECL \\", file=f)
        i = 0  
        for port in all_ports:     
            if i > 0:
                print(" \\", file=f)
            print("\twire" + signal_size(port[1], port[2]) + " " + port[0] + ";", file=f, end='')
            i += 1
        print("", file=f)
        print("", file=f)

        print("`define SCOPE_SIGNALS_DATA_LIST \\", file=f)
        i = 0
        for port in all_ports:
            if port[3]:
                continue
            if i > 0:
                print(", \\", file=f)
            print("\t" + port[0], file=f, end='')            
            i += 1
        print("", file=f)
        print("", file=f)

        print("`define SCOPE_SIGNALS_UPD_LIST \\", file=f)
        i = 0
        for port in all_ports:
            if not port[3]:
                continue
            if i > 0:
                print(", \\", file=f)
            print("\t" + port[0], file=f, end='')            
            i += 1
        print("", file=f)
        print("", file=f)

        print("`define SCOPE_TRIGGERS \\", file=f)
        i = 0
        for trigger in triggers:
            arr = trigger_size(trigger[0], all_ports)
            if arr is None:
                continue
            [size, asize] = arr
            for prefix in trigger_prefices(asize):
                if i > 0:
                    print(" | \\", file=f)                
                print("\t(", file=f, end='')
                for j in range(len(trigger)):
                    if j > 0:
                        print(" && ", file=f, end='')                
                    print(trigger_name(trigger[j] + prefix, size), file=f, end='')
                print(")", file=f, end='')         
                i += 1
        print("", file=f)
        print("", file=f)

        print(footer, file=f)

        return all_ports

def main():    
    parser = argparse.ArgumentParser(description='Scope headers generator.')
    parser.add_argument('-vl', nargs='?', default='scope-defs.vh', metavar='file', help='Output Verilog header')
    parser.add_argument('-cc', nargs='?', default='scope-defs.h', metavar='file', help='Output C++ header')
    parser.add_argument('-D', nargs='?', action='append', metavar='macro[=value]', help='define macro')
    parser.add_argument('-I', nargs='?', action='append', metavar='<includedir>', help='include directory')
    parser.add_argument('config', help='Json config file')
    args = parser.parse_args()
    print("args=", args)

    global parameters
    global exclude_files
    global include_dirs
    global macros
    global br_stack

    if args.I:
        load_include_dirs(args.I)
    
    if args.D:
        load_defines(args.D)

    config = load_config(args.config) 
    
    exclude_files.append(os.path.basename(args.vl))

    if "includes" in config:
        parse_includes(config["includes"])

    parameters = config["parameters"]
    for key in parameters:
        parameters[key] = int(eval(expand_text(str(parameters[key]))))
        
    ports = gen_vl_header(args.vl, config["taps"], config["triggers"])
    gen_cc_header(args.cc, ports)

if __name__ == "__main__":
    main()