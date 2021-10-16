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

exclude_files = []
include_dirs = []
macros = []
br_stack = []

def translate_ternary(text):
    
    def skip_space(text, i, ln, step):
        while (i >= 0) and (i < ln):
            c = text[i]
            if not c.isspace():
                break
            i += step        
        return i

    def skip_expr(text, i, ln, step):                        
        paren = 0
        checkparen = True
        while (i >= 0) and (i < ln):
            c = text[i]                                    
            if checkparen and (((step < 0) and (c == ')')) or ((step > 0) and (c == '('))):
                paren += 1
            elif checkparen and (((step < 0) and (c == '(')) or ((step > 0) and (c == ')'))):
                if (0 == paren):
                    break
                paren -= 1
                if (0 == paren):
                    i = skip_space(text, i + step, ln, step)
                    checkparen = False
                    continue
            elif (0 == paren) and not (c.isalnum() or (c == '_')):
                break
            i += step
        return (i - step) 

    def parse_ternary(text):
        ternary = None
        ln = len(text)    
        for i in range(1, ln):
            c = text[i]
            if not (c == '?'):
                continue
            # parse condition expression              
            i0 = skip_space(text, i - 1, ln, -1)
            if (i < 0):
                raise Exception("invalid condition expression")
            i1 = skip_expr(text, i0, ln, -1)
            if (i1 > i0):
                raise Exception("invalid condition expression")
            # parse true expression                
            i2 = skip_space(text, i + 1, ln, 1)
            if (i2 >= ln):
                raise Exception("invalid true expression")
            i3 = skip_expr(text, i2, ln, 1)
            if (i3 < i2):
                raise Exception("invalid true expression")                        
            # parse colon                
            i4 = skip_space(text, i3 + 1, ln, 1)
            if (i4 >= ln):
                raise Exception("invalid colon")
            if not (text[i4] == ':'):
                raise Exception("missing colon")
            # parse false expression           
            i5 = skip_space(text, i4 + 1, ln, 1)
            if (i5 >= ln):
                raise Exception("invalid false expression")
            i6 = skip_expr(text, i5, ln, 1)
            if (i6 < i5):
                raise Exception("invalid false expression")
            ternary = (i0, i1, i2, i3, i5, i6) 
            break
        return ternary

    while True:
        pos = parse_ternary(text)    
        if pos is None:
            break
        # convert to python ternary
        newText = text[:pos[1]] + text[pos[2]:pos[3]+1] + " if " + text[pos[1]:pos[0]+1] + " else " + text[pos[4]:pos[5]+1] + text[pos[5]+1:]
        text = newText

    return text

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

def load_include_path(dir):
    if not dir in include_dirs:
        print("*** include path: " + dir)
        include_dirs.append(dir)

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
    raise Exception("couldn't find include file: " + filename + " in " + parent_dir)

def remove_comments(text):
    text = re.sub(re.compile("/\*.*?\*/",re.DOTALL ), "", text) # multiline
    text = re.sub(re.compile("//.*?\n" ), "\n", text) # singleline
    return text

def add_macro(name, args, value):
    macro = (name, args, value)
    macros.append(macro)
    '''
    if not args is None:
        print("*** token: " + name + "(", end='')        
        for i in range(len(args)):
            if i > 0:
                print(', ', end='')
            print(args[i], end='')
        print(")=" + value)
    else:
        print("*** token: " + name + "=" + value)
    '''

def find_macro(name):
    for macro in macros:
        if macro[0] == name:
            return macro
    return None

def expand_text(text, params):

    def re_pattern_args(args):
        p = "(?<![0-9a-zA-Z_])("
        i = 0
        for arg in args:
            if i > 0:
                p += "|"    
            p += arg
            i += 1
        p += ")(?![0-9a-zA-Z_])"
        return p

    class DoReplParam(object):
        def __init__(self, params):            
            self.params = params
            self.expanded = False
        def __call__(self, match):
            name = match.group(1)
            self.expanded = True
            return self.params[name]

    class DoReplMacro(object):
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
                        
                        pattern = re_pattern_args(args)
                        params = {}
                        for i in range(len(args)):
                            params[args[i]] = f_args[0][i]
                        dorepl = DoReplParam(params)
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
        if iter > 65536:
            raise Exception("Macro recursion!")    
        has_func = False
        while True:
            params_updated = False
            if not params is None:
                do_repl = DoReplParam(params)
                pattern = re_pattern_args(params)
                new_text = re.sub(pattern, do_repl, text)    
                if do_repl.expanded:
                    text = new_text
                    params_updated = True
            do_repl = DoReplMacro()
            new_text = re.sub(vl_expand_re, do_repl, text)
            has_func = do_repl.has_func            
            if not (params_updated or do_repl.expanded):
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
    print("*** parsing: " + filename + "...")        
    if nesting > 99:
        raise Exception("include recursion!")    
    #print("*** parsing '" + filename + "'...")    
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
            #print("*** " + key + "(" + cond + ") => " + str(taken))
            continue  
        # parse endif
        m = re.match(vl_endif_re, line)
        if m:
            key = m.group(1)
            top = br_stack.pop()
            if key == 'else':                
                br_stack.append(not top)
            #print("*** " + key)
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
        load_include_path(os.path.dirname(include))

    # restore current directory
    os.chdir(old_dir)      

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

def eval_node(text, params):
    def clog2(x):
        l2 = math.log2(x)
        cl = math.ceil(l2)
        return int(cl)

    if not type(text) == str:
        return text

    expanded = expand_text(text, params)
    if expanded:
        text = expanded    

    try:        
        __text = text.replace('$clog2', '__clog2')
        __text = translate_ternary(__text)
        __text = __text.replace('||', 'or')
        __text = __text.replace('&&', 'and')
        e = eval(__text, {'__clog2': clog2})
        return e
    except (NameError, SyntaxError):
        return text

def gen_vl_header(file, modules, taps):

    header = '''
`ifndef VX_SCOPE_DEFS
`define VX_SCOPE_DEFS
'''
    footer = '`endif'

    def signal_size(size, mn):
        if type(size) == int:
            if (size != mn):
                return "[" + str(size-1) + ":0]"
            else:
                return ""
        else:
            return "[" + size + "-1:0]"

    def create_signal(key, ports):
        if not key in ports:
            ports[key] = []
        return ports[key]

    def dic_insert(gdic, ldic, key, value, enabled):
        if enabled:
            ldic[key] = value
        if key in gdic:
            return False
        if enabled:
            gdic[key] = None        
        return True

    def trigger_name(name, size):
        if type(size) == int:
            if size != 1:
                return "(| " + name + ")"
            else:
                return name
        else:
            return "(| " + name + ")"

    def trigger_subscripts(asize):
        def Q(arr, ss, asize, idx, N):
            a = asize[idx]
            if (a != 0):
                for i in range(a):  
                    tmp = ss + '[' + str(i) + ']'                  
                    if (idx + 1) < N:
                        Q(arr, tmp, asize, idx + 1, N)
                    else:
                        arr.append(tmp)           
            else:                
                if (idx + 1) < N:
                    Q(arr, ss, asize, idx + 1, N)
                else:
                    arr.append(ss)

        if asize is None:
            return [""]        
        ln = len(asize)   
        if (0 == ln):     
            return [""]
        arr = []
        Q(arr, "", asize, 0, ln)
        return arr


    def visit_path(alltaps, ports, ntype, paths, modules, taps):
        curtaps = {}

        if (len(paths) != 0):
            spath = paths.pop(0)
            snodes = modules[ntype]["submodules"]                        
            if not spath in snodes:
                raise Exception("invalid path: " + spath + " in " + ntype)  

            snode = snodes[spath]

            stype = snode["type"]

            enabled = True
            if "enabled" in snode:
                enabled = eval_node(snode["enabled"], None)

            subtaps = visit_path(alltaps, ports, stype, paths, modules, taps)
            
            scount = 0   
            if "count" in snode:
                scount = eval_node(snode["count"], None)

            params = None
            if "params" in snode:
                params = snode["params"]

            new_staps = []

            nn = "SCOPE_IO_" + ntype
            pp = create_signal(nn, ports)
        
            for key in subtaps:
                subtap = subtaps[key]
                s = subtap[0]
                a = subtap[1]
                t = subtap[2]                                                        

                aa = [scount]
                sa = signal_size(scount, 0)
                if a:
                    for i in a:
                        x = eval_node(i, params)
                        aa.append(x)
                        sa += signal_size(x, 0)
                
                if dic_insert(alltaps, curtaps, spath + '/' + key, (s, aa, t), enabled):
                    skey = key.replace('/', '_')
                    if enabled:
                        pp.append("\toutput wire" + sa + signal_size(s, 1) + " scope_" + spath + '_' + skey + ',')
                    new_staps.append(skey)

            ports[nn] = pp

            if (0 == scount):
                nn = "SCOPE_BIND_" + ntype + '_' + spath             
                pp = create_signal(nn, ports)

                for st in new_staps:
                    if enabled:
                        pp.append("\t.scope_" + st + "(scope_" + spath + '_' + st + "),")
                    else:
                        pp.append("\t`UNUSED_PIN (scope_" + st + "),")

                ports[nn] = pp
            else:
                nn = "SCOPE_BIND_" + ntype + '_' + spath + "(__i__)"
                pp = create_signal(nn, ports)

                for st in new_staps:
                    if enabled:
                        pp.append("\t.scope_" + st + "(scope_" + spath + '_' + st + "[__i__]),")
                    else:
                        pp.append("\t`UNUSED_PIN (scope_" + st + "),")

                ports[nn] = pp
        else:
            nn = "SCOPE_IO_" + ntype
            pp = create_signal(nn, ports) 
            
            for tk in taps:
                trigger = 0
                name = tk
                size = eval_node(taps[tk], None)
                if name[0] == '!':
                    name = name[1:]
                    trigger = 1
                elif name[0] == '?':
                    name = name[1:]
                    trigger = 2
                if dic_insert(alltaps, curtaps, name, (size, None, trigger), True):
                    pp.append("\toutput wire" + signal_size(size, 1) + " scope_" + name + ',')

            ports[nn] = pp

        return curtaps

    toptaps = {}

    with open(file, 'w') as f:        

        ports = {}
        alltaps = {}
        
        for key in taps:
            skey_list = key.split(',')
            _taps = taps[key]
            for skey in skey_list:
                #print('*** processing node: ' + skey + ' ...')
                paths = skey.strip().split('/')
                ntype = paths.pop(0)
                curtaps = visit_path(alltaps, ports, ntype, paths, modules, _taps)
                for tk in curtaps:
                    toptaps[tk] = curtaps[tk]

        print(header, file=f)

        for key in ports:
            print("`define " + key + ' \\', file=f)
            for port in ports[key]:
                print(port + ' \\', file=f)
            print("", file=f)

        print("`define SCOPE_DECL_SIGNALS \\", file=f)
        i = 0
        for key in toptaps:
            tap = toptaps[key]
            name = key.replace('/', '_')
            size = tap[0]
            asize = tap[1]
            sa = ""
            if asize:
                for a in asize:
                    sa += signal_size(a, 0)
            if i > 0:
                print(" \\", file=f)
            print('\t wire' + sa + signal_size(size, 1) + " scope_" + name + ';', file=f, end='')
            i += 1
        print("", file=f)
        print("", file=f)

        print("`define SCOPE_DATA_LIST \\", file=f)
        i = 0
        for key in toptaps:
            tap = toptaps[key]
            trigger = tap[2]
            if trigger != 0:
                continue
            name = key.replace('/', '_')
            if i > 0:
                print(", \\", file=f)
            print("\t scope_" + name, file=f, end='')
            i += 1
        print("", file=f)
        print("", file=f)

        print("`define SCOPE_UPDATE_LIST \\", file=f)
        i = 0
        for key in toptaps:
            tap = toptaps[key]
            trigger = tap[2]
            if trigger == 0:
                continue
            name = key.replace('/', '_')
            if i > 0:
                print(", \\", file=f)
            print("\t scope_" + name, file=f, end='')
            i += 1
        print("", file=f)
        print("", file=f)

        print("`define SCOPE_TRIGGER \\", file=f)
        i = 0
        for key in toptaps:
            tap = toptaps[key]
            if tap[2] != 2:
                continue
            size = tap[0]
            asize = tap[1]            
            sus = trigger_subscripts(asize)
            for su in sus:
                if i > 0:
                    print(" | \\", file=f)         
                print("\t(", file=f, end='')            
                name = trigger_name("scope_" + key.replace('/', '_') + su, size)
                print(name, file=f, end='')
                print(")", file=f, end='')         
                i += 1
        print("", file=f)
        print("", file=f)

        print(footer, file=f)

    return toptaps

def gen_cc_header(file, taps):

    header = '''
#pragma once

struct scope_module_t {
    const char* name;
    int index;
    int parent;
};

struct scope_tap_t {
    int width;
    const char* name;
    int module;
};
'''
    def flatten_path(paths, sizes):
        def Q(arr, ss, idx, N, paths, sizes):
            size = sizes[idx]
            if size != 0:
                for i in range(sizes[idx]):  
                    tmp = ss + ('/' if (ss != '') else '')
                    tmp += paths[idx] + '_' + str(i)
                    if (idx + 1) < N:
                        Q(arr, tmp, idx + 1, N, paths, sizes)
                    else:
                        arr.append(tmp)            
            else:
                tmp = ss + ('/' if (ss != '') else '')
                tmp += paths[idx]
                if (idx + 1) < N:
                    Q(arr, tmp, idx + 1, N, paths, sizes)
                else:
                    arr.append(tmp)

        arr = []
        Q(arr, "", 0, len(asize), paths, asize)
        return arr

    # flatten the taps
    fdic = {}
    for key in taps:
        tap = taps[key]
        size = str(tap[0])            
        trigger = tap[2]
        if (trigger != 0):
            continue
        paths = key.split('/')
        if (len(paths) > 1):                
            name = paths.pop(-1)
            asize = tap[1]    
            for ss in flatten_path(paths, asize):
                fdic[ss + '/' + name ] = [size, 0]
        else:
            fdic[key] = [size, 0]
    for key in taps:
        tap = taps[key]
        size = str(tap[0])            
        trigger = tap[2]
        if (trigger == 0):
            continue
        paths = key.split('/')
        if (len(paths) > 1):                
            name = paths.pop(-1)
            asize = tap[1]    
            for ss in flatten_path(paths, asize):
                fdic[ss + '/' + name ] = [size, 0]
        else:
            fdic[key] = [size, 0]

    # generate module dic
    mdic = {}
    mdic["*"] = ("*", 0, -1)
    for key in fdic:         
        paths = key.split('/')
        if len(paths) == 1:
            continue
        paths.pop(-1)
        parent = 0
        mk = ""
        for path in paths:
            mk += '/' + path
            if not mk in mdic:                
                index = len(mdic)
                mdic[mk] = (path, index, parent)
                parent = index
            else:    
                parent = mdic[mk][1]
        fdic[key][1] = parent

    with open(file, 'w') as f:
        print(header, file=f)

        print("static constexpr scope_module_t scope_modules[] = {", file=f)
        i = 0
        for key in mdic:
            m = mdic[key]
            if i > 0:
                print(',', file=f)
            print("\t{\"" + m[0] + "\", " + str(m[1]) + ", " + str(m[2]) + "}", file=f, end='')                
            i += 1
        print("", file=f)
        print("};", file=f)

        print("", file=f)
        print("static constexpr scope_tap_t scope_taps[] = {", file=f)
        i = 0
        for key in fdic:
            size = fdic[key][0]
            parent = fdic[key][1]
            paths = key.split('/')
            if len(paths) > 1:
                name = paths.pop(-1)
            else:
                name = key
            if i > 0:
                print(',', file=f)
            print("\t{" + size + ", \"" + name + "\", " + str(parent) + "}", file=f, end='')                
            i += 1
        print("", file=f)
        print("};", file=f)

def main():    
    parser = argparse.ArgumentParser(description='Scope headers generator.')
    parser.add_argument('-vl', nargs='?', default='scope-defs.vh', metavar='file', help='Output Verilog header')
    parser.add_argument('-cc', nargs='?', default='scope-defs.h', metavar='file', help='Output C++ header')
    parser.add_argument('-D', nargs='?', action='append', metavar='macro[=value]', help='define macro')
    parser.add_argument('-I', nargs='?', action='append', metavar='<includedir>', help='include directory')
    parser.add_argument('config', help='Json config file')
    args = parser.parse_args()
    print("args=", args)

    global exclude_files
    global include_dirs
    global macros
    global br_stack

    if args.I:
        for dir in args.I:
            load_include_path(dir)
    
    if args.D:
        load_defines(args.D)

    config = load_config(args.config) 
    
    exclude_files.append(os.path.basename(args.vl))

    if "include_paths" in config:
        for path in config["include_paths"]:
            load_include_path(path)

    if "includes" in config:
        parse_includes(config["includes"])
        
    taps = gen_vl_header(args.vl, config["modules"], config["taps"])
    gen_cc_header(args.cc, taps)

if __name__ == '__main__':
    main()