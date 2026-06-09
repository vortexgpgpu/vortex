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

import re
import sys

def read_file_contents(filename):
    with open(filename, 'r') as file:
        return file.read()

def write_to_console(output):
    print(output, end='')

def check_module_and_parameter_existence(file_content, top_module, params):
    # Check if the module exists
    module_exists_pattern = re.compile(rf'\bmodule\s+{top_module}\b', re.DOTALL)
    if not re.search(module_exists_pattern, file_content):
        print(f"Error: Top module '{top_module}' not found in file '{filename}'.")
        sys.exit(1)

    # Check if parameters exist
    for param in params.keys():
        param_exists_pattern = re.compile(rf'\bparameter\s+((?:\[\s*\d*\s*:\s*\d*\s*\]\s*)?(\w+\s+)?){param}\s*(\[\s*\])?\s*=', re.DOTALL)    
        if not re.search(param_exists_pattern, file_content):
            print(f"Error: Parameter '{param}' not found in module '{top_module}'.")
            sys.exit(1)

def replace_parameter(file_content, top_module, param, value):
    # Define a pattern to locate the specified top module's parameter section
    module_header_pattern = re.compile(rf'(module\s+{top_module}\s*.*?\(\s*)(.*?)(\)\s*;)', re.DOTALL)
    param_declaration_pattern = re.compile(rf'(\bparameter\b\s+(?:\w+\s+)?{param}\s*=\s*)([^,;]+)', re.DOTALL)

    def parameter_replacer(match):
        before_params, params_section, after_params = match.groups()
        # Check if the specific parameter is found within the parameter section
        if re.search(param_declaration_pattern, params_section):
            # Replace the parameter value, avoiding f-string for backreference
            new_params_section = re.sub(param_declaration_pattern, lambda m: m.group(1) + value, params_section)
            return f'{before_params}{new_params_section}{after_params}'
        else:
            return match.group(0)  # Return original content if parameter not found

    # Apply the replacement within the specified top module
    modified_content, num_replacements = re.subn(module_header_pattern, parameter_replacer, file_content)

    if num_replacements == 0:
        print(f"Warning: Top module '{top_module}' or parameter '{param}' not found. No replacement made.")

    return modified_content

def main():
    args = sys.argv[1:]
    filename = ''
    top_module = ''
    params = {}

    for i, arg in enumerate(args):
        if arg.startswith('-G'):
            param, value = arg[2:].split('=')
            params[param] = value
        elif arg.startswith('-T'):
            top_module = arg[2:]
        else:
            filename = arg

    if not top_module:
        print("Error: Top module not specified.")
        sys.exit(1)

    if not filename:
        print("Error: Verilog file name not specified.")
        sys.exit(1)

    file_content = read_file_contents(filename)

    check_module_and_parameter_existence(file_content, top_module, params)

    for param, value in params.items():
        file_content = replace_parameter(file_content, top_module, param, value)

    write_to_console(file_content)

if __name__ == "__main__":
    main()
