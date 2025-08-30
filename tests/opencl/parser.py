import os
import re
import sys 

def parse_perf_file(filepath):
    instrs_total = 0
    cycles_total = 0
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'instrs=(\d+), cycles=(\d+)', line)
            if match:
                instrs_total += int(match.group(1))
                cycles_total += int(match.group(2))
    return instrs_total, cycles_total

def extract_O_value(filename):
    match = re.search(r'_O(\d+)\.txt', filename)
    return int(match.group(1)) if match else None

def run_perf_summary_on_folder(folder_path):
    str_option = ""
    str_data = ""
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt') and '_O' in filename:
            O_value = extract_O_value(filename)
            if O_value is not None:
                filepath = os.path.join(folder_path, filename)
                instrs_total, cycles_total = parse_perf_file(filepath)  
                #print(f"{O_value} {instrs_total} {cycles_total}")  
                str_option = str_option + str(O_value) + " "
                str_data = str_data + str(instrs_total) + " " + str(cycles_total) + " " 
    print(str_option)
    print(str_data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)
    folder_path = sys.argv[1]
    run_perf_summary_on_folder(folder_path)
