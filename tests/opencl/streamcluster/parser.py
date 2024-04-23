import sys

def parser(filename):
    output = open('output.txt', 'a')
    try:
        with open(filename, 'r') as file:
            instrs = 0
            cycles = 0
            for line in file:
                if line.startswith('PERF: instrs'):
                    print(line.strip())
                    start = line.find("instrs=") + len("instrs=")
                    end = line.find(",", start)
                    instrs += int(line[start:end])
                    start = line.find("cycles=") + len("cycles=")
                    end = line.find(",", start)
                    cycles += int(line[start:end])
            sum_str = filename + " " + str(instrs) + " " + str(cycles) + " " + str((instrs / cycles))
            output.write(sum_str);
            output.close()

    except FileNotFoundError:
        print(f"The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if len(sys.argv) != 2:
    print("Usage: python script.py <filename>")
else:
    filename = sys.argv[1]
    parser(filename)
        
