#!/usr/bin/env python3

import argparse
import re
import sys
from collections import OrderedDict


INSTANCE_RE = re.compile(r"^\s*\(INSTANCE\s+(\S+)\s*$")
NET_RE = re.compile(
    r"^\s*\((\S+)\s+\(T0\s+(\d+)\)\s+\(T1\s+(\d+)\)"
    r"\s+\(TZ\s+(\d+)\)\s+\(TX\s+(\d+)\)\s+\(TB\s+(\d+)\)"
    r"\s+\(TC\s+(\d+)\)\)\s*$"
)
DURATION_RE = re.compile(r"^\s*\(DURATION\s+(\d+)\)\s*$")
TIMESCALE_RE = re.compile(r"^\s*\(TIMESCALE\s+(.+)\)\s*$")


def paren_delta(line):
    return line.count("(") - line.count(")")


def extract_instances(path, instance_name, count):
    instances = []
    duration = None
    timescale = None
    capture = None
    depth = 0

    with open(path, "r", encoding="utf-8") as src:
        for line in src:
            if duration is None:
                match = DURATION_RE.match(line)
                if match:
                    duration = int(match.group(1))
            if timescale is None:
                match = TIMESCALE_RE.match(line)
                if match:
                    timescale = match.group(1)

            if capture is None:
                match = INSTANCE_RE.match(line)
                if match and match.group(1) == instance_name:
                    capture = []
                    depth = 0

            if capture is not None:
                capture.append(line)
                depth += paren_delta(line)
                if depth == 0:
                    instances.append(capture)
                    capture = None
                    if len(instances) == count:
                        break

    if duration is None or timescale is None:
        raise ValueError("SAIF is missing DURATION or TIMESCALE")
    if len(instances) != count:
        raise ValueError(
            f"expected {count} '(INSTANCE {instance_name}' blocks, found {len(instances)}"
        )
    return duration, timescale, instances


def parse_instance(lines):
    root = INSTANCE_RE.match(lines[0]).group(1)
    stack = [root]
    nets = OrderedDict()
    in_net = False

    for line in lines[1:]:
        instance = INSTANCE_RE.match(line)
        if instance:
            stack.append(instance.group(1))
            continue
        if re.match(r"^\s*\(NET\s*$", line):
            in_net = True
            continue
        net = NET_RE.match(line)
        if in_net and net:
            key = tuple(stack[1:] + [net.group(1)])
            nets[key] = tuple(int(net.group(i)) for i in range(2, 8))
            continue
        if re.match(r"^\s*\)\s*$", line):
            if in_net:
                in_net = False
            elif len(stack) > 1:
                stack.pop()

    return nets


def average_instances(instances):
    parsed = [parse_instance(lines) for lines in instances]
    reference = set(parsed[0])
    for index, nets in enumerate(parsed[1:], 2):
        if set(nets) != reference:
            missing = len(reference - set(nets))
            extra = len(set(nets) - reference)
            raise ValueError(
                f"FEDP instance {index} has mismatched nets: missing={missing}, extra={extra}"
            )

    averaged = OrderedDict()
    for key in parsed[0]:
        columns = zip(*(nets[key] for nets in parsed))
        averaged[key] = tuple(round(sum(values) / len(parsed)) for values in columns)
    return averaged


def normalize_name(name):
    return re.sub(r"\((\d+)\)", lambda match: rf"\[{match.group(1)}\]", name)


def opensta_name(name):
    return re.sub(r"\\([\[\]])", r"\1", normalize_name(name))


def emit_saif(path, duration, timescale, nets):
    tree = OrderedDict()
    for path_parts, activity in nets.items():
        node = tree
        for part in path_parts[:-1]:
            node = node.setdefault(normalize_name(part), OrderedDict())
        node.setdefault("__nets__", OrderedDict())[normalize_name(path_parts[-1])] = activity

    def emit_node(out, node, indent):
        net_map = node.get("__nets__", {})
        if net_map:
            out.write(f"{indent}(NET\n")
            for name, values in net_map.items():
                t0, t1, tz, tx, tb, tc = values
                out.write(
                    f"{indent} ({name} (T0 {t0}) (T1 {t1}) (TZ {tz}) "
                    f"(TX {tx}) (TB {tb}) (TC {tc}))\n"
                )
            out.write(f"{indent})\n")
        for name, child in node.items():
            if name == "__nets__":
                continue
            out.write(f"{indent}(INSTANCE {name}\n")
            emit_node(out, child, indent + " ")
            out.write(f"{indent})\n")

    with open(path, "w", encoding="utf-8") as out:
        out.write("(SAIFILE\n")
        out.write('(SAIFVERSION "2.0")\n')
        out.write('(DIRECTION "backward")\n')
        out.write('(DESIGN "VX_tcu_fedp_tfr")\n')
        out.write('(VENDOR "Verilator")\n')
        out.write('(PROGRAM_NAME "average_fedp_saif.py")\n')
        out.write("(DIVIDER / )\n")
        out.write(f"(TIMESCALE {timescale})\n")
        out.write(f"(DURATION {duration})\n")
        out.write(" (INSTANCE VX_tcu_fedp_tfr\n")
        emit_node(out, tree, "  ")
        out.write(" )\n")
        out.write(")\n")


def emit_opensta_tcl(path, duration, nets, clock_frequency_mhz):
    clock_activity = next(
        (
            values
            for parts, values in nets.items()
            if len(parts) == 1 and opensta_name(parts[0]) == "clk"
        ),
        None,
    )
    if clock_activity is None or clock_activity[5] == 0:
        raise ValueError("cannot normalize OpenSTA activity without top-level clk toggles")

    clock_toggle_density = 2 * clock_frequency_mhz * 1e6
    clock_tc = clock_activity[5]
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Auto-generated from averaged FEDP SAIF input activity.\n")
        out.write(
            f"# Toggle densities are normalized to a {clock_frequency_mhz:g} MHz clock.\n"
        )
        for parts, values in nets.items():
            names = [opensta_name(part) for part in parts]
            if names[-1] == "clk":
                continue
            _, t1, _, _, _, tc = values
            normalized = "/".join(names)
            density = tc / clock_tc * clock_toggle_density
            duty = t1 / duration
            if len(parts) == 1:
                query = f"get_ports -quiet {{{normalized}}}"
                option = "-input_ports"
            else:
                query = f"get_pins -quiet {{{normalized}}}"
                option = "-pins"
            out.write(f"set activity_object [{query}]\n")
            out.write("if {[llength $activity_object] > 0} {\n")
            out.write(
                f"  set_power_activity {option} $activity_object "
                f"-density {density:.12g} -duty {duty:.12g}\n"
            )
            out.write("}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Average matching activity from repeated FEDP SAIF instances."
    )
    parser.add_argument("input", help="Verilator SAIF from full rtlsim")
    parser.add_argument("output", help="Standalone VX_tcu_fedp_tfr SAIF")
    parser.add_argument("--instance", default="fedp", help="instance basename to extract")
    parser.add_argument("--count", type=int, default=4, help="number of instances to average")
    parser.add_argument("--opensta-tcl", help="write measured top-input activity as OpenSTA Tcl")
    parser.add_argument(
        "--clock-frequency-mhz",
        type=float,
        default=800,
        help="target clock frequency used to scale OpenSTA toggle density",
    )
    args = parser.parse_args()

    try:
        duration, timescale, instances = extract_instances(
            args.input, args.instance, args.count
        )
        nets = average_instances(instances)
        emit_saif(args.output, duration, timescale, nets)
        if args.opensta_tcl:
            emit_opensta_tcl(
                args.opensta_tcl, duration, nets, args.clock_frequency_mhz
            )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Averaged {args.count} '{args.instance}' instances into {args.output} "
        f"({len(nets)} nets, duration={duration} {timescale})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
