#!/usr/bin/env python3
"""Build and measure the OpenCL ThreadSplit matrix from isolated generated trees."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATIONS = ('IPDOM_native', 'IPDOM_SW', 'SCS', 'ITS')
FIELDS = ['benchmark', 'parameters', 'threads', 'warps', 'cores', 'backend',
          'configuration', 'status', 'cycles', 'dynamic_instructions', 'issued_instructions',
          'active_lane_instructions', 'simd_utilization', 'vx_yield_count',
          'subgroup_switch_count', 'watchdog_switch_count', 'peak_live_split_contexts',
          'normalized_time', 'normalized_instructions', 'wall_seconds', 'log', 'notes']


def matrix(pilot=False):
    cases = []
    def add(benchmark, **params):
        base = dict(size=256, resources=1, rounds=4, percent=100, stride=1, local=32)
        base.update(params)
        cases.append((benchmark, base))
    if pilot:
        for benchmark in ('tl_cg', 'tl_fg', 'am_cg', 'am_fg', 'lockht', 'atm', 'lclist', 'cp_ds', 'bh_st'):
            add(benchmark, size=32, resources=16 if benchmark == 'cp_ds' else
                8 if benchmark in ('tl_fg', 'am_fg', 'lockht', 'atm', 'lclist') else 1, rounds=1)
        return cases
    for rounds in (4, 16):
        add('tl_cg', rounds=rounds)
    for locks in (2, 8, 32):
        for stride in (1, 16):
            add('tl_fg', resources=locks, stride=stride)
    for percent in (25, 50, 100):
        add('am_cg', percent=percent)
    for locks in (2, 8, 32):
        for percent in (25, 100):
            add('am_fg', resources=locks, percent=percent)
    for size in (128, 512):
        for buckets in (1, 8, 64):
            add('lockht', size=size, resources=buckets, rounds=1)
    for accounts in (2, 8, 32):
        add('atm', resources=accounts)
    for keys in (8, 32, 128):
        for updates in (0, 50, 100):
            add('lclist', size=128, resources=keys, percent=updates, rounds=1)
    for side in (4, 8, 16):
        add('cp_ds', size=512, resources=side * side, rounds=2)
    for size in (128, 256, 512):
        add('bh_st', size=size, rounds=1)
    for benchmark, args in [('atomicreduce', '-n1024'), ('histogram', '-n1024'),
                            ('psort', '-f -n256'), ('pathfinder', '-r16 -c64 -y2 -b16')]:
        cases.append((benchmark, {'args': args}))
    return cases


def environment(threads):
    env = os.environ.copy()
    for key in ('DEBUG', 'PERF', 'SCOPE', 'CONFIGS'):
        env.pop(key, None)
    env['CONFIGS'] = (f'-DVX_CFG_EXT_A_ENABLE -DVX_CFG_NUM_CORES=1 -DVX_CFG_NUM_WARPS=8 '
                      f'-DVX_CFG_NUM_THREADS={threads} -DTHREADSPLIT_EVAL')
    env['CCACHE_DISABLE'] = '1'
    return env


def command(args, cwd, env, logfile):
    with logfile.open('w') as out:
        subprocess.run(args, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def prepare(build, backend, threads, tooldir):
    build.mkdir(exist_ok=True)
    env = environment(threads)
    command(['../configure', '--xlen=32', f'--tooldir={tooldir}'], build, env, build / 'configure.log')
    command(['make', '-s', '-j8', '-C', 'sw/kernel'], build, env, build / 'kernel-build.log')
    command(['make', '-s', '-j8', '-C', f'sw/runtime/{backend}', f'DESTDIR={build}/sw/runtime'],
            build, env, build / f'{backend}-build.log')


def app_environment(build, backend, threads, app):
    env = environment(threads)
    command(['make', '-s', '-C', f'tests/opencl/{app}', 'all'], build, env, build / f'{app}-build.log')
    output = subprocess.check_output(['make', '-s', '-n', '-C', f'tests/opencl/{app}',
                                      f'run-{backend}'], cwd=build, env=env, text=True)
    launch = [line for line in output.splitlines() if line.startswith('LD_LIBRARY_PATH=')][-1]
    tokens = shlex.split(launch)
    for token in tokens:
        if '=' not in token or token.startswith('./'):
            break
        key, value = token.split('=', 1)
        env[key] = value
    # Run the copy rules from the generated Makefile without starting the app.
    command(['make', '-s', '-C', f'tests/opencl/{app}', 'kernel.cl'], build, env,
            build / f'{app}-kernel-copy.log')
    source = ROOT / 'tests/opencl' / app / 'common.h'
    if source.exists():
        command(['make', '-s', '-C', f'tests/opencl/{app}', 'common.h'], build, env,
                build / f'{app}-common-copy.log')
    return env


def write_csv(path, rows):
    bases = {}
    for row in rows:
        normal = 'args' in json.loads(row['parameters'])
        baseline = 'IPDOM_native' if normal else 'IPDOM_SW'
        if row['configuration'] == baseline and row['status'] == 'PASS':
            bases[(row['benchmark'], row['parameters'], str(row['threads']), row['backend'])] = row
    for row in rows:
        key = (row['benchmark'], row['parameters'], str(row['threads']), row['backend'])
        if row['status'] == 'PASS' and key in bases:
            for raw, norm in [('cycles', 'normalized_time'), ('dynamic_instructions', 'normalized_instructions')]:
                row[norm] = float(row[raw]) / float(bases[key][raw])
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, required=True, choices=(4, 8, 16, 32))
    parser.add_argument('--backend', default='rtlsim', choices=('rtlsim', 'simx'))
    parser.add_argument('--tooldir', type=Path, default=ROOT / 'build32-eval-tools')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--native-timeout', type=int, default=20)
    parser.add_argument('--configurations', nargs='+', default=list(CONFIGURATIONS))
    parser.add_argument('--benchmarks', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build = ROOT / f'build32-eval-t{args.threads}'
    if args.prepare:
        prepare(build, args.backend, args.threads, args.tooldir)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / f'{args.backend}-t{args.threads}.csv'
    rows = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
    # A resumed invocation retries nonterminal rows and replaces them with the
    # fresh attempt, keeping the published CSV one row per configuration.
    rows = [r for r in rows if r['status'] not in ('TIMEOUT', 'ERROR')]
    done = {(r['benchmark'], r['parameters'], r['configuration']) for r in rows}
    app_envs = {}
    for benchmark, params in matrix(args.pilot):
        if args.benchmarks and benchmark not in args.benchmarks:
            continue
        parameters = json.dumps(params, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(parameters.encode()).hexdigest()[:10]
        for config in CONFIGURATIONS:
            if config not in args.configurations and config != 'ITS':
                continue
            if (benchmark, parameters, config) in done:
                continue
            row = dict.fromkeys(FIELDS, '')
            row.update(benchmark=benchmark, parameters=parameters, threads=args.threads,
                       warps=8, cores=1, backend=args.backend, configuration=config)
            if config == 'ITS':
                rows.append(row)
                write_csv(csv_path, rows)
                continue
            app = benchmark if 'args' in params else 'threadsplit'
            if app not in app_envs:
                app_envs[app] = app_environment(build, args.backend, args.threads, app)
            env = app_envs[app].copy()
            env['VORTEX_SCS_MODE'] = 'scs' if config == 'SCS' else 'ipdom'
            env['VORTEX_EVAL_METRICS'] = '1'
            env['POCL_CACHE_DIR'] = str(build / f'pocl-{config}')
            env['POCL_VORTEX_CFLAGS'] += f' -mllvm -vortex-scs-yield={int(config == "SCS")}'
            if 'args' in params:
                app_args = shlex.split(params['args'])
            else:
                app_args = ['--benchmark', benchmark, '--software', str(int(config == 'IPDOM_SW'))]
                for key, value in params.items():
                    app_args += [f'--{key}', str(value)]
            log = args.output / f'{args.backend}-t{args.threads}-{benchmark}-{digest}-{config}.log'
            started = time.monotonic()
            with log.open('w') as out:
                process = subprocess.Popen([f'./{app}'] + app_args, cwd=build / 'tests/opencl' / app,
                                           env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    limit = args.native_timeout if config == 'IPDOM_native' and 'args' not in params else args.timeout
                    rc = process.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    rc = None
            row['wall_seconds'] = round(time.monotonic() - started, 3)
            row['log'] = log.name
            text = log.read_text(errors='replace')
            row['status'] = 'TIMEOUT' if rc is None else 'PASS' if rc == 0 and 'PASSED' in text.upper() else 'ERROR'
            if rc is None and config == 'IPDOM_native' and 'args' not in params and 'WORKLOAD:' in text:
                row['status'] = 'DEADLOCK'
                row['notes'] = ('Natural IPDOM has a same-warp holder/waiter dependency at the polling-loop '
                                'reconvergence; observed nontermination after ' + str(args.native_timeout) + 's. '
                                'See README for the per-kernel dependency argument.')
            perf = re.findall(r'PERF: instrs=(\d+), cycles=(\d+)', text)
            metrics = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
                       for line in text.splitlines() if line.startswith('EVAL:')]
            if row['status'] == 'PASS' and perf and metrics:
                row['dynamic_instructions'] = sum(int(p[0]) for p in perf)
                row['cycles'] = sum(int(p[1]) for p in perf)
                for source, target in [('issued', 'issued_instructions'), ('active_lanes', 'active_lane_instructions'),
                                       ('yields', 'vx_yield_count'), ('switches', 'subgroup_switch_count'),
                                       ('watchdog_switches', 'watchdog_switch_count')]:
                    row[target] = sum(m[source] for m in metrics)
                row['peak_live_split_contexts'] = max(m['peak_contexts'] for m in metrics)
                row['simd_utilization'] = row['active_lane_instructions'] / (args.threads * row['issued_instructions'])
            elif row['status'] == 'PASS':
                row['status'] = 'ERROR'
                row['notes'] = 'Output passed, but required performance counters are missing.'
            rows.append(row)
            write_csv(csv_path, rows)
            print(benchmark, args.threads, config, row['status'], row['cycles'], row['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
