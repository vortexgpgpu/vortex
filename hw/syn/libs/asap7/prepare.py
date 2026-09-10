#!/usr/bin/env python3

import hashlib
import gzip
import json
import pathlib
import re
import argparse


GROUPS = ('INVBUF', 'SIMPLE', 'AO', 'OA', 'SEQ')
CORNERS = ('TT', 'SS', 'FF')


def read_manifest(manifest):
    metadata = {}
    files = []
    for line in manifest.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'):
            continue
        if fields[0] in ('repository', 'commit', 'source_repository', 'source_commit'):
            metadata[fields[0]] = fields[1]
        elif re.fullmatch(r'[0-9a-f]{64}', fields[0]) and len(fields) == 2:
            files.append((fields[0], fields[1]))
    return metadata, files


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(downloads, files):
    for expected, name in files:
        path = downloads / name
        if not path.is_file():
            raise SystemExit(f'ERROR: missing {path}; rerun install.sh')
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f'ERROR: checksum mismatch for {path}')


def read_text(path):
    with path.open('rb') as stream:
        compressed = stream.read(2) == b'\x1f\x8b'
    if compressed:
        with gzip.open(path, 'rt') as stream:
            return stream.read()
    return path.read_text()


def library_parts(text, path):
    match = re.search(r'\blibrary\s*\([^)]*\)\s*\{', text)
    if not match:
        raise ValueError(f'{path}: library declaration not found')
    start = match.end() - 1
    depth = 0
    in_string = False
    escaped = False
    line_comment = False
    block_comment = False
    end = None
    index = start
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ''
        if line_comment:
            line_comment = char != '\n'
        elif block_comment:
            if char == '*' and next_char == '/':
                block_comment = False
                index += 1
        elif in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '/' and next_char == '/':
            line_comment = True
            index += 1
        elif char == '/' and next_char == '*':
            block_comment = True
            index += 1
        elif char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end = index
                break
        index += 1
    if end is None:
        raise ValueError(f'{path}: unbalanced library braces')
    return text[start + 1:end].strip()


def select_file(paths, group, flavor, corner):
    matches = [path for path in paths
               if f'_{group}_{flavor}_{corner}_' in path.name]
    if len(matches) != 1:
        raise SystemExit(f'ERROR: expected one {group}/{corner} file, found {matches}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=pathlib.Path, required=True)
    parser.add_argument('--downloads', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--vt', choices=('lvt', 'rvt'), required=True)
    parser.add_argument('--models', action='store_true')
    args = parser.parse_args()

    flavor = args.vt.upper()
    metadata, all_files = read_manifest(args.manifest)
    files = [(checksum, name) for checksum, name in all_files
             if f'_{flavor}_' in pathlib.PurePosixPath(name).name
             and (args.models or not name.endswith('.v'))]
    verify(args.downloads, files)
    libraries = [args.downloads / name for _, name in files
                 if name.endswith('.lib') or name.endswith('.lib.gz')]
    models = [args.downloads / name for _, name in files if name.endswith('.v')]

    lib_dir = args.output / 'lib'
    verilog_dir = args.output / 'verilog'
    metadata_dir = args.output / 'metadata'
    for directory in (lib_dir, verilog_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    outputs = []
    for corner in CORNERS:
        bodies = []
        cell_count = 0
        for group in GROUPS:
            source = select_file(libraries, group, flavor, corner)
            body = library_parts(read_text(source), source)
            cells = len(re.findall(r'(?m)^\s*cell\s*\(', body))
            if cells == 0:
                raise SystemExit(f'ERROR: no cells found in {source}')
            cell_count += cells
            bodies.append(f'/* {group} */\n{body}')
        output = lib_dir / f'asap7_{args.vt}_{corner.lower()}.lib'
        output.write_text(f'library (asap7_{args.vt}_{corner.lower()}) {{\n' +
                          '\n'.join(bodies) + '\n}\n')
        print(f'Prepared {output} ({cell_count} cells)')
        outputs.append(output)

    if args.models:
        model_parts = []
        module_count = 0
        for group in GROUPS:
            source = select_file(models, group, flavor, 'TT')
            text = source.read_text()
            modules = len(re.findall(r'(?m)^\s*module\s+', text))
            if modules == 0:
                raise SystemExit(f'ERROR: no modules found in {source}')
            module_count += modules
            model_parts.append(f'// {group}: {source.name}\n{text.rstrip()}')
        models_output = verilog_dir / f'asap7_{args.vt}_cells.v'
        models_output.write_text('\n\n'.join(model_parts) + '\n')
        print(f'Prepared {models_output} ({module_count} modules)')
        outputs.append(models_output)

    versions = {
        'upstream': metadata,
        'groups': list(GROUPS),
        'corners': list(CORNERS),
        'flavor': flavor,
        'inputs': {name: checksum for checksum, name in files},
        'outputs': {str(path.relative_to(args.output)): sha256(path) for path in outputs},
    }
    versions_output = metadata_dir / f'asap7_{args.vt}.json'
    versions_output.write_text(json.dumps(versions, indent=2, sort_keys=True) + '\n')
    print(f'Prepared {versions_output}')


if __name__ == '__main__':
    main()
