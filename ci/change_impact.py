#!/usr/bin/env python3
# Copyright © 2019-2026
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Vortex change-impact classifier (advisory).

Classifies a diff into R0..R3 from the rules in docs/review/change-impact.yaml
and the normative policy in docs/review/code-review-policy.md. Paths give at
most R2; only an externally visible change found in the diff content (added,
changed or removed) gives R3. Evidence checks report what a reviewer must
confirm and never give R3.

  change_impact.py classify --base REF --head REF [--event E] [--format text|json|markdown]
  change_impact.py explain  --base REF --head REF | --path P [P ...]
  change_impact.py replay   --range A..B [--format text|json|tsv]
  change_impact.py lint-policy

Python 3 standard library only, so it runs before any dependency is installed.
"""

import argparse
import fnmatch
import functools
import json
import os
import re
import subprocess
import sys
import zlib


def _source_root():
    """The source tree. configure copies ci/ into a build tree, whose config.mk
    records the sources' location (VORTEX_HOME)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(here, "config.mk")) as fh:
            for line in fh:
                m = re.match(r"\s*VORTEX_HOME\s*[?:]?=\s*(\S+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return here


ROOT = _source_root()
DEFAULT_POLICY = os.path.join(ROOT, "docs", "review", "change-impact.yaml")
CLASSES = ("R0", "R1", "R2", "R3")
PATH_RULE_CEILING = "R2"
EVIDENCE_CEILING = "R2"
EVENTS = ("push", "pull_request", "replay")
ZERO_SHA = re.compile(r"^0+$")
MAX_BLOB = 4 << 20


def rank(cls):
    return CLASSES.index(cls)


def max_class(classes, floor="R0"):
    best = floor
    for c in classes:
        if rank(c) > rank(best):
            best = c
    return best


# --------------------------------------------------------------------------- #
# Policy: a YAML subset
# --------------------------------------------------------------------------- #

class PolicyError(Exception):
    pass


_KEY = re.compile(r"^([A-Za-z_][\w.-]*):(?:\s+(.*))?$")
_INT = re.compile(r"^[-+]?(0x[0-9a-fA-F]+|\d+)$")
_FLOAT = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")


def _strip_yaml_comment(line):
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _split_flow(text, lineno):
    items, cur, quote = [], "", None
    for ch in text:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            cur += ch
        elif ch == ",":
            items.append(cur.strip())
            cur = ""
        elif ch in "[]{}":
            raise PolicyError("line {}: nested flow collections are not supported".format(lineno))
        else:
            cur += ch
    if quote:
        raise PolicyError("line {}: unterminated quote".format(lineno))
    if cur.strip():
        items.append(cur.strip())
    elif items:
        raise PolicyError("line {}: empty list item".format(lineno))
    return items


def _scalar(text, lineno):
    if text.startswith("["):
        if not text.endswith("]"):
            raise PolicyError("line {}: unterminated flow list".format(lineno))
        return [_scalar(t, lineno) for t in _split_flow(text[1:-1], lineno)]
    if text[0] in "{|>&*!":
        raise PolicyError("line {}: unsupported YAML construct {!r}".format(lineno, text[0]))
    if text.startswith('"'):
        if len(text) < 2 or not text.endswith('"'):
            raise PolicyError("line {}: unterminated string".format(lineno))
        try:
            return json.loads(text)
        except ValueError as exc:
            raise PolicyError("line {}: bad string: {}".format(lineno, exc))
    if text.startswith("'"):
        if len(text) < 2 or not text.endswith("'"):
            raise PolicyError("line {}: unterminated string".format(lineno))
        return text[1:-1].replace("''", "'")
    if text in ("true", "false"):
        return text == "true"
    if text in ("null", "~"):
        return None
    if _INT.match(text):
        return int(text, 0)
    if _FLOAT.match(text):
        return float(text)
    return text


def _is_seq_item(text):
    return text == "-" or text.startswith("- ")


def _parse_block(lines, pos, indent):
    if _is_seq_item(lines[pos][1]):
        return _parse_seq(lines, pos, indent)
    return _parse_map(lines, pos, indent)


def _parse_map(lines, pos, indent):
    out = {}
    while pos < len(lines):
        ind, text, n = lines[pos]
        if ind < indent:
            break
        if ind > indent:
            raise PolicyError("line {}: unexpected indent".format(n))
        if _is_seq_item(text):
            raise PolicyError("line {}: list item where a key was expected".format(n))
        m = _KEY.match(text)
        if not m:
            raise PolicyError("line {}: expected 'key: value'".format(n))
        key, rest = m.group(1), m.group(2)
        if key in out:
            raise PolicyError("line {}: duplicate key {!r}".format(n, key))
        pos += 1
        if rest:
            out[key] = _scalar(rest, n)
        elif pos < len(lines) and lines[pos][0] > ind:
            out[key], pos = _parse_block(lines, pos, lines[pos][0])
        elif pos < len(lines) and lines[pos][0] == ind and _is_seq_item(lines[pos][1]):
            out[key], pos = _parse_seq(lines, pos, ind)
        else:
            out[key] = None
    return out, pos


def _parse_seq(lines, pos, indent):
    out = []
    while pos < len(lines):
        ind, text, n = lines[pos]
        if ind < indent:
            break
        if ind > indent:
            raise PolicyError("line {}: unexpected indent".format(n))
        if not _is_seq_item(text):
            break
        item = text[1:].strip()
        if not item:
            pos += 1
            if pos >= len(lines) or lines[pos][0] <= ind:
                out.append(None)
                continue
            val, pos = _parse_block(lines, pos, lines[pos][0])
        elif _KEY.match(item) and not item.startswith(("'", '"')):
            # "- key: v" opens a mapping whose keys sit two columns in.
            lines[pos] = (ind + 2, item, n)
            val, pos = _parse_map(lines, pos, ind + 2)
        else:
            val = _scalar(item, n)
            pos += 1
        out.append(val)
    return out, pos


def load_yaml_subset(text):
    lines = []
    for n, raw in enumerate(text.splitlines(), 1):
        body = _strip_yaml_comment(raw).rstrip()
        if not body.strip():
            continue
        lead = body[:len(body) - len(body.lstrip())]
        if "\t" in lead:
            raise PolicyError("line {}: tab in indentation".format(n))
        lines.append((len(lead), body.strip(), n))
    if not lines:
        return {}
    value, pos = _parse_block(lines, 0, lines[0][0])
    if pos != len(lines):
        raise PolicyError("line {}: content after the document".format(lines[pos][2]))
    return value


DETECTOR_KINDS = {}
EVIDENCE_KINDS = {}


def lint_policy(policy):
    """Schema and invariant errors in a loaded policy (empty list = valid)."""
    errors = []
    if not isinstance(policy, dict):
        return ["policy must be a mapping"]
    if policy.get("schema") != 1:
        errors.append("schema must be 1")
    if not isinstance(policy.get("policy_version"), int):
        errors.append("policy_version must be an integer")
    classes = policy.get("classes") or {}
    if sorted(classes) != list(CLASSES):
        errors.append("classes must define exactly {}".format(", ".join(CLASSES)))
    for key in ("default_class", "cosmetic_class"):
        if policy.get(key) not in CLASSES:
            errors.append("{} must be one of {}".format(key, ", ".join(CLASSES)))
    ind = policy.get("indeterminate") or {}
    for event in EVENTS:
        if ind.get(event) not in CLASSES:
            errors.append("indeterminate.{} must be a class".format(event))
    seen = set()

    def check_entry(section, entry, kinds):
        if not isinstance(entry, dict):
            errors.append("{}: every entry must be a mapping".format(section))
            return None
        eid = entry.get("id")
        if not isinstance(eid, str) or not eid:
            errors.append("{}: entry without an id".format(section))
        elif eid in seen:
            errors.append("{}: duplicate id {!r}".format(section, eid))
        seen.add(eid)
        keys = ("rtl", "simx") if entry.get("kind") == "lockstep" else ("paths",)
        for key in keys:
            paths = entry.get(key)
            if not isinstance(paths, list) or not paths or \
                    not all(isinstance(p, str) and p for p in paths):
                errors.append("{} {}: {} must be a non-empty list of globs".format(section, eid, key))
        if kinds is not None and entry.get("kind") not in kinds:
            errors.append("{} {}: unknown kind {!r}".format(section, eid, entry.get("kind")))
        return eid

    for entry in policy.get("path_rules") or []:
        eid = check_entry("path_rules", entry, None)
        if isinstance(entry, dict):
            cls = entry.get("class")
            if cls not in CLASSES:
                errors.append("path_rules {}: class must be a class".format(eid))
            elif rank(cls) > rank(PATH_RULE_CEILING):
                errors.append("path_rules {}: a path rule may not exceed {}; R3 comes "
                              "from a detector".format(eid, PATH_RULE_CEILING))
    if not policy.get("path_rules"):
        errors.append("path_rules must not be empty")
    for entry in policy.get("detectors") or []:
        eid = check_entry("detectors", entry, DETECTOR_KINDS)
        if isinstance(entry, dict):
            for key, kind in (("names", "sv_names"), ("patterns", "decoder_ops"),
                              ("keys", "version_pins")):
                if entry.get("kind") == kind and not isinstance(entry.get(key), list):
                    errors.append("detectors {}: {} needs a {} list".format(eid, kind, key))
            if entry.get("kind") == "make_surface" and not isinstance(entry.get("groups"), dict):
                errors.append("detectors {}: make_surface needs a groups mapping".format(eid))
            for rx in (entry.get("patterns") or []) + list((entry.get("groups") or {}).values()):
                try:
                    re.compile(rx)
                except (re.error, TypeError) as exc:
                    errors.append("detectors {}: bad pattern {!r}: {}".format(eid, rx, exc))
            if entry.get("class") not in CLASSES:
                errors.append("detectors {}: class must be a class".format(eid))
            if not entry.get("why"):
                errors.append("detectors {}: needs a why".format(eid))
    for entry in policy.get("evidence") or []:
        eid = check_entry("evidence", entry, EVIDENCE_KINDS)
        if isinstance(entry, dict):
            cls = entry.get("min_class")
            if cls not in CLASSES:
                errors.append("evidence {}: min_class must be a class".format(eid))
            elif rank(cls) > rank(EVIDENCE_CEILING):
                errors.append("evidence {}: min_class may not exceed {}".format(
                    eid, EVIDENCE_CEILING))
            if not entry.get("why"):
                errors.append("evidence {}: needs a why".format(eid))
    return errors


def load_policy(path):
    with open(path, encoding="utf-8") as fh:
        policy = load_yaml_subset(fh.read())
    errors = lint_policy(policy)
    if errors:
        raise PolicyError("; ".join(errors))
    return policy


# --------------------------------------------------------------------------- #
# Path globs: `*` stays within a path component, `**` spans components.
# --------------------------------------------------------------------------- #

_GLOB_CACHE = {}


def _glob_re(pattern):
    rx = _GLOB_CACHE.get(pattern)
    if rx is None:
        out, i = "", 0
        while i < len(pattern):
            if pattern.startswith("**/", i):
                out += "(?:.*/)?"
                i += 3
            elif pattern.startswith("**", i):
                out += ".*"
                i += 2
            elif pattern[i] == "*":
                out += "[^/]*"
                i += 1
            elif pattern[i] == "?":
                out += "[^/]"
                i += 1
            else:
                out += re.escape(pattern[i])
                i += 1
        rx = _GLOB_CACHE[pattern] = re.compile(out + r"\Z")
    return rx


def path_matches(path, patterns):
    return any(_glob_re(p).match(path) for p in patterns)


# --------------------------------------------------------------------------- #
# Source normalization: decide whether a diff is cosmetic, and what it changed
# --------------------------------------------------------------------------- #

_C_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".cu",
          ".cl", ".js", ".java", ".scala", ".glsl", ".comp", ".vert", ".frag"}
_SV_EXT = {".sv", ".svh", ".v", ".vh", ".vlt"}
_HASH_EXT = {".sh", ".bash", ".mk", ".cmake", ".tcl", ".xdc", ".sdc", ".toml",
             ".pl", ".rb", ".cfg", ".ini", ".txt.in"}
_INDENT_EXT = {".py"}
_YAML_EXT = {".yaml", ".yml"}
_HASH_NAMES = {"Makefile", "configure", "Dockerfile", "VERSION", "CMakeLists.txt"}

# Every alternative that starts also matches, through the end of the line or
# file when it is unterminated, so a scan never revisits text: linear on any input.
_DQ = r'"(?:\\.|[^"\\\n])*"?'
_SQ = r"'(?:\\.|[^'\\\n])*'?"
_C_TOKENS = re.compile(_DQ + "|" + _SQ + r"|//[^\n]*|/\*(?:.*?\*/|.*)", re.S)
_SV_TOKENS = re.compile(_DQ + r"|//[^\n]*|/\*(?:.*?\*/|.*)", re.S)
_WORDS = re.compile(r"\w+|[^\w\s]")
_SPACES = {lang: re.compile("|".join(lits + [r"[ \t\r\f\v]+"]))
           for lang, lits in (("c", [_DQ, _SQ]), ("sv", [_DQ]), ("hash", [_DQ, _SQ]),
                              ("make", [_DQ, _SQ]), ("indent", [_DQ, _SQ]), ("yaml", [_DQ, _SQ]),
                              ("text", [_DQ]))}
_CPP_LINE = r"^[ \t]*#(?:\\[\s\S]|[^\n\\])*"


def language(path):
    name = os.path.basename(path)
    if name.endswith(".in"):
        name = name[:-3]
    ext = os.path.splitext(name)[1].lower()
    if name.startswith("Makefile") or ext == ".mk":
        return "make"
    if name in _HASH_NAMES:
        return "hash"
    if ext in _SV_EXT:
        return "sv"
    if ext in _C_EXT:
        return "c"
    if ext in _INDENT_EXT:
        return "indent"
    if ext in _YAML_EXT:
        return "yaml"
    if ext in _HASH_EXT:
        return "hash"
    return "text"


def _comment(tok):
    return tok.startswith("//") or (tok.startswith("/*") and len(tok) >= 4 and tok.endswith("*/"))


def _strip_c_like(text, rx):
    return rx.sub(lambda m: " " if _comment(m.group(0)) else m.group(0), text)


def _strip_hash(text):
    return "\n".join(_strip_yaml_comment(line) for line in text.split("\n"))


_BLOCK_SCALAR = re.compile(r"(?:^|\s)[|>][-+0-9]*$")


def _strip_yaml(text):
    """YAML comments removed; the lines of a block scalar are content and stay whole."""
    out, block = [], None
    for line in text.split("\n"):
        stripped = line.lstrip()
        ind = len(line) - len(stripped)
        if block is not None and (not stripped or ind > block):
            out.append(line)
            continue
        body = _strip_yaml_comment(line).rstrip()
        block = ind + re.match(r"(?:-\s+)*", stripped).end() if _BLOCK_SCALAR.search(body) else None
        out.append(body)
    return "\n".join(out)


def strip_comments(text, lang):
    if lang == "sv":
        return _strip_c_like(text, _SV_TOKENS)
    if lang == "c":
        return _strip_c_like(text, _C_TOKENS)
    if lang in ("hash", "make", "indent"):
        return _strip_hash(text)
    if lang == "yaml":
        return _strip_yaml(text)
    return text


def normalize(text, lang):
    """A form equal across comment and whitespace edits, and only those: blank
    lines go, and each run of spaces outside a string literal becomes one.
    Indentation is kept where it is syntax, a recipe's leading tab included."""
    spaces = _SPACES[lang]
    lines = []
    for line in strip_comments(text, lang).split("\n"):
        body = line.strip()
        if not body:
            continue
        lead = ""
        if lang in ("indent", "yaml"):
            lead = line[:len(line) - len(line.lstrip())]
        elif lang == "make" and line[0] == "\t":
            lead = "\t"
        lines.append(lead + spaces.sub(lambda m: " " if m.group(0)[0].isspace() else m.group(0), body))
    return tuple(lines)


def drop_calls(text, names, backtick, statement=True):
    """Remove macro or function calls by name, with their balanced arguments."""
    if not names:
        return text
    prefix = "`" if backtick else r"\b"
    rx = re.compile(prefix + r"(?:" + "|".join(re.escape(n) for n in names) + r")\s*\(")
    out, pos = [], 0
    while True:
        m = rx.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos:m.start()])
        i, depth, quote = m.end(), 1, False
        while i < len(text) and depth:
            ch = text[i]
            if quote:
                if ch == "\\":
                    i += 1
                elif ch == '"':
                    quote = False
            elif ch == '"':
                quote = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if statement:
            while i < len(text) and text[i] in " \t":
                i += 1
            if i < len(text) and text[i] == ";":
                i += 1
        out.append(" ")
        pos = i
    return "".join(out)


# --------------------------------------------------------------------------- #
# Git access
# --------------------------------------------------------------------------- #

class GitError(Exception):
    pass


class Change:
    __slots__ = ("status", "path", "old_path", "old_oid", "new_oid")

    def __init__(self, status, path, old_path=None, old_oid=None, new_oid=None):
        self.status = status
        self.path = path
        self.old_path = old_path or path
        self.old_oid = old_oid
        self.new_oid = new_oid

    def paths(self):
        return {self.path, self.old_path}

    def as_dict(self):
        d = {"status": self.status, "path": self.path}
        if self.old_path != self.path:
            d["old_path"] = self.old_path
        return d


class Repo:
    def __init__(self, path):
        self.path = path
        self._batch = None

    def git(self, *args, check=True):
        proc = subprocess.run(["git", "-C", self.path] + list(args),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check and proc.returncode != 0:
            raise GitError("git {}: {}".format(" ".join(args),
                                               proc.stderr.decode(errors="replace").strip()))
        return proc.stdout.decode("utf-8", errors="replace") if proc.returncode == 0 else None

    def commit(self, rev):
        if not rev or ZERO_SHA.match(rev):
            return None
        out = self.git("rev-parse", "--verify", "--quiet", rev + "^{commit}", check=False)
        return out.strip() if out else None

    def merge_base(self, a, b):
        out = self.git("merge-base", a, b, check=False)
        return out.strip() if out else None

    def parents(self, rev):
        out = self.git("rev-list", "--parents", "-n", "1", rev).split()
        return out[1:]

    def is_shallow_root(self, rev):
        shallow = os.path.join(self.git("rev-parse", "--git-common-dir").strip(), "shallow")
        if not os.path.isabs(shallow):
            shallow = os.path.join(self.path, shallow)
        try:
            with open(shallow) as fh:
                return rev in fh.read().split()
        except OSError:
            return False

    def changes(self, base, head):
        """The changed files, each with the object ids its contents are read by,
        so no path ever travels over a newline-delimited protocol."""
        out = self.git("diff", "--raw", "-z", "-M", "--no-abbrev", "--no-ext-diff", base, head)
        toks = out.split("\0")
        changes, i = [], 0
        while i < len(toks) and toks[i]:
            _, _, old_oid, new_oid, status = toks[i][1:].split(" ")
            if status[0] in "RC":
                changes.append(Change(status[0], toks[i + 2], toks[i + 1], old_oid, new_oid))
                i += 3
            else:
                changes.append(Change(status[0], toks[i + 1], None, old_oid, new_oid))
                i += 2
        return changes

    def patch(self, base, head, path):
        """The textual diff of one file, whatever .gitattributes says about it."""
        return self.git("--literal-pathspecs", "diff", "-U3", "--text", "--no-ext-diff", "--no-textconv",
                        base, head, "--", path) or ""

    def messages(self, base, head):
        return self.git("log", "--format=%B", "{}..{}".format(base, head), check=False) or ""

    def blob(self, oid):
        """Text of a blob, or None if absent, binary, a submodule or too large."""
        if not oid or ZERO_SHA.match(oid):
            return None
        if self._batch is None:
            self._batch = subprocess.Popen(["git", "-C", self.path, "cat-file", "--batch"],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._batch.stdin.write((oid + "\n").encode())
        self._batch.stdin.flush()
        header = self._batch.stdout.readline().decode().split()
        if len(header) != 3:
            return None
        size = int(header[2])
        data = self._batch.stdout.read(size + 1)[:size]
        if header[1] != "blob" or size > MAX_BLOB or b"\0" in data[:8192]:
            return None
        return data.decode("utf-8", errors="replace")

    def close(self):
        if self._batch:
            self._batch.stdin.close()
            self._batch.wait()
            self._batch = None


class Diff:
    """The changed files between two commits, with cached contents."""

    def __init__(self, repo, base, head, changes, messages=""):
        self.repo = repo
        self.base = base
        self.head = head
        self.changes = changes
        self.messages = messages
        self.check_remote = False
        self._old, self._new, self._cosmetic = {}, {}, {}

    def old(self, change):
        if change.status == "A":
            return None
        if change.old_path not in self._old:
            self._old[change.old_path] = self.repo.blob(change.old_oid)
        return self._old[change.old_path]

    def new(self, change):
        if change.status == "D":
            return None
        if change.path not in self._new:
            self._new[change.path] = self.repo.blob(change.new_oid)
        return self._new[change.path]

    def cosmetic(self, change):
        """True when a modified file differs only in comments or whitespace."""
        if change.status != "M":
            return False
        if change.path not in self._cosmetic:
            old, new = self.old(change), self.new(change)
            lang = language(change.path)
            self._cosmetic[change.path] = (old is not None and new is not None
                                           and normalize(old, lang) == normalize(new, lang))
        return self._cosmetic[change.path]

    def matching(self, patterns, exclude=()):
        return [c for c in self.changes
                if any(path_matches(p, patterns) and not path_matches(p, exclude)
                       for p in c.paths())]


# --------------------------------------------------------------------------- #
# Contract detectors. Each returns a list of {"path", "detail"} findings.
# --------------------------------------------------------------------------- #

def detector(kind):
    def register(fn):
        DETECTOR_KINDS[kind] = fn
        return fn
    return register


def evidence_check(kind):
    def register(fn):
        EVIDENCE_KINDS[kind] = fn
        return fn
    return register


def _ws(text):
    return re.sub(r"\s+", " ", text).strip()


@functools.lru_cache(maxsize=8)
def _brace_pairs(text):
    pairs, opened = {}, []
    for m in re.finditer(r"[{}]", text):
        if m.group(0) == "{":
            opened.append(m.start())
        elif opened:
            pairs[opened.pop()] = m.end()
    return pairs


def _match_brace(text, start):
    """Index just past the brace that closes the one at text[start]."""
    return _brace_pairs(text).get(start, len(text))


def _match_paren(text, start):
    """Index just past the parenthesis that closes the one at text[start]."""
    depth = 0
    for m in re.compile(r"[()]").finditer(text, start):
        depth += 1 if m.group(0) == "(" else -1
        if depth == 0:
            return m.end()
    return len(text)


def _split_top(text):
    """Comma-separated items, not splitting inside brackets."""
    items, depth, start = [], 0, 0
    for m in re.finditer(r"[()\[\]{},]", text):
        ch = m.group(0)
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            items.append(text[start:m.start()])
            start = m.end()
    items.append(text[start:])
    return items


def _drop_directives(text):
    return re.sub(r"^\s*`(?:ifdef|ifndef|else|elsif|endif|include|define|undef)\b[^\n]*", " ",
                  text, flags=re.M)


def parse_sv_interface(text):
    """(params, signals, modports) of a SystemVerilog interface."""
    body = _drop_directives(strip_comments(text, "sv"))
    header, rest = "", body
    m = re.search(r"\binterface\s+(?:(?:static|automatic)\s+)?\w+\s*", body)
    if m:
        pos = m.end()
        im = re.compile(r"import\s[^;]*;\s*").match(body, pos)
        while im:
            pos = im.end()
            im = re.compile(r"import\s[^;]*;\s*").match(body, pos)
        pm = re.compile(r"#\s*\(").match(body, pos)
        if pm:
            pos = _match_paren(body, pm.end() - 1)
            header = body[pm.end() - 1:pos]
        lm = re.compile(r"\s*\(").match(body, pos)
        if lm:
            pos = _match_paren(body, lm.end() - 1)
        semi = body.find(";", pos)
        rest = body[semi + 1:] if semi != -1 else body[pos:]
    params = {}
    for item in _split_top(header[1:-1]):
        pm = re.match(r"[^=]*?(\w+)\s*(?:\[[^\]]*\]\s*)*=(.*)$", item, re.S)
        if pm:
            params[pm.group(1)] = _ws(pm.group(2))
    modports = {}
    for mm in re.finditer(r"\bmodport\s+(\w+)\s*\((.*?)\)\s*;", rest, re.S):
        ports, direction = set(), ""
        for item in mm.group(2).split(","):
            words = item.split()
            if not words:
                continue
            if words[0] in ("input", "output", "inout", "ref"):
                direction, words = words[0], words[1:]
            elif words[0] == "import":
                direction, words = "import", words[1:]
            if words:
                ports.add("{} {}".format(direction, words[-1]))
        modports[mm.group(1)] = ports
    decls = re.sub(r"\bmodport\s+\w+\s*\(.*?\)\s*;", " ", rest, flags=re.S)
    decls = re.sub(r"\b(function|task)\b.*?\bend\1\b", " ", decls, flags=re.S)
    signals = {}
    for stmt in decls.split(";"):
        stmt = _ws(stmt)
        if not stmt or stmt.startswith(("endinterface", "import ", "assign ", "localparam ", "genvar")):
            continue
        sm = re.match(r"^((?:\w+(?:::\w+)?)(?:\s*\[[^\]]*\])*)\s+(.+)$", stmt)
        if not sm:
            continue
        for name in sm.group(2).split(","):
            nm = re.match(r"\s*(\w+)", name)
            if nm:
                signals[nm.group(1)] = _ws(sm.group(1) + " " + name)
    return params, signals, modports


def _dict_delta(old, new, label):
    out = []
    for k in sorted(set(old) - set(new)):
        out.append("-{} {}".format(label, k))
    for k in sorted(set(new) - set(old)):
        out.append("+{} {}".format(label, k))
    for k in sorted(set(old) & set(new)):
        if old[k] != new[k]:
            out.append("~{} {}".format(label, k))
    return out


def _listing(label, names, limit=12):
    names = sorted(names)
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += " (+{} more)".format(len(names) - limit)
    return "{}: {}".format(label, shown)


def _delta_parts(old, new, changed_label="changed"):
    """added/changed/removed listings between two name -> value maps."""
    parts = []
    added = set(new) - set(old)
    changed = {k for k in old if k in new and old[k] != new[k]}
    removed = set(old) - set(new)
    if added:
        parts.append(_listing("added", added))
    if changed:
        parts.append(_listing(changed_label, changed))
    if removed:
        parts.append(_listing("removed", removed))
    return parts


@detector("sv_interface")
def detect_sv_interface(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        if diff.cosmetic(c):
            continue
        old, new = diff.old(c), diff.new(c)
        if new is None:
            found.append({"path": c.old_path, "detail": "interface removed"})
            continue
        pn, sn, mn = parse_sv_interface(new)
        if old is None:
            found.append({"path": c.path, "detail": "interface added: {} signals; modports {}".format(
                len(sn), ", ".join(sorted(mn)) or "none")})
            continue
        po, so, mo = parse_sv_interface(old)
        delta = _dict_delta(po, pn, "param") + _dict_delta(so, sn, "signal")
        for name in sorted(set(mo) | set(mn)):
            a, b = mo.get(name), mn.get(name)
            if a is None or b is None:
                delta.append("{}modport {}".format("+" if a is None else "-", name))
            else:
                delta += ["modport {}: -{}".format(name, p) for p in sorted(a - b)]
                delta += ["modport {}: +{}".format(name, p) for p in sorted(b - a)]
        if delta:
            found.append({"path": c.path, "detail": "; ".join(delta)})
    return found


def parse_sv_typedefs(text):
    """name -> normalized body of every typedef in a package."""
    body = strip_comments(text, "sv")
    out = {}
    for m in re.finditer(r"\btypedef\b", body):
        i = m.end()
        semi = body.find(";", i)
        brace = body.find("{", i)
        if brace != -1 and (semi == -1 or brace < semi):
            end = _match_brace(body, brace)
            nm = re.match(r"\s*(\w+)\s*(?:\[[^\]]*\]\s*)*;", body[end:])
            if nm:
                out[nm.group(1)] = " ".join(_WORDS.findall(body[i:end]))
        elif semi != -1:
            decl = body[i:semi]
            nm = re.search(r"(\w+)\s*(?:\[[^\]]*\]\s*)*$", decl)
            if nm:
                out[nm.group(1)] = " ".join(_WORDS.findall(decl))
    return out


@detector("sv_typedefs")
def detect_sv_typedefs(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        if diff.cosmetic(c):
            continue
        parts = _delta_parts(parse_sv_typedefs(diff.old(c) or ""),
                             parse_sv_typedefs(diff.new(c) or ""))
        if parts:
            found.append({"path": c.path, "detail": "typedefs " + "; ".join(parts)})
    return found


def parse_sv_names(text, names):
    """name -> normalized value of each localparam or `define whose name matches a glob."""
    body = strip_comments(text, "sv")
    out = {}
    for m in re.finditer(r"\blocalparam\s+(?:[\w:]+\s+)?(?:\[[^\]]*\]\s*)*(\w+)\s*=\s*([^;]+);", body):
        out[m.group(1)] = _ws(m.group(2))
    for m in re.finditer(r"^[ \t]*`define[ \t]+(\w+)((?:\\\n|[^\n])*)", body, flags=re.M):
        out[m.group(1)] = _ws(m.group(2).replace("\\\n", " "))
    return {k: v for k, v in out.items() if any(fnmatch.fnmatchcase(k, n) for n in names)}


@detector("sv_names")
def detect_sv_names(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        if diff.cosmetic(c):
            continue
        parts = _delta_parts(parse_sv_names(diff.old(c) or "", rule["names"]),
                             parse_sv_names(diff.new(c) or "", rule["names"]))
        if parts:
            found.append({"path": c.path, "detail": "encodings " + "; ".join(parts)})
    return found


def decoder_tokens(text, path, patterns):
    """The set of operation names a decoder's code (not its comments) mentions."""
    body = strip_comments(text, language(path))
    out = set()
    for pattern in patterns:
        for m in re.finditer(pattern, body):
            out.add(m.group(1) if m.groups() else m.group(0))
    return out


@detector("decoder_ops")
def detect_decoder_ops(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        if diff.cosmetic(c):
            continue
        old = decoder_tokens(diff.old(c) or "", c.old_path, rule["patterns"])
        new = decoder_tokens(diff.new(c) or "", c.path, rule["patterns"])
        parts = []
        if new - old:
            parts.append(_listing("decodes new", new - old))
        if old - new:
            parts.append(_listing("no longer decodes", old - new))
        if parts:
            found.append({"path": c.path, "detail": "; ".join(parts)})
    return found


def parse_toml_keys(text):
    out, section = {}, ""
    for line in _strip_hash(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        sm = re.match(r"^\[\s*([^\]]+?)\s*\]$", line)
        if sm:
            section = sm.group(1)
            continue
        km = re.match(r"^([\w.\-\"]+)\s*=\s*(.*)$", line)
        if km:
            out[(section, km.group(1))] = _ws(km.group(2))
    return out


def toml_delta(old_text, new_text):
    """added/changed/removed keys; a key moved between sections with its value is neither."""
    old, new = parse_toml_keys(old_text or ""), parse_toml_keys(new_text or "")

    def by_key(table):
        out = {}
        for (sec, key), val in table.items():
            out.setdefault(key, []).append(val)
        return out
    old_by, new_by = by_key(old), by_key(new)
    added, changed, removed = [], [], []
    for (sec, key), val in sorted(old.items()):
        if (sec, key) in new:
            if new[(sec, key)] != val:
                changed.append("{}.{}: {} -> {}".format(sec, key, val, new[(sec, key)]))
        elif val not in new_by.get(key, []):
            removed.append("{}.{}".format(sec, key))
    for (sec, key), val in sorted(new.items()):
        if (sec, key) not in old and val not in old_by.get(key, []):
            added.append("{}.{}".format(sec, key))
    parts = []
    if added:
        parts.append(_listing("added", added))
    if changed:
        parts.append("changed " + "; ".join(changed[:8]) +
                     (" (+{} more)".format(len(changed) - 8) if len(changed) > 8 else ""))
    if removed:
        parts.append(_listing("removed", removed))
    return parts


@detector("toml_keys")
def detect_toml_keys(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        if diff.cosmetic(c):
            continue
        parts = toml_delta(diff.old(c), diff.new(c))
        if parts:
            found.append({"path": c.path, "detail": "; ".join(parts)})
    return found


@detector("config_knobs")
def detect_config_knobs(diff, rule):
    """VX_CFG_* knobs and their derivations: the toml, configure, the generator."""
    found = []
    for c in diff.matching(rule["paths"]):
        if diff.cosmetic(c):
            continue
        old, new = diff.old(c) or "", diff.new(c) or ""
        if c.path.endswith(".toml"):
            parts = toml_delta(old, new)
        elif path_matches(c.path, rule.get("engine_paths", [])):
            parts = ["the knob generator changed (every derived VX_CFG_* value can move)"]
        else:
            lang = language(c.path)

            def knob_lines(text):
                return {_ws(line) for line in strip_comments(text, lang).split("\n") if "VX_CFG_" in line}
            o, n = knob_lines(old), knob_lines(new)
            parts = []
            if n - o:
                parts.append("{} VX_CFG_ line(s) added or changed".format(len(n - o)))
            if o - n:
                parts.append("{} VX_CFG_ line(s) removed or changed".format(len(o - n)))
            knobs_o = set(re.findall(r"VX_CFG_\w+", "\n".join(o)))
            knobs_n = set(re.findall(r"VX_CFG_\w+", "\n".join(n)))
            if knobs_n != knobs_o:
                parts.append(_listing("knobs +", knobs_n - knobs_o) + "; " + _listing("knobs -", knobs_o - knobs_n))
        if parts:
            found.append({"path": c.path, "detail": "; ".join(parts)})
    return found


def _logical_lines(text, lang):
    """Comment-stripped lines with backslash continuations joined."""
    return [_ws(line) for line in strip_comments(text, lang).replace("\\\n", " ").split("\n") if line.strip()]


@detector("make_surface")
def detect_make_surface(diff, rule):
    """Install rules and lists, exported toolchain paths, and ABI/ISA flags."""
    groups = {name: re.compile(rx) for name, rx in rule["groups"].items()}
    found = []
    for c in diff.matching(rule["paths"], rule.get("exclude", ())):
        if diff.cosmetic(c):
            continue
        lang = language(c.path)
        old = _logical_lines(diff.old(c) or "", lang)
        new = _logical_lines(diff.new(c) or "", lang)
        parts = []
        for name, rx in sorted(groups.items()):
            o = sorted(line for line in old if rx.search(line))
            n = sorted(line for line in new if rx.search(line))
            if o != n:
                parts.append("{} lines changed (+{} -{})".format(
                    name, len(set(n) - set(o)), len(set(o) - set(n))))
        if parts:
            found.append({"path": c.path, "detail": "; ".join(parts)})
    return found


@detector("version_pins")
def detect_version_pins(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        old, new = parse_version(diff.old(c)), parse_version(diff.new(c))
        moved = ["{} {} -> {}".format(k, old.get(k), new.get(k))
                 for k in rule["keys"] if old.get(k) != new.get(k)]
        if moved:
            found.append({"path": c.path, "detail": "; ".join(moved)})
    return found


def _yaml_keys(lines, indent):
    """key -> normalized value of the keys at `indent`; deeper lines fold into
    the key above them."""
    out, key = {}, None
    for line in lines:
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        m = re.match(r"^\s*(?:-\s+)?([\w.+-]+)\s*:\s*(.*)$", line)
        if m and ind == indent:
            key = m.group(1)
            out[key] = _ws(m.group(2))
        elif key is not None and ind > indent:
            out[key] = _ws(out[key] + " " + line.strip())
    return out


def parse_testcases(text):
    """(file-level keys, {case id: its keys}) of a ci/testcases file, comments dropped."""
    if text is None:
        return {}, {}
    lines = _strip_hash(text).split("\n")
    head, cases, current, indent = [], {}, None, None
    for line in lines:
        m = _ID_LINE.match(line)
        if m:
            current = m.group(1)
            indent = len(line) - len(line.lstrip()) + 2
            cases[current] = [line]
        elif current is not None and (not line.strip() or len(line) - len(line.lstrip()) >= indent):
            cases[current].append(line)
        else:
            current = None
            head.append(line)
    top = _yaml_keys(head, 0)
    top.pop("tests", None)
    return top, {cid: _yaml_keys(body, len(body[0]) - len(body[0].lstrip()) + 2)
                 for cid, body in cases.items()}


def _changed_keys(old, new):
    parts = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            parts.append("{} {} -> {}".format(key, old.get(key, "(none)") or "''",
                                             new.get(key, "(none)") or "''"))
    return parts


@detector("testcase_coverage")
def detect_testcase_coverage(diff, rule):
    found, old_all, new_all = [], {}, {}
    changes = diff.matching(rule["paths"])
    parsed = {}
    for c in changes:
        old_top, old_cases = parse_testcases(diff.old(c))
        new_top, new_cases = parse_testcases(diff.new(c))
        parsed[c.path] = (old_top, new_top, old_cases, new_cases)
        old_all.update(old_cases)
        new_all.update(new_cases)
    for c in changes:
        old_top, new_top, old_cases, new_cases = parsed[c.path]
        parts = []
        if old_cases and old_top != new_top:
            parts.append("file defaults: " + "; ".join(_changed_keys(old_top, new_top)))
        for cid in sorted(old_cases):
            if cid not in new_all:
                parts.append("case {} removed".format(cid))
            elif cid in new_cases and old_cases[cid] != new_cases[cid]:
                parts.append("case {}: {}".format(cid, "; ".join(
                    _changed_keys(old_cases[cid], new_cases[cid]))))
            elif cid not in new_cases and old_cases[cid] != new_all[cid]:
                parts.append("case {} moved and changed: {}".format(cid, "; ".join(
                    _changed_keys(old_cases[cid], new_all[cid]))))
        if parts:
            found.append({"path": c.path, "detail": "; ".join(parts)})
    return found


def _json_leaves(value, prefix=""):
    """path -> scalar for every leaf of a JSON document."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(_json_leaves(v, "{}/{}".format(prefix, k)))
        return out
    if isinstance(value, list):
        out = {}
        for i, v in enumerate(value):
            out.update(_json_leaves(v, "{}[{}]".format(prefix, i)))
        return out
    return {prefix: value}


@detector("golden_values")
def detect_golden_values(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        if c.status == "A":
            continue
        old, new = diff.old(c), diff.new(c)
        if new is None:
            found.append({"path": c.path, "detail": "removed" if c.status == "D" else "replaced"})
            continue
        if old is None:
            found.append({"path": c.path, "detail": "replaced"})
            continue
        try:
            before, after = _json_leaves(json.loads(old)), _json_leaves(json.loads(new))
        except ValueError:
            if old != new:
                found.append({"path": c.path, "detail": "changed"})
            continue
        up, down, other, gone = [], [], [], []
        for key in sorted(before):
            if key not in after:
                gone.append(key)
            elif before[key] != after[key]:
                a, b = before[key], after[key]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                        and not isinstance(a, bool) and not isinstance(b, bool):
                    (up if b > a else down).append(key)
                else:
                    other.append(key)
        parts = ["{} {}".format(len(v), label) for v, label in
                 ((up, "values raised"), (down, "values lowered"),
                  (other, "values changed"), (gone, "entries removed")) if v]
        if parts:
            sample = (up + down + other + gone)[:3]
            found.append({"path": c.path, "detail": "{} (e.g. {})".format(
                ", ".join(parts), ", ".join(s.lstrip("/") for s in sample))})
    return found


_THRESHOLD_WORD = re.compile(r"(?i)(?<![a-z])(?:tol|tolerances?|thresholds?|margins?)(?![a-z])")


def _threshold_lines(text):
    """Normalized code lines that name a tolerance, threshold or margin and hold a number."""
    if text is None:
        return []
    return [_ws(line) for line in _strip_hash(text).split("\n")
            if _THRESHOLD_WORD.search(line) and re.search(r"\d", line)]


@detector("gate_thresholds")
def detect_gate_thresholds(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        old, new = _threshold_lines(diff.old(c)), _threshold_lines(diff.new(c))
        gone = [line for line in old if line not in new]
        if gone:
            added = [line for line in new if line not in old]
            found.append({"path": c.path, "detail": "; ".join(
                ["- " + line for line in gone[:4]] + ["+ " + line for line in added[:4]])})
    return found


def parse_c_structs(text):
    """name -> normalized body of every struct, union or enum definition."""
    body = strip_comments(text, "c")
    body = re.sub(_CPP_LINE, " ", body, flags=re.M)
    out = {}
    for m in re.finditer(r"\b(typedef\s+)?(struct|union|enum(?:\s+class)?)\s+(?:alignas\s*\([^)]*\)\s*|__attribute__\s*\(\([^)]*\)\)\s*)*(\w+)?\s*(?::[^{;]*)?\{", body):
        end = _match_brace(body, m.end() - 1)
        name = m.group(3)
        if m.group(1):
            nm = re.match(r"\s*(?:__attribute__\s*\(\([^)]*\)\)\s*)?(\w+)", body[end:])
            if nm:
                name = nm.group(1)
        if name:
            out.setdefault(name, " ".join(_WORDS.findall(body[m.end():end - 1])))
    return out


def parse_c_macros(text):
    """name -> (function-like, normalized parameters and value) of every #define."""
    body = strip_comments(text, "c")
    out = {}
    for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(\w+)((?:\\\n|[^\n])*)", body, flags=re.M):
        out[m.group(1)] = (m.group(2).startswith("("),
                           " ".join(_WORDS.findall(m.group(2).replace("\\\n", " "))))
    return out


_C_KEYWORDS = {"if", "while", "for", "switch", "return", "sizeof", "defined", "alignof",
               "decltype", "static_assert", "__attribute__", "alignas", "noexcept",
               "operator", "throw", "case", "do", "else", "new", "delete", "template"}


def _top_statements(body):
    """Namespace-scope statements, with extern "C" and namespace blocks opened up."""
    chars = list(body)
    for m in re.finditer(r'(?:\bextern\s*"[^"\n]*"|\bnamespace(?:\s+\w+)?)\s*\{', body):
        open_at = m.end() - 1
        close = _match_brace(body, open_at) - 1
        chars[open_at] = ";"
        if close < len(chars) and chars[close] == "}":
            chars[close] = ";"
    text = "".join(chars)
    stmts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
            if depth == 0:
                head = text[start:i].split("{")[0]
                if "(" in head and not re.match(r"\s*(typedef|struct|union|enum|class|using)\b", head) \
                        and "=" not in head.split("(")[0]:
                    stmts.append(text[start:i + 1])
                    start = i + 1
        elif ch == ";" and depth == 0:
            stmts.append(text[start:i])
            start = i + 1
    stmts.append(text[start:])
    return stmts


def _hash_strings(body):
    """String literals become stable placeholders, so braces inside them cannot
    unbalance a scan and a changed asm string still changes the text."""
    return re.sub(_DQ, lambda m: '"s{:08x}"'.format(zlib.crc32(m.group(0).encode())), body)


def parse_c_symbols(text):
    """(symbols, prototypes, definitions) a header declares at namespace scope."""
    body = strip_comments(text, "c")
    symbols = set(re.findall(r"^\s*#\s*define\s+(\w+)", body, flags=re.M))
    body = re.sub(_CPP_LINE, " ", body, flags=re.M)
    body = _hash_strings(body)
    body = drop_calls(body, ["__attribute__", "alignas", "__declspec"], False, statement=False)
    prototypes, definitions = {}, {}
    body = re.sub(r"\btemplate\s*<[^<>]*(?:<[^<>]*>[^<>]*)*>", " ", body)
    for stmt in _top_statements(body):
        for m in re.finditer(r"\b(struct|union|enum|class)\s+(\w+)\s*(?=[:{;]|$)", stmt.split("{")[0]):
            symbols.add(m.group(2))
        for m in re.finditer(r"\benum\b[^{;]*\{([^}]*)\}", stmt):
            for em in re.finditer(r"(?:^|,)\s*(\w+)", m.group(1)):
                symbols.add(em.group(1))
        head = stmt.split("{")[0]
        if re.match(r"\s*typedef\b", stmt):
            tm = re.search(r"\(\s*\*\s*(\w+)\s*\)", head) or \
                re.search(r"(\w+)\s*(?:\[[^\]]*\]\s*)*$", stmt)
            if tm:
                symbols.add(tm.group(1))
            continue
        um = re.match(r"\s*using\s+(\w+)\s*=", stmt)
        if um:
            symbols.add(um.group(1))
            continue
        if "(" not in head or re.match(r"\s*(struct|union|enum|class)\b", head):
            continue
        before = head.split("(")[0]
        if "=" in before:
            continue
        fm = re.search(r"([\w*&>]\s*[\s*&]|\w\s)\s*(~?\w+)\s*$", before)
        if fm and fm.group(2) not in _C_KEYWORDS:
            name = fm.group(2)
            symbols.add(name)
            depth, end = 0, len(head)
            for i in range(len(before), len(head)):
                if head[i] == "(":
                    depth += 1
                elif head[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            prototypes[name] = " ".join(t for t in _WORDS.findall(head[:end])
                                        if t not in ("inline", "static", "extern"))
            definitions[name] = " ".join(_WORDS.findall(stmt))
    return symbols, prototypes, definitions


@detector("c_surface")
def detect_c_surface(diff, rule):
    """Symbols added or removed, macro values, struct/union/enum layouts,
    prototypes, and (for body_paths) inline definitions of a header."""
    body_paths = rule.get("body_paths") or []
    found = []
    for c in diff.matching(rule["paths"]):
        if diff.cosmetic(c):
            continue
        old, new = diff.old(c) or "", diff.new(c) or ""
        so, po, do = parse_c_symbols(old)
        sn, pn, dn = parse_c_symbols(new)
        parts = []
        if sn - so:
            parts.append(_listing("added", sn - so))
        if so - sn:
            parts.append(_listing("removed", so - sn))
        mo, mn = parse_c_macros(old), parse_c_macros(new)
        changed = {k for k in mo if k in mn and mo[k] != mn[k]}
        if changed:
            parts.append(_listing("macro value changed", changed))
        lo, ln = parse_c_structs(old), parse_c_structs(new)
        changed = {k for k in lo if k in ln and lo[k] != ln[k]}
        if changed:
            parts.append(_listing("layout changed", changed))
        changed = {k for k in po if k in pn and po[k] != pn[k]}
        if changed:
            parts.append(_listing("prototype changed", changed))
        if path_matches(c.path, body_paths):
            bodies = {k for k in do if k in dn and do[k] != dn[k]} - changed
            if bodies:
                parts.append(_listing("definition changed", bodies))
        if parts:
            found.append({"path": c.path, "detail": "; ".join(parts)})
    return found


_PR_EVENTS = frozenset(("pull_request", "pull_request_target"))
_NAME = r"""(?:([\w-]+)|'([\w-]+)'|"([\w-]+)")"""


def _name(m):
    return next(g for g in m.groups() if g)


def _workflow_triggers(text):
    """Events of the top-level `on:` key, in its scalar, flow, list or mapping form."""
    m = re.search(r"""^(?:on|'on'|"on")[ \t]*:[ \t]*(.*)$""", text, flags=re.M)
    if not m:
        return set()
    inline = m.group(1).strip()
    if inline:
        return set(re.findall(r"[\w-]+", inline))
    out, indent = set(), None
    for line in text[m.end() + 1:].split("\n"):
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        if indent is None:
            if ind == 0 and not line.startswith("-"):
                break
            indent = ind
        if ind < indent or (ind == 0 and not line.startswith("-")):
            break
        if ind != indent:
            continue
        km = re.match(r"^\s*" + _NAME + r"\s*:", line)
        lm = re.match(r"^\s*-\s+" + _NAME + r"\s*$", line)
        if km or lm:
            out.add(_name(km or lm))
    return out


def _workflow_jobs(text):
    """job name -> its lines, for the top-level `jobs:` mapping."""
    jobs, current, in_jobs, indent = {}, None, False, None
    for line in text.split("\n"):
        if re.match(r"^jobs:\s*(?:#.*)?$", line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if line.strip() and not line[0].isspace() and not line.startswith("#"):
            break
        jm = re.match(r"^( +)([\w-]+):\s*(?:#.*)?$", line)
        if jm and (indent is None or len(jm.group(1)) == indent):
            indent = len(jm.group(1))
            current = jm.group(2)
            jobs[current] = []
            continue
        if current:
            jobs[current].append(line)
    return jobs


def _split_op(expr, op):
    """expr split on a top-level operator, outside parentheses and strings."""
    parts, depth, start, quote, i = [], 0, 0, None, 0
    while i < len(expr):
        ch = expr[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and expr.startswith(op, i):
            parts.append(expr[start:i])
            i += len(op)
            start = i
            continue
        i += 1
    parts.append(expr[start:])
    return parts


_EVENT_EQ = re.compile(r"""github\.event_name\s*(==|!=)\s*['"]([\w-]+)['"]|['"]([\w-]+)['"]\s*(==|!=)\s*github\.event_name""")
_FORK_GUARDS = [re.compile(rx) for rx in (
    r"github\.event\.pull_request\.head\.repo\.full_name\s*==\s*github\.repository",
    r"github\.repository\s*==\s*github\.event\.pull_request\.head\.repo\.full_name",
    r"github\.event\.pull_request\.head\.repo\.fork\s*==\s*false",
    r"!\s*github\.event\.pull_request\.head\.repo\.fork",
    r"(?:github\.event\.)?inputs\.[\w-]+",
    r"""(?:github\.event\.)?inputs\.[\w-]+\s*==\s*(?:true|'[^']+'|"[^"]+")""")]


def _reachable_from_pr(expr):
    """The pull request events from a fork on which a job condition can hold."""
    expr = expr.strip()
    while expr.startswith("(") and _match_paren(expr, 0) == len(expr):
        expr = expr[1:-1].strip()
    disjuncts = _split_op(expr, "||")
    if len(disjuncts) > 1:
        return frozenset().union(*map(_reachable_from_pr, disjuncts))
    conjuncts = _split_op(expr, "&&")
    if len(conjuncts) > 1:
        return frozenset.intersection(*map(_reachable_from_pr, conjuncts))
    em = _EVENT_EQ.fullmatch(expr)
    if em:
        op, event = em.group(1) or em.group(4), em.group(2) or em.group(3)
        return _PR_EVENTS & {event} if op == "==" else _PR_EVENTS - {event}
    if any(rx.fullmatch(expr) for rx in _FORK_GUARDS):
        return frozenset()
    return _PR_EVENTS


def _job_condition(lines):
    """A job's own `if:` expression, or None."""
    props = [ln for ln in lines if ln.strip()]
    if not props:
        return None
    ind = len(props[0]) - len(props[0].lstrip())
    for i, line in enumerate(props):
        cm = re.match(r"^ {%d}if:[ \t]*(.*)$" % ind, line)
        if not cm:
            continue
        expr = cm.group(1).strip()
        if expr[:1] in ("|", ">"):
            more = []
            for nxt in props[i + 1:]:
                if len(nxt) - len(nxt.lstrip()) <= ind:
                    break
                more.append(nxt.strip())
            expr = " ".join(more)
        elif expr[:1] == "'" and expr.endswith("'"):
            expr = expr[1:-1].replace("''", "'")
        elif expr[:1] == '"' and expr.endswith('"'):
            expr = expr[1:-1]
        expr = expr.strip()
        if expr.startswith("${{") and expr.endswith("}}"):
            expr = expr[3:-2]
        return expr
    return None


def _scripts(lines):
    """The text of every `run:` and `script:` value. A block scalar keeps its
    `#` lines: expressions expand there too, before any shell sees them."""
    out, key_indent, block = [], None, False
    for line in lines:
        stripped = line.lstrip()
        ind = len(line) - len(stripped)
        rm = re.match(r"^(-\s+)?(?:run|script):[ \t]*(.*)$", stripped)
        if rm:
            key_indent = ind + len(rm.group(1) or "")
            value = rm.group(2)
            block = value[:1] in ("|", ">")
            if not block:
                out.append(_strip_yaml_comment(value))
        elif key_indent is not None and (not stripped or ind > key_indent):
            out.append(line if block else _strip_yaml_comment(line))
        else:
            key_indent = None
    return out


_TAINTED = re.compile(r"\$\{\{\s*(github\.event\.(?:pull_request|issue|comment|review|head_commit|commits|pages"
                      r"|workflow_run|discussion)[^}]*|github\.head_ref[^}]*)\}\}")


def workflow_risks(text, is_action):
    """The set of trust-boundary exposures a workflow file contains."""
    risks = set()
    if text is None:
        return risks
    body = _strip_hash(text)
    triggers = _workflow_triggers(body) if not is_action else set()
    pr = triggers & _PR_EVENTS
    if "pull_request_target" in triggers:
        risks.add("trigger pull_request_target")
    for name, lines in (_workflow_jobs(text).items() if not is_action else [("action", text.split("\n"))]):
        props = [_strip_yaml_comment(line) for line in lines]
        job = "\n".join(props)
        runs_on = re.search(r"^\s+runs-on:\s*(.*(?:\n\s+-\s.*)*)", job, flags=re.M)
        cond = _job_condition(props)
        reachable = pr & (_reachable_from_pr(cond) if cond else _PR_EVENTS)
        if reachable and runs_on and "self-hosted" in runs_on.group(1):
            risks.add("job {}: self-hosted runner reachable from a pull request".format(name))
        for script in _scripts(lines):
            for tm in _TAINTED.finditer(script):
                expr = _ws(tm.group(1))
                risks.add("job {}: `${{{{ {} }}}}` interpolated into a script".format(name, expr))
        if pr:
            for wm in re.finditer(r"^\s+([\w-]+):\s*write\b", job, flags=re.M):
                risks.add("job {}: {} write permission on a pull request workflow".format(name, wm.group(1)))
    if pr and re.search(r"^permissions:\s*write-all", body, flags=re.M):
        risks.add("workflow: write-all permissions on a pull request workflow")
    if pr:
        top = re.search(r"^permissions:\s*\n((?:\s+.*\n)+)", body, flags=re.M)
        if top:
            for wm in re.finditer(r"^\s+([\w-]+):\s*write\b", top.group(1), flags=re.M):
                risks.add("workflow: {} write permission on a pull request workflow".format(wm.group(1)))
    return risks


@detector("workflow_trust")
def detect_workflow_trust(diff, rule):
    found = []
    for c in diff.matching(rule["paths"]):
        if c.status == "D" or diff.cosmetic(c):
            continue
        is_action = c.path.startswith(".github/actions/")
        new = workflow_risks(diff.new(c), is_action) - workflow_risks(diff.old(c), is_action)
        if new:
            found.append({"path": c.path, "detail": "; ".join(sorted(new))})
    return found


# --------------------------------------------------------------------------- #
# Evidence checks. Each returns a list of {"severity", "detail", "paths"}.
# --------------------------------------------------------------------------- #

def _substantive(diff, change, lang, macros):
    old, new = diff.old(change), diff.new(change)
    if old is None and new is None:
        return change.status != "M"

    def form(text):
        if text is None:
            return ()
        body = drop_calls(strip_comments(text, lang), macros, lang == "sv")
        return tuple(_WORDS.findall(body))
    return form(old) != form(new)


@evidence_check("lockstep")
def check_lockstep(diff, rule):
    rtl = [c.path for c in diff.matching(rule["rtl"], rule.get("rtl_exclude", ()))
           if language(c.path) == "sv"
           and _substantive(diff, c, "sv", rule.get("rtl_ignore_macros", []))]
    simx = [c.path for c in diff.matching(rule["simx"], rule.get("simx_exclude", ()))
            if language(c.path) == "c"
            and _substantive(diff, c, "c", rule.get("simx_ignore_macros", []))]
    if rtl and not simx:
        return [{"severity": "warn", "paths": rtl,
                 "detail": "RTL logic changed with no SimX change: confirm it moves no cycles "
                           "(model_parity case named), or update the SimX timing model"}]
    if simx and not rtl:
        return [{"severity": "warn", "paths": simx,
                 "detail": "SimX model changed with no RTL change: confirm it is functional only, "
                           "or that it corrects SimX toward rtlsim (parity before/after)"}]
    return []


@evidence_check("golden")
def check_golden(diff, rule):
    golden = [c.path for c in diff.matching(rule["paths"])]
    if not golden:
        return []
    text = diff.messages.lower()
    stated = any(m.lower() in text for m in rule.get("regen_markers", []))
    others = [c for c in diff.changes if c.path not in golden and not diff.cosmetic(c)
              and not path_matches(c.path, ["docs/**", "**/*.md"])]
    items = []
    if not stated:
        items.append({"severity": "warn", "paths": golden,
                      "detail": "golden data changed and no commit message names how it was "
                                "regenerated (--update-baseline[s]); a hand edit absorbs a regression"})
    else:
        items.append({"severity": "info", "paths": golden,
                      "detail": "golden data regenerated; review the numeric delta, not only that it changed"})
    if not others:
        items.append({"severity": "info", "paths": golden,
                      "detail": "golden data changed with no code change: name the earlier commit "
                                "whose effect it records"})
    return items


_ID_LINE = re.compile(r"^\s*-\s+id:\s*['\"]?([\w.+-]+)")


def _hunks(patch):
    hunks, cur = [], None
    for line in patch.split("\n"):
        if line.startswith("@@"):
            cur = ([], [])
            hunks.append(cur)
        elif cur is not None and line.startswith("-") and not line.startswith("---"):
            cur[0].append(line[1:])
        elif cur is not None and line.startswith("+") and not line.startswith("+++"):
            cur[1].append(line[1:])
    return hunks


def _numbers(text):
    """Numeric values of a flag string: 0.05, -n64, --size=128, 4x8."""
    out = []
    for tok in re.split(r"[\s,=x\"'\[\]{}]+", text):
        m = re.match(r"^(?:-{1,2}[A-Za-z]?)?(\d+(?:\.\d+)?)$", tok)
        if m:
            out.append(float(m.group(1)))
    return out


def _known_issues(text):
    if not text:
        return 0
    return len(re.findall(r"^\s*(?:-\s+)?known_issue:\s*(?!\"\"|''|$)\S", _strip_hash(text), flags=re.M))


@evidence_check("testcase_silencing")
def check_testcase_silencing(diff, rule):
    keys = rule.get("tolerance_keys", ["tolerance"])
    key_rx = re.compile(r"^\s*(?:-\s+)?({})\s*:\s*(.*)$".format("|".join(re.escape(k) for k in keys)))
    args_rx = re.compile(r"^\s*(?:-\s+)?args\s*:\s*(.*)$")
    items = []
    added_ids, removed_ids = set(), {}
    changes = diff.matching(rule["paths"])
    for c in changes:
        old, new = diff.old(c), diff.new(c)
        old_ids = {m.group(1) for m in map(_ID_LINE.match, (old or "").split("\n")) if m}
        new_ids = {m.group(1) for m in map(_ID_LINE.match, (new or "").split("\n")) if m}
        added_ids |= new_ids - old_ids
        if old_ids - new_ids:
            removed_ids[c.path] = old_ids - new_ids
        if _known_issues(new) > _known_issues(old):
            items.append({"severity": "warn", "paths": [c.path],
                          "detail": "known_issue added ({} -> {}): the case still runs but can no "
                                    "longer fail CI".format(_known_issues(old), _known_issues(new))})
        if c.status == "D" or old is None:
            continue
        for removed, added in _hunks(diff.repo.patch(diff.base, diff.head, c.path)):
            olds = [(m.group(1), _numbers(m.group(2))) for m in map(key_rx.match, removed) if m]
            news = [(m.group(1), _numbers(m.group(2))) for m in map(key_rx.match, added) if m]
            for key, nv in news:
                prev = [ov for ok, ov in olds if ok == key and ov]
                if nv and prev and max(nv) > max(prev[0]):
                    items.append({"severity": "warn", "paths": [c.path],
                                  "detail": "{} widened {} -> {}".format(key, max(prev[0]), max(nv))})
            oargs = [_numbers(m.group(1)) for m in map(args_rx.match, removed) if m]
            nargs = [_numbers(m.group(1)) for m in map(args_rx.match, added) if m]
            for ov, nv in zip(oargs, nargs):
                if ov and len(ov) == len(nv) and all(b <= a for a, b in zip(ov, nv)) and ov != nv:
                    items.append({"severity": "warn", "paths": [c.path],
                                  "detail": "args shrunk {} -> {}".format(
                                      [int(x) if x.is_integer() else x for x in ov],
                                      [int(x) if x.is_integer() else x for x in nv])})
    for path, ids in sorted(removed_ids.items()):
        gone = sorted(ids - added_ids)
        if gone:
            items.append({"severity": "warn", "paths": [path],
                          "detail": "cases removed: {}".format(", ".join(gone))})
    return items


def parse_version(text):
    out = {}
    for line in (text or "").split("\n"):
        m = re.match(r"^\s*([A-Z_]+)\s*=\s*(\S*)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


@evidence_check("toolchain_pin")
def check_toolchain_pin(diff, rule):
    items = []
    for c in diff.matching(rule["paths"]):
        old, new = parse_version(diff.old(c)), parse_version(diff.new(c))
        for key in rule.get("keys", []):
            if old.get(key) != new.get(key):
                item = {"severity": "warn", "paths": [c.path],
                        "detail": "{} {} -> {}: land the component repos, then the prebuilt tag, "
                                  "then this change last".format(key, old.get(key), new.get(key))}
                if key == "TOOLCHAIN_REV" and diff.check_remote and new.get(key):
                    out = diff.repo.git("ls-remote", rule["prebuilt_repo"],
                                        "refs/tags/" + new[key], check=False)
                    item["detail"] += "; prebuilt tag {}".format("exists" if out else "NOT FOUND")
                    if not out:
                        item["severity"] = "error"
                items.append(item)
    return items


@evidence_check("configure_copied")
def check_configure_copied(diff, rule):
    paths = [c.path for c in diff.matching(rule["paths"]) if not diff.cosmetic(c)]
    if not paths:
        return []
    return [{"severity": "info", "paths": paths,
             "detail": "configure copies these into the build tree: re-run configure before testing"}]


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def _indeterminate(policy, base, head, event, why):
    cls = policy["indeterminate"][event]
    return {"base": base, "head": head, "diff_base": None, "policy_version": policy["policy_version"],
            "event": event, "indeterminate": True, "class": cls,
            "reasons": ["indeterminate: " + why], "files": [], "contracts": [], "evidence": []}


def classify_diff(diff, policy, event):
    files = []
    for c in diff.changes:
        rules = [r for r in policy["path_rules"] if any(path_matches(p, r["paths"]) for p in c.paths())]
        cls = max_class((r["class"] for r in rules), floor="R0") if rules else policy["default_class"]
        cosmetic = diff.cosmetic(c)
        if cosmetic:
            cls = policy["cosmetic_class"]
        entry = c.as_dict()
        entry.update({"class": cls, "rules": [r["id"] for r in rules], "cosmetic": cosmetic})
        files.append(entry)
    contracts, evidence = [], []
    for rule in policy.get("detectors") or []:
        for f in DETECTOR_KINDS[rule["kind"]](diff, rule):
            f.update({"detector": rule["id"], "class": rule["class"]})
            contracts.append(f)
    for rule in policy.get("evidence") or []:
        for item in EVIDENCE_KINDS[rule["kind"]](diff, rule):
            item.update({"check": rule["id"], "min_class": rule["min_class"]})
            evidence.append(item)
    final = max_class([f["class"] for f in files] + [f["class"] for f in contracts]
                      + [e["min_class"] for e in evidence])
    reasons = []
    for f in contracts:
        if f["class"] == final:
            reasons.append("detector:" + f["detector"])
    for e in evidence:
        if e["min_class"] == final:
            reasons.append("evidence:" + e["check"])
    rule_class = {r["id"]: r["class"] for r in policy["path_rules"]}
    for f in files:
        if f["class"] == final:
            if f["cosmetic"]:
                reasons.append("cosmetic")
            else:
                reasons += ["path:" + r for r in f["rules"] if rule_class[r] == final] or ["path:default"]
    seen = set()
    reasons = [r for r in reasons if not (r in seen or seen.add(r))]
    return {"base": diff.base, "head": diff.head, "diff_base": diff.base,
            "policy_version": policy["policy_version"], "event": event, "indeterminate": False,
            "class": final, "reasons": reasons, "files": files,
            "contracts": contracts, "evidence": evidence}


def classify(repo, policy, base, head, event="pull_request", check_remote=False):
    head_sha = repo.commit(head)
    if not head_sha:
        raise GitError("unknown head {!r}".format(head))
    base_sha = repo.commit(base)
    if not base_sha:
        return _indeterminate(policy, base, head_sha, event, "no usable base {!r}".format(base))
    diff_base = repo.merge_base(base_sha, head_sha)
    if not diff_base:
        return _indeterminate(policy, base_sha, head_sha, event, "no merge base")
    diff = Diff(repo, diff_base, head_sha, repo.changes(diff_base, head_sha),
                repo.messages(diff_base, head_sha))
    diff.check_remote = check_remote
    result = classify_diff(diff, policy, event)
    result["base"] = base_sha
    return result


def classify_commit(repo, policy, sha):
    parents = repo.parents(sha)
    if not parents:
        why = "shallow-clone boundary" if repo.is_shallow_root(sha) else "root commit"
        return _indeterminate(policy, None, sha, "replay", why)
    diff = Diff(repo, parents[0], sha, repo.changes(parents[0], sha),
                repo.git("log", "-1", "--format=%B", sha))
    diff.check_remote = False
    return classify_diff(diff, policy, "replay")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def _short(sha):
    return sha[:9] if sha else "-"


def render_text(result, verbose=False):
    lines = ["{}  {}..{}  ({} files{})".format(
        result["class"], _short(result["diff_base"] or result["base"]), _short(result["head"]),
        len(result["files"]), ", indeterminate" if result["indeterminate"] else "")]
    lines.append("  reasons: " + ", ".join(result["reasons"]))
    for f in result["contracts"]:
        lines.append("  contract [{}] {}: {}".format(f["detector"], f["path"], f["detail"]))
    for e in result["evidence"]:
        paths = e["paths"]
        shown = ", ".join(paths[:4]) + (" (+{} more)".format(len(paths) - 4) if len(paths) > 4 else "")
        lines.append("  evidence [{}] {}: {} — {}".format(e["check"], e["severity"], e["detail"], shown))
    if verbose:
        for f in result["files"]:
            lines.append("  {}  {} {}{}  [{}]".format(
                f["class"], f["status"], f["path"],
                " (from {})".format(f["old_path"]) if "old_path" in f else "",
                ", ".join(f["rules"] + (["cosmetic"] if f["cosmetic"] else [])) or "default"))
    return "\n".join(lines)


def render_markdown(result, policy_note=""):
    out = ["### Change impact: {}".format(result["class"]), ""]
    if result["indeterminate"]:
        out.append("Indeterminate diff ({}).".format("; ".join(
            r.split(": ", 1)[-1] for r in result["reasons"])))
    else:
        out.append("`{}...{}`, {} files. Reasons: {}.".format(
            _short(result["diff_base"]), _short(result["head"]), len(result["files"]),
            ", ".join("`{}`".format(r) for r in result["reasons"])))
    if result["contracts"]:
        out += ["", "| contract | file | change |", "|---|---|---|"]
        for f in result["contracts"]:
            detail = f["detail"] if len(f["detail"]) <= 300 else f["detail"][:297] + "..."
            out.append("| {} | `{}` | {} |".format(f["detector"], f["path"], detail.replace("|", "\\|")))
    if result["evidence"]:
        out += ["", "| evidence | severity | detail | files |", "|---|---|---|---|"]
        for e in result["evidence"]:
            paths = e["paths"]
            shown = ", ".join("`{}`".format(p) for p in paths[:3])
            if len(paths) > 3:
                shown += " +{}".format(len(paths) - 3)
            out.append("| {} | {} | {} | {} |".format(e["check"], e["severity"],
                                                       e["detail"].replace("|", "\\|"), shown))
    counts = {}
    for f in result["files"]:
        counts[f["class"]] = counts.get(f["class"], 0) + 1
    if counts:
        out += ["", "Files by class: " + ", ".join("{} {}".format(k, counts[k]) for k in sorted(counts))]
    out += ["", "_Advisory: policy v{}{}; see docs/review/code-review-policy.md._".format(
        result["policy_version"], policy_note)]
    return "\n".join(out) + "\n"


def explain_paths(policy, paths):
    lines = []
    for p in paths:
        lines.append(p)
        rules = [r for r in policy["path_rules"] if path_matches(p, r["paths"])]
        if rules:
            for r in rules:
                lines.append("  path rule {} -> {}".format(r["id"], r["class"]))
        else:
            lines.append("  no path rule -> default {}".format(policy["default_class"]))
        for d in policy.get("detectors") or []:
            if path_matches(p, d["paths"]):
                lines.append("  detector {} ({}) can raise to {}: {}".format(
                    d["id"], d["kind"], d["class"], d["why"]))
        for e in policy.get("evidence") or []:
            pats = e.get("paths") or (e.get("rtl", []) + e.get("simx", []))
            if path_matches(p, pats):
                lines.append("  evidence {} ({}), floor {}".format(e["id"], e["kind"], e["min_class"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _open(args):
    policy = load_policy(args.policy)
    return Repo(args.repo), policy


def cmd_classify(args):
    repo, policy = _open(args)
    try:
        result = classify(repo, policy, args.base, args.head, args.event, args.check_remote)
    finally:
        repo.close()
    if args.format == "json":
        print(json.dumps(result, indent=2))
    elif args.format == "markdown":
        sys.stdout.write(render_markdown(result, args.policy_note))
    else:
        print(render_text(result))
    return 0


def cmd_explain(args):
    if args.path:
        print(explain_paths(load_policy(args.policy), args.path))
        return 0
    if not (args.base and args.head):
        sys.stderr.write("explain needs --base and --head, or --path\n")
        return 2
    repo, policy = _open(args)
    try:
        result = classify(repo, policy, args.base, args.head, args.event, args.check_remote)
    finally:
        repo.close()
    print(render_text(result, verbose=True))
    return 0


def cmd_replay(args):
    repo, policy = _open(args)
    limit = ["--max-count={}".format(args.max_count)] if args.max_count else []
    shas = list(reversed(repo.git("rev-list", "--no-merges", *limit, args.range).split()))
    rows, counts = [], dict.fromkeys(CLASSES, 0)
    try:
        for sha in shas:
            result = classify_commit(repo, policy, sha)
            subject = repo.git("log", "-1", "--format=%s", sha).strip()
            counts[result["class"]] += 1
            rows.append({"sha": sha, "subject": subject, "class": result["class"],
                         "reasons": result["reasons"], "indeterminate": result["indeterminate"],
                         "contracts": result["contracts"], "evidence": result["evidence"],
                         "files": len(result["files"])})
    finally:
        repo.close()
    total = len(rows)
    if args.format == "json":
        print(json.dumps({"range": args.range, "total": total, "counts": counts, "commits": rows}, indent=2))
        return 0
    for r in rows:
        if args.format == "tsv":
            print("\t".join([r["sha"][:9], r["class"], ",".join(r["reasons"]), r["subject"]]))
        else:
            print("{} {} {:<60} {}".format(r["sha"][:9], r["class"], r["subject"][:60], ",".join(r["reasons"])))
    print("# {}: {} commits; ".format(args.range, total) + ", ".join(
        "{} {} ({:.1f}%)".format(c, counts[c], 100.0 * counts[c] / total if total else 0) for c in CLASSES))
    return 0


def cmd_lint_policy(args):
    try:
        with open(args.policy, encoding="utf-8") as fh:
            policy = load_yaml_subset(fh.read())
    except (OSError, PolicyError) as exc:
        sys.stderr.write("POLICY ERROR: {}\n".format(exc))
        return 1
    errors = lint_policy(policy)
    for e in errors:
        sys.stderr.write("POLICY ERROR: {}\n".format(e))
    if errors:
        return 1
    print("OK: policy v{}: {} path rules, {} detectors, {} evidence checks".format(
        policy["policy_version"], len(policy["path_rules"]), len(policy.get("detectors") or []),
        len(policy.get("evidence") or [])))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Vortex change-impact classifier (advisory)")
    p.add_argument("--policy", default=DEFAULT_POLICY)
    p.add_argument("--repo", default=ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)

    def diff_args(sp, required):
        sp.add_argument("--base", required=required)
        sp.add_argument("--head", required=required)
        sp.add_argument("--event", choices=EVENTS, default="pull_request")
        sp.add_argument("--check-remote", action="store_true",
                        help="resolve a moved TOOLCHAIN_REV against the prebuilt repository")

    c = sub.add_parser("classify", help="classify the diff base...head")
    diff_args(c, True)
    c.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    c.add_argument("--policy-note", default="", help="appended to the markdown footer")
    c.set_defaults(func=cmd_classify)

    e = sub.add_parser("explain", help="per-file trace of a classification, or the rules a path meets")
    diff_args(e, False)
    e.add_argument("--path", nargs="+")
    e.set_defaults(func=cmd_explain)

    r = sub.add_parser("replay", help="classify every non-merge commit of a range")
    r.add_argument("--range", required=True, help="a rev-list range, e.g. A..B or a branch")
    r.add_argument("--max-count", type=int, help="only the newest N commits of the range")
    r.add_argument("--format", choices=("text", "json", "tsv"), default="text")
    r.set_defaults(func=cmd_replay)

    sub.add_parser("lint-policy", help="validate the policy file").set_defaults(func=cmd_lint_policy)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (GitError, PolicyError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
