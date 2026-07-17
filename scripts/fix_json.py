#!/usr/bin/env python3
"""
JSON Fixer - Removes broken characters from malformed JSON files.

Fixes applied (in order):
  1. Trailing commas before } or ]
  2. Extra closing braces } that push bracket balance below zero
  3. Extra closing brackets ] that push bracket balance below zero

A backup is created at <file>.bak before any modification unless --no-backup
is passed. String content is never touched (parser is quote-aware).
"""

import json
import sys
import os
import re
import shutil


def _remove_extra_closing(text, char_open, char_close):
    """Remove closing delimiters that make the running balance go negative.
    Skips characters inside JSON strings (quote-aware, escape-aware)."""
    result = list(text)
    balance = 0
    removed = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(result):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == char_open:
            balance += 1
        elif char == char_close:
            if balance <= 0:
                result[i] = ""
                removed += 1
            else:
                balance -= 1

    return "".join(result), removed


def _fix_trailing_commas(text):
    """Remove commas immediately before a closing } or ] (ignoring whitespace)."""
    pattern = re.compile(r",(\s*[}\]])")
    fixed, count = pattern.subn(r"\1", text)
    return fixed, count


def _report_balance(content):
    open_b = content.count("{")
    close_b = content.count("}")
    open_br = content.count("[")
    close_br = content.count("]")
    print(f"  Braces  : {{ {open_b}  }} {close_b}  (diff {open_b - close_b:+d})")
    print(f"  Brackets: [ {open_br}  ] {close_br}  (diff {open_br - close_br:+d})")


def fix_json_file(filepath, backup=True):
    if not os.path.exists(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        original = f.read()

    # Quick check — already valid
    try:
        json.loads(original)
        print(f"{filepath}: already valid JSON, nothing to do.")
        return True
    except json.JSONDecodeError as e:
        print(f"{filepath}: invalid JSON — {e.msg} at line {e.lineno}, col {e.colno}")
        print("Before fix:")
        _report_balance(original)

    if backup:
        backup_path = filepath + ".bak"
        shutil.copy2(filepath, backup_path)
        print(f"Backup: {backup_path}")

    content = original
    fixes = []

    # Pass 1: trailing commas
    content, n_commas = _fix_trailing_commas(content)
    if n_commas:
        fixes.append(f"removed {n_commas} trailing comma(s)")

    # Pass 2: extra closing braces
    content, n_braces = _remove_extra_closing(content, "{", "}")
    if n_braces:
        fixes.append(f"removed {n_braces} extra closing brace(s)")

    # Pass 3: extra closing brackets
    content, n_brackets = _remove_extra_closing(content, "[", "]")
    if n_brackets:
        fixes.append(f"removed {n_brackets} extra closing bracket(s)")

    if not fixes:
        print("No automatically fixable errors found (imbalance may require manual edit).")
        return False

    print("After fix:")
    _report_balance(content)

    # Validate result
    try:
        json.loads(content)
        valid = True
        print("Result: valid JSON")
    except json.JSONDecodeError as e:
        valid = False
        print(f"Result: still invalid — {e.msg} at line {e.lineno}, col {e.colno}")
        print("File written anyway; manual inspection needed.")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print("Fixes applied:")
    for fix in fixes:
        print(f"  • {fix}")

    return valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix common JSON errors in Optuna journal files."
    )
    parser.add_argument("file", help="Path to the JSON/journal file to fix")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a .bak backup before modifying",
    )
    args = parser.parse_args()

    ok = fix_json_file(args.file, backup=not args.no_backup)
    sys.exit(0 if ok else 1)
