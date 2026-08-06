#!/usr/bin/env bash
# Manuscript orthography and naming guard -- TPAMI-2026-05-1699 (IsalSR).
#
# Two checks, both raised by Reviewer 2 comment 8:
#   (1) spelling variant -- one variant throughout; the package is American.
#                           Rules live in scripts/manuscript_style_wordlist.txt.
#   (2) product name     -- every occurrence of the name must go through the
#                           \IsalSR macro, so the rendered form cannot drift
#                           between "ISALSR" and "IsalSR". The only permitted
#                           plain-text occurrence is the \newcommand that
#                           defines the macro.
#
# Usage:
#   scripts/check_manuscript_style.sh <dir-or-file> [<dir-or-file> ...]
#
# Scans *.tex under each argument. Exits 0 when clean, 1 on any violation,
# 2 on a usage error. No third-party dependencies: POSIX tools plus grep -P,
# so it runs anywhere the manuscript does and needs no environment set-up.
#
# Typical invocation over the whole submission package:
#   ROOT=/path/to/journal/69c1637a28a81fea2badda9a
#   scripts/check_manuscript_style.sh "$ROOT/article" "$ROOT/double_blind" \
#       "$ROOT/previously_published_statement" \
#       "$ROOT/reviews/internal_copy_reviewed_article"

set -u -o pipefail

WORDLIST="$(cd "$(dirname "$0")" && pwd)/manuscript_style_wordlist.txt"
[ -r "$WORDLIST" ] || { echo "missing word list: $WORDLIST" >&2; exit 2; }
[ "$#" -ge 1 ] || { echo "usage: $0 <dir-or-file> [...]" >&2; exit 2; }

files=()
for target in "$@"; do
    if [ -d "$target" ]; then
        while IFS= read -r f; do files+=("$f"); done < <(find "$target" -name '*.tex' -type f)
    elif [ -f "$target" ]; then
        files+=("$target")
    else
        echo "no such path: $target" >&2; exit 2
    fi
done
[ "${#files[@]}" -gt 0 ] || { echo "no .tex files found under: $*" >&2; exit 2; }

report=""

# --- check 1: spelling variant --------------------------------------------
while IFS=$'\t' read -r bad good; do
    case "${bad:-}" in ''|'#'*) continue ;; esac
    hits="$(grep -HInP -- "$bad" "${files[@]}" 2>/dev/null | grep -vP '^[^:]+:\d+:\s*%' || true)"
    if [ -n "$hits" ]; then
        while IFS= read -r line; do
            report+="SPELLING  ${line}"$'\n'
            report+="          -> use '${good}' spelling"$'\n'
        done <<< "$hits"
    fi
done < "$WORDLIST"

# --- check 2: the name must route through the \IsalSR macro ---------------
# A violation is any "IsalSR" not immediately preceded by a backslash. Two
# legitimate exceptions: the \newcommand line that defines the macro, and
# occurrences inside a \url{} or \href{} argument, where the name is part of
# a literal address and must not be re-typeset.
name_hits="$(grep -HInP -- '(?<!\\)IsalSR' "${files[@]}" 2>/dev/null \
             | grep -v 'newcommand{\\IsalSR}' \
             | grep -vP '\\(url|href)\{[^}]*IsalSR' \
             | grep -vP '^[^:]+:\d+:\s*%' || true)"
if [ -n "$name_hits" ]; then
    while IFS= read -r line; do
        report+="NAMING    ${line}"$'\n'
        report+="          -> route this occurrence through the \\IsalSR macro"$'\n'
    done <<< "$name_hits"
fi

if [ -n "$report" ]; then
    printf '%s' "$report"
    echo "manuscript style check: FAILED (${#files[@]} files scanned)" >&2
    exit 1
fi

echo "manuscript style check: OK (${#files[@]} files scanned)"
exit 0
