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
# Each file is scrubbed before matching: the arguments of commands whose
# contents never reach the reader -- cross-reference keys, citation keys,
# figure file names, literal addresses -- are emptied. Renaming
# \label{fig:neighbourhood} or \includegraphics{fig_neighbourhood.pdf} buys
# nothing in the rendered document and costs an undefined reference or a
# missing graphic if it is applied to one of the two and not the other.
# The scrub is line-for-line, so reported locations are locations in the
# original file, and the reported text is the original text.
#
# Two further exemptions, both of which the response letter needs and neither
# of which weakens the check on the manuscript:
#
#   * the body of an `rcomment' environment is a reviewer's comment quoted
#     verbatim. It must not be edited under any circumstance, so a rule firing
#     inside one is never actionable.
#   * a line carrying the marker `% style-guard-allow', or a run of lines
#     between `% style-guard-allow-begin' and `% style-guard-allow-end', is
#     skipped. This exists for use-versus-mention: the answer to R2.8 lists
#     the British forms it swept, and "the same split ran through
#     \emph{neighbourhood}" is a mention of the word, not a use of it. Every
#     exemption must carry its reason beside the marker.
#     Prefer the block form inside a paragraph. A trailing `%' suppresses the
#     newline, so an end-of-line marker mid-paragraph glues two words
#     together; a marker on a line of its own is inert in \LaTeX{}.
#
# All are line-for-line blanks, so reported locations stay true.
scrub_dir="$(mktemp -d)"
trap 'rm -rf "$scrub_dir"' EXIT
scrub_re='\\(label|ref|eqref|cref|Cref|autoref|pageref|cite[A-Za-z]*|includegraphics|input|include|bibliography|url|href|path|nolinkurl)(\[[^][]*\])?\{[^{}]*\}'

blank_exempt () {
    awk '
        /^[[:space:]]*%[[:space:]]*style-guard-allow-begin/ { blk = 1 }
        /\\begin\{rcomment\}/                               { inq = 1 }
        { print (inq || blk || /style-guard-allow/) ? "" : $0 }
        /\\end\{rcomment\}/                                 { inq = 0 }
        /^[[:space:]]*%[[:space:]]*style-guard-allow-end/   { blk = 0 }
    ' "$1"
}

n_file=0
for f in "${files[@]}"; do
    n_file=$((n_file + 1))
    scrubbed="$scrub_dir/$n_file.tex"
    blank_exempt "$f" | sed -E "s#$scrub_re#\\\\\1{}#g" > "$scrubbed"
    while IFS=$'\t' read -r bad good; do
        case "${bad:-}" in ''|'#'*) continue ;; esac
        hits="$(grep -InP -- "$bad" "$scrubbed" 2>/dev/null | grep -vP '^\d+:\s*%' || true)"
        [ -n "$hits" ] || continue
        while IFS= read -r hit; do
            lineno="${hit%%:*}"
            report+="SPELLING  ${f}:${lineno}:$(sed -n "${lineno}p" "$f")"$'\n'
            report+="          -> use '${good}' spelling"$'\n'
        done <<< "$hits"
    done < "$WORDLIST"
done

# --- check 2: the name must route through the \IsalSR macro ---------------
# A violation is any "IsalSR" not immediately preceded by a backslash. Four
# legitimate exceptions: the \newcommand line that defines the macro;
# occurrences inside a \url{} or \href{} argument, where the name is part of
# a literal address and must not be re-typeset; a quoted reviewer comment;
# and a line marked `% style-guard-allow'. The last two are the exemptions
# documented above check 1, applied here through the same blanking pass.
name_hits=""
n_file=0
for f in "${files[@]}"; do
    n_file=$((n_file + 1))
    blank_exempt "$f" > "$scrub_dir/name_$n_file.tex"
    while IFS= read -r hit; do
        lineno="${hit%%:*}"
        name_hits+="${f}:${lineno}:$(sed -n "${lineno}p" "$f")"$'\n'
    done < <(grep -InP -- '(?<!\\)IsalSR' "$scrub_dir/name_$n_file.tex" 2>/dev/null || true)
done
name_hits="$(printf '%s' "$name_hits" \
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
