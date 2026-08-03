# LaTeX, figures, and the build

Everything here is specific to `reviews/response_to_reviewers.tex` and was learned
by breaking it.

---

## 1. The letter's own environments

Defined in the preamble; use them, do not invent parallel ones.

| Environment / macro | Purpose |
|---|---|
| `\begin{rcomment}{R1.3}` | Verbatim reviewer comment. **Never edit the body.** |
| `\begin{response}` | Your answer. Sets `\leftskip` to `1.4em`. |
| `\todoblock{…}` | Red placeholder. Delete when the block is written. |
| `\changeref{…}` | Blue trailing note listing where the manuscript changed. |
| `\reviewerhead{1}{Good}` | Per-reviewer section header with the rating. |
| `rdefn` (`\newtheorem*`) | Unnumbered `Definition` for stating a formal object. |

Floats escape the `response` environment's `\leftskip`. That is correct and
wanted; do not try to indent them back.

## 2. Packages the letter needs

`amsmath, amssymb, amsthm, xcolor, enumitem, booktabs, float, graphicx,
algorithm, algpseudocode, microtype, hyperref`.

`amsthm` supplies `\newtheorem*`; `float` supplies `[H]`; `algorithm` +
`algpseudocode` supply the pseudocode float. All are present in the local TeX
Live — verify with `kpsewhich <pkg>.sty` before adding anything new, because a
missing package on Overleaf fails the build for everyone.

## 3. Float and caption conventions

**Tables**

```latex
\begin{table}[!ht]
\centering
\caption{One sentence saying what the table shows, plus the counting rule.}
\label{tab:incidence}
\begin{tabular}{@{}lrrr@{}}
\toprule
… \bottomrule
\end{tabular}
\end{table}
```

- `\caption` **above** the body. `\centering`. `\label`. Always cited by `\ref`
  from the prose: "The result is given in Table~\ref{tab:incidence}."
- `booktabs` rules only: `\toprule`, `\midrule`, `\bottomrule`. No vertical rules.
- The caption states the *counting rule* when a count could be read two ways
  ("A DAG is counted as violating when at least one non-variable node has no
  variable ancestor").

**Figures** — caption **below**, `\centering`, `\label`, cited by `\ref`. Panels
referenced as `Figure~\ref{fig:x}(a)`.

**Algorithms** — `\caption` immediately after `\begin{algorithm}[H]`, so it renders
on top. Use `\Require` / `\Ensure` for the contract.

**Numbering collision.** Once the letter has numbered tables of its own, a bare
"Table 3" meaning the *manuscript's* Table 3 is ambiguous. Write "Table 3 of the
manuscript" in running prose. Inside `\changeref{}` location lists the context is
unambiguous, so leave those alone. Never rewrite an `rcomment` to disambiguate.

## 4. Compile gate

```bash
cd /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/reviews
for i in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error response_to_reviewers.tex > /tmp/tex.log 2>&1
  echo "pass$i exit=$?"
done
grep -cE "Overfull" /tmp/tex.log          # must be 0
grep -E "Output written" /tmp/tex.log
grep -E "LaTeX Warning" /tmp/tex.log      # must be empty
pdftotext response_to_reviewers.pdf - | grep -n '??'   # must be empty
```

Two passes are required for `\ref` resolution. Do not read the exit status through
a pipe — it reports the last command's status, not `pdflatex`'s.

Then **render and look**:

```bash
pdftoppm -f <page> -l <page> -png -r 85 response_to_reviewers.pdf /tmp/page
# note: pdftoppm zero-pads multi-page output, e.g. /tmp/page-04.png
```

### The annotated manuscript, when the answer also edits the paper

Manuscript edits go into `reviews/internal_copy_reviewed_article/`, wrapped in
`{\color{blue}…}`; `article/` stays clean. See the Paths section of `SKILL.md`
for the full convention. Its gate needs **three** passes, since the paper carries
numbered environments whose `\ref`s take an extra round to settle:

```bash
cd .../reviews/internal_copy_reviewed_article/paper
for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; done
grep -c "^! " main.log                     # must be 0
grep -c "Reference.*undefined" main.log    # must be 0
grep -c "color{red}" methodology.tex       # must be 0 after integrating a patch
pdftotext main.pdf - | grep -oE "(Theorem|Lemma|Definition|Corollary) [0-9]+\.[0-9]+" \
  | sort -u -V                             # numbering the reviewers cite must survive
# repeat in ../supplementary on supplementary.tex
```

`main.log` is the file to grep, not a redirected `/tmp/tex.log`: the paper's
build is multi-file and the interesting errors surface through `\input`.

## 5. Figure generators

Live in `experiments/scripts/generate_fig_<topic>.py`, take `--output <path>`,
write vector PDF into the `reviews/` directory, and are `ruff`-clean.

**Verify numerically before drawing.** Write a throwaway script that builds the
object through the real API and prints the properties the figure will claim, on
**both engines** when the claim touches core semantics:

```python
for backend in ("python", "cpp"):
    try:
        print(backend, fast_canonical_string(dag, backend=backend))
    except Exception as exc:
        print(backend, "REFUSED", type(exc).__name__, exc)
```

**Embed nothing by hand.** The generator computes every string and value it
draws, asserts the property it illustrates, and prints them so the transcript is
auditable:

```python
word = fast_canonical_string(after, backend="cpp")
assert evaluate_dag(before, inputs) == evaluate_dag(after, inputs)
print(f"  canonical string : {word}")
```

### Three defects that survive a clean compile

**Arrowheads vanish.** `shrinkA`/`shrinkB` on `FancyArrowPatch` are in **points**;
computing them from data units silently produces a headless line. Trim the
endpoints yourself in data space instead, which is exact under `set_aspect("equal")`:

```python
ux, uy = dx / length, dy / length
FancyArrowPatch(
    (x0 + ux * (NODE_R + 0.05), y0 + uy * (NODE_R + 0.05)),
    (x1 - ux * (NODE_R + 0.09), y1 - uy * (NODE_R + 0.09)),
    arrowstyle="-|>,head_length=7,head_width=3.4",
    mutation_scale=1.0, edgecolor=c, facecolor=c,
)
```

On a *directed* graph a missing arrowhead destroys the figure's entire content, and
nothing in the log mentions it.

**Mathtext pads operators.** `$\mathtt{VkVspv+Ppc}$` renders as `VkVspv + Ppc`,
because mathtext spaces `+` as a binary operator. For token strings and code, use
two text runs with `family="monospace"` and no math mode.

**Text is too small after scaling.** The figure is scaled to `\linewidth`, typically
about 65%, so a 9.5 pt annotation renders at ~6 pt. Size in-figure fonts for the
*final* size: node labels ~18 pt, panel titles ~16 pt, annotations ~12 pt. Enlarge
`NODE_R` with the font or the label overflows the disc. Use
`plt.rcParams["font.family"] = "serif"` and `mathtext.fontset = "dejavuserif"` so
the figure sits with the body text.

### Other conventions

- `matplotlib.use("Agg")` before `pyplot`.
- Muted, print-safe palette, legible in greyscale; one accent colour for the thing
  that changed.
- Share node positions across before/after panels so the eye tracks the single
  difference.
- `bbox_inches="tight", pad_inches=0.15` — this crops letterboxing, so `figsize`
  mainly controls the font-to-drawing ratio, not the final aspect.
- Commit the generated PDF alongside the `.tex`. Overleaf compiles from the repo
  and has no access to the generator.

## 6. Overleaf

Remote `https://git@git.overleaf.com/69c1637a28a81fea2badda9a`, branch `master`,
token at `/home/mpascual/research/token-overleaf.txt`.

Use a `GIT_ASKPASS` helper so the token never enters argv or `.git/config`, fetch
before committing to confirm you are in sync, and fetch again after pushing to
confirm the remote actually moved. Full command block in `SKILL.md` §6.

Check that any `\includegraphics` target is tracked. `git status` stays silent when
a figure is already committed at an older revision, so a stale figure pushes
cleanly and then looks wrong in the compiled PDF.

**`double_blind/paper/*.tex` is a byte-identical copy of `article/paper/*.tex`,
not a symlink.** Mirror every manuscript edit. This does not apply to the letter
itself, which exists only under `reviews/`.
