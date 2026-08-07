# Campaign C2 — the submission as revised for SCBI (2026-08-07)

Manuel (soporte@scbi.uma.es) asked for two changes to the campaign described in the
mail of 2026-08-06. Both are implemented, tested and measured. This file is the
reference for the reply and for `EXECUTION-PLAN` §11.

---

## 1. What they asked, and what we did

| SCBI request | Response |
|---|---|
| Group short jobs so each combined job lasts **≥ 2 h** | **12,600 tasks → 3,474.** Shortest expected task **20 h**, mean 23 h. No task is planned under 2 h |
| Use **localscratch**, copy back only results | Every task writes its whole output tree to `$LOCALSCRATCH` on the compute node and copies back one finished cell at a time |

They were right, and it was measurable in our own accounting before we changed
anything. Of the **23,058 job records this account produced between 2026-07-01 and
2026-08-07, 51.5 % ran for under two minutes and only 0.9 % reached two hours.**
Two things produced that: our 15-minute smoke waves, and the aborted 2026-08-06
submission in which all 12,600 tasks died in seconds on a defect of ours (a seed-spec
guard that did not expand ranges — fixed in 2ff0050).

It is also true of the production payload, which is the part that matters. From the
previous campaign's own `sacct` record (COMPLETED tasks, 2026-03-01 → 2026-07-01):

| array | n | mean | p50 | under 2 h |
|---|---|---|---|---|
| `udfs` hard / cherrypicked / roundoff | 1,680 | 12.01 h | 12.01 h | 0 % |
| `udfs` feynman | 4,924 | 0.82 h | 0.03 h | ~93 % |
| `bingo` nguyen | 5,608 | 0.14 h | 0.02 h | 100 % |
| `bingo` feynman | 297 | 0.20 h | 0.02 h | 100 % |
| `bingo` hard | 564 | 5.27 h | 5.96 h | 26 % |
| `bingo` roundoff | 479 | 0.69 h | 0.73 h | 100 % |

The workload is bimodal: one solver saturates a 12 h budget on the problems it cannot
solve exactly and exits in seconds on the ones it can; the other stops on an evaluation
count and is fast on the easy suites. About **43 % of the 12,600 planned tasks would
have run under two hours.**

---

## 2. The revised submission

| | before | after |
|---|---|---|
| Arrays | 42 | 42 |
| **SLURM tasks** | **12,600** | **3,474** (−72 %) |
| Scientific units ("cells") | 12,600 | 12,600 (unchanged) |
| Cells per task | 1 | 2 – 164, sized per array |
| Shortest expected task | **0.1 h** | **20.0 h** |
| Mean expected task | 6.7 h | 23.0 h |
| `--time` per task | 16 h | 25 – 47 h, sized per array |
| `--ntasks` / `--cpus-per-task` | 1 / 1 | 1 / 1 (unchanged) |
| `--mem` | 16 G / 32 G | 16 G / 32 G (unchanged) |
| `--constraint` | `sr` | `sr` (unchanged) |
| Peak concurrent tasks | 2,016 | 2,016 (unchanged) |
| Output during the run | FSCRATCH | **`$LOCALSCRATCH`**, copied back per cell |
| Total core-hours | ~80,000 | ~80,000 (unchanged) |
| **Makespan** | **48.0 h** | **49.2 h** (+2.5 %) |

Longest wall is **47 h**, a full day inside `medium_uma`'s `MaxWall = 3-00:00:00`.

### Why grouping costs almost nothing

An array of `N` cells of duration `T` under throttle `K` finishes at `N·T/K`. Grouped
into `N/B` tasks of duration `B·T` it finishes at `(N/B)·(B·T)/K = N·T/K` — identical.
Two things keep that true and both are enforced in code:

1. **Each array keeps at least `K` tasks.** The slot apportionment caps the throttle at
   the *post-grouping* task count, so no slot idles.
   (`c2_slot_plan.build_plan`, asserted by `test_no_array_is_starved_of_its_slots_by_chunking`.)
2. **Tasks are indivisible**, so the last round may be partly empty. That residue is the
   whole cost: **48.0 h → 49.2 h**, asserted under a 10 % budget by
   `test_chunking_does_not_cost_makespan`.

Manuel's own prediction ("no alarga el tiempo de ejecución del conjunto, de hecho es
probable que lo reduzca") is right in direction: we pay 1.2 h of quantisation and give
back 9,126 scheduling decisions.

---

## 3. How a grouped task behaves

Each task runs its cells **sequentially, as separate processes**, so peak memory is
still a per-cell quantity and `--mem` is unchanged.

**Deadline.** A task refuses to *start* a cell unless the full per-cell budget
(12 h) plus teardown still fits inside its wall. So a SLURM `TIMEOUT` is impossible by
construction, and every cell that runs gets its full, identical budget — trimming a late
cell's budget would make its wall clock incomparable with its paired cells, and wall
clock is a reported quantity in this paper.

**The first cell is exempt from the deadline.** Without that exemption a task whose
start-up alone exceeded the cutoff would defer its whole chunk, the sweep would re-derive
the same chunk and defer it again, and the array would livelock. Caught by a mock run,
not by inspection.

**Sweep.** Cells not started are printed as `DEFERRED` and picked up by a sweep pass
submitted at launch with `--dependency=afterany`. It re-derives the identical partition,
and the resume logic skips completed cells, so it costs seconds when nothing spilled.
Simulated against the measured distributions it carries 5–20 cells out of 12,600.

**Failure isolation.** A failing cell does not abort the rest of the chunk; it is named
in the log, recorded in the status ledger, and makes the task exit non-zero so `sacct`
still works as a first-line census.

---

## 4. Localscratch

```
$LOCALSCRATCH/$USER/c2_<jobid>_<taskid>/out/       # the payload's entire output tree
$LOCALSCRATCH/$USER/c2_<jobid>_<taskid>/pycache/   # this task's private bytecode cache
```

**Copy-back happens twice, and both are load-bearing:**

- **After each cell** — the cell's `seed_NN/` directory. A node loss at hour 30 of 47
  costs the cell in flight and nothing else.
- **At the end of the task** — the **entire** `out/` tree, mirrored wholesale, from a
  `trap … EXIT` handler. Also from `trap … TERM`, because SLURM sends SIGTERM and waits
  `KillWait` (30 s) before SIGKILL: without a TERM handler a wall-clock kill or an
  `scancel` terminates the shell *without* running the EXIT trap, and the whole chunk's
  finished cells die on the node.

🔴 **The final mirror is not belt-and-braces; the per-cell copy alone loses data.**
Measured 2026-08-07 by running the same chunk twice — once staged, once direct — and
diffing the two trees file by file:

```
LOST: ./metadata.json
```

The orchestrator writes `metadata.json` at the **root** of the output tree
(`orchestrator.py:665`), not inside any cell's `seed_NN/`, so a per-cell copy-back never
touched it. `c2_certify.py:842` reads that file and criterion **C1.4 fails without it**.
The campaign would have run for two days and then failed certification on evidence that
no longer existed anywhere. Every per-cell artefact came back correctly, which is exactly
why counting `run_log.json` did not catch it.

The fix is structural: the final copy mirrors the tree and knows nothing about what is in
it, so the next artefact written at the root cannot repeat this.

**Other guarantees:**

- Root-level files are copied to a per-task temp name and then `mv`-ed into place. Every
  one of the 3,474 tasks writes `metadata.json` to the same durable path, and `cp`
  truncates before it writes; a rename within one directory is atomic, so a concurrent
  reader sees the old file or the new one, never half of one. A truncated `metadata.json`
  fails C1.4 the same way a missing one does.
- The node's copy is deleted **only after** the copy-back succeeded and the durable file
  count was checked against the local one. On failure the task says so loudly and leaves
  the data on the node rather than reporting success having deleted the only copy.
- `sr` nodes carry **800 GB** of localscratch (`sinfo -o "%20N %15d"`); the campaign's
  footprint is megabytes.
- Existing artefacts are staged **in** before a cell runs, so the resume logic still sees
  what is already done and cannot overwrite a good result with a shorter re-run.
- The task's bytecode cache is scoped to its own localscratch directory. Picasso's login
  profile exports a **shared** `PYTHONPYCACHEPREFIX`, and with up to 128 tasks per node
  writing the same `.pyc` paths that is a race — the 2026-08-07 mock lost one cell in
  twelve to an intermittent `ModuleNotFoundError` on a file that was present on disk.

Honest note: our payload does not write thousands of temporary files *during* the search
— it writes ~6 small files per cell at the end, ~72,000 over the campaign. Localscratch
still removes those from the shared filesystem and batches them into one copy per cell,
and it removes whatever the libraries write that we have not audited.

---

## 5. Evidence

| Check | Result |
|---|---|
| `tests/unit/test_c2_slot_plan.py` | 71 passed |
| `tests/unit/test_c2_task_spec_chunking.py` | 30 passed |
| Full unit suite | 7,611 passed (one pre-existing `test_numerical_audit` failure, `paper/*.tex`, untouched here) |
| `slurm/c2_smoke/mock_chunk_test.sh` — local, real payload, no SLURM | **26/26 checks** |
| **`slurm/c2_smoke/chunk_smoke.sh` — Picasso, 2×2×3×2, both waves** | **24/24 tasks COMPLETED, 48 cells, `chunk_smoke_verify.py` all 6 checks PASS** |
| `sbatch --test-only`, full campaign shape | **42/42 accepted** at the new 25–47 h walls |
| Stage F gate G9 / G9b / G12 | plan is 42 arrays / 12,600 cells; min task 20.0 h, max wall 47 h; chunked decode partitions the array exactly |

### The Picasso copy-back smoke (`chunk_smoke.sh`)

The decisive test. 2 problems × 2 seeds × 3 arms × 2 methods = 24 cells, run **twice** on
`sr` nodes under the real `worker.sh`: once with `C2_USE_LOCALSCRATCH=1` (the campaign's
path) and once with `=0` (the reference). The waves are identical in every other respect,
so the two trees can be diffed file by file.

```
Staged (localscratch):   125 files, 24 cells
Direct (reference)   :   125 files, 24 cells

  PASS  no file from the direct run is missing after staging
  PASS  root metadata.json came back AND parses (c2_certify C1.4)   2406 bytes
  PASS  staged ran all 24 cells                                     24/24
  PASS  the two waves cover the same cells
  PASS  every cell directory has its full artefact set              24 cell dirs
  PASS  no staged file arrived empty where the reference is not
```

Per-artefact counts are identical across the two waves — 24 `run_log.json`,
24 `trajectory.csv`, 24 `status.json`, 24 `complexity.json`, 16 `fallback_ledger.json`,
12 `convergence_log.npz`, 1 `metadata.json`. Every one of the 12 staged tasks logged
`Copy-back: N file(s) verified`, all 24 tasks reported `2 ok, 0 failed, 0 deferred`, and
**all seven `sr` nodes used were left with zero localscratch residue.**

The per-task copy counts sum to 136 against 125 files on disk; the difference is exactly
the 11 `metadata.json` overwrites, i.e. the atomic-rename path was exercised 12 times
concurrently and left **no `.tmp` residue** and a file that parses.

### Two further defects the smoke found

Neither was visible to inspection, and both are fixed:

3. **`--export` truncated the problem list.** `sbatch --export` is comma-separated, so a
   comma *inside a value* cuts it short: `C2_PROBLEMS=Nguyen-1,Nguyen-2` was delivered as
   `Nguyen-1`. The array was sized for 2 tasks, the decode produced 1, and all twelve
   task-2's died with `index 2 out of range [1, 1]`. This is the same trap that killed the
   2026-08-06 submission through `C2_SEEDS`, hit again through a new variable. List-valued
   variables are now shipped colon-separated, **and the launcher derives the task count
   from the decoder instead of hard-coding it** — a hard-coded count cannot disagree with
   the decode, so it cannot catch a decode that changed.
4. **The copy-back verification could fail on a healthy task.** `finalize` walked the
   *shared* results root with `find` to count files. Under `set -euo pipefail`, a
   concurrent task renaming its atomic temp file made `find` fail to stat a vanished entry
   and exit non-zero; `pipefail` propagated it, `set -e` aborted the trap, and the task
   exited 1 **after its data was already safely copied** — reporting failure and skipping
   its own cleanup. One task in twelve. `finalize` now runs with `set +e`, returns the
   entry status unchanged, and verifies only the paths the task itself owns.

> An EXIT trap that can fail turns cosmetic errors into false alarms and skipped cleanup.
> The trap is the last thing standing between a finished job and lost results; it must be
> the most defensive code in the worker, not the least.

### One pre-existing limitation, noted not fixed

`metadata.json` is written by every task to the **campaign root**, so the surviving copy
describes whichever `(method, suite, arm)` finished last — the smoke's copy records
`seeds: [101], variants: ['isalsr']`. This is unchanged from the pre-localscratch path
(the orchestrator has always written it per invocation to the shared root) and is in fact
now safer, because the write is atomic and there are 3,474 writers instead of 12,600.
`c2_certify` C1.4 reads only `config.benchmarks`, which is identical across the tasks of
one suite — but with seven suites in one root it will check the shapes of one suite, not
seven. Out of scope here; worth a ticket.

The partition is proved to be an even bijection over every shape the planner can emit
(`n_cells` 1–130 × `bundle` 1–40): no cell is dropped, none is run twice, and block sizes
differ by at most one.

Two defects were found by running the mock rather than by reading the code, and both are
fixed:

1. **Livelock.** A task whose start-up alone exceeded the cutoff deferred its entire
   chunk; the sweep re-derived the same chunk and deferred it again. The first cell of a
   chunk is now exempt from the deadline, which is sound because the planner asserts the
   wall always has room for one full cell.
2. **Shared bytecode cache.** Picasso's login profile exports a single
   `PYTHONPYCACHEPREFIX` for the whole user, so every task on a node writes the same
   `.pyc` paths — 128 per `sr` node, 2,016 across the campaign. The first mock lost one
   cell in twelve to an intermittent `ModuleNotFoundError` on a module that was present on
   disk. Each task now gets its own cache inside its localscratch directory; the re-run
   was 12/12.

**Side benefit.** The inode projection falls from 84,381 to **75,255** (results 71,781 +
logs 3,474), because logs scale with tasks, not cells. Live headroom is 90,300.

---

## 6. Borrador de respuesta a Manuel (SCBI)

> Estimado Manuel:
>
> Muchas gracias por la respuesta y por el detalle. Hemos aplicado las dos
> indicaciones y le resumo el resultado, porque en ambos casos tenía usted razón y
> lo hemos podido medir sobre nuestros propios datos.
>
> **1. Duración de los trabajos.** Revisando nuestro histórico de `sacct`, de los
> 23.058 registros de trabajo que esta cuenta generó entre el 1 de julio y el 7 de
> agosto, el **51,5 % duró menos de dos minutos** y sólo el 0,9 % llegó a las dos
> horas. En parte fueron pruebas cortas nuestras, y en parte el envío del 6 de
> agosto, en el que las 12.600 tareas murieron a los pocos segundos por un fallo
> nuestro en el script (ya corregido). Pero también afecta a la carga real: nuestro
> trabajo es bimodal, con un solver que agota su presupuesto de 12 h en los
> problemas que no resuelve y termina en segundos en los que sí, de forma que
> alrededor del 43 % de las tareas previstas habrían durado menos de dos horas.
>
> Hemos reescrito el envío para que **cada tarea ejecute varias unidades de trabajo
> en secuencia**, dimensionando el grupo por cada array a partir de la duración por
> unidad medida en la campaña anterior:
>
> | | antes | ahora |
> |---|---|---|
> | Arrays | 42 | 42 |
> | **Tareas SLURM** | **12.600** | **3.474** (−72 %) |
> | Duración mínima prevista por tarea | 0,1 h | **20 h** |
> | Duración media prevista por tarea | 6,7 h | 23 h |
> | `--time` por tarea | 16 h | 25–47 h |
> | Núcleos / memoria por tarea | 1 / 16–32 GB | sin cambios |
> | Tareas simultáneas máximas | 2.016 | sin cambios |
> | Horas de núcleo totales | ~80.000 | sin cambios |
>
> Ninguna tarea está planificada por debajo de las dos horas, y hemos añadido una
> comprobación automática que **rechaza el envío** si alguna lo estuviera.
>
> Su previsión se cumple: el conjunto no se alarga. Agrupar es neutro en tiempo
> total mientras cada array conserve al menos tantas tareas como ranuras
> simultáneas tenga asignadas, y así lo hemos impuesto en el código. El único coste
> es que las tareas son indivisibles y la última ronda puede quedar incompleta:
> medido sobre el plan real, **48,0 h → 49,2 h**, un 2,5 %, a cambio de 9.126
> decisiones de planificación menos.
>
> Cada tarea lleva además un plazo interno: no empieza una unidad nueva si no cabe
> entera en su `--time`, de modo que un `TIMEOUT` es imposible por construcción. Lo
> que no le da tiempo a empezar lo recoge un array dependiente (`afterany`) que
> enviamos en el mismo momento.
>
> **2. Localscratch.** Aplicado. Cada tarea genera todo su árbol de salida en el
> disco local del nodo y lo copia a fscratch **dos veces**: unidad a unidad según
> terminan, para acotar la pérdida si el nodo cae a mitad de la tarea, y el árbol
> completo al final de la tarea. Esa copia final se ejecuta también al recibir
> `SIGTERM`, de modo que un `scancel` o un fin de `--time` no se lleva por delante
> el trabajo ya hecho, y el directorio local sólo se borra una vez verificada la
> copia. Como efecto secundario, los ficheros de log bajan de 12.600 a 3.474 y la
> previsión de inodos de la campaña pasa de 84.381 a 75.255.
>
> Le indico también, por si les resulta útil: el perfil de login exporta un
> `PYTHONPYCACHEPREFIX` común para todo el usuario, de manera que todas las tareas
> de un mismo nodo escriben los mismos ficheros `.pyc`. En nuestras pruebas eso
> provocó un fallo de importación intermitente (una unidad de doce). Lo hemos
> resuelto dando a cada tarea su propia caché dentro de su directorio de
> localscratch, pero quizá merezca la pena tenerlo en cuenta para otros usuarios
> con arrays grandes de Python.
>
> **Validación.** Antes de escribirle hemos comprobado el envío completo con
> `sbatch --test-only` (42/42 aceptados) y hemos ejecutado un array real de prueba
> de 3 tareas × 4 unidades en nodos `sr`, que completó las 12 unidades usando
> localscratch. El resto son pruebas locales.
>
> Quedamos a su disposición si prefieren que ajustemos algo más — el ritmo de
> envío, el número de tareas simultáneas o la franja horaria. Enviaremos los 42
> arrays espaciados 20 segundos, como acordamos, y no lanzaremos nada hasta tener
> su visto bueno.
>
> Muchas gracias de nuevo por el tiempo y por la orientación.
>
> Un cordial saludo,
>
> Mario Pascual González
> Grupo de Inteligencia Computacional y Análisis de Imagen

---

## 7. Submitting C2 on Picasso — the runbook

**Run the steps in order. Every one of them is cheap; the campaign is not.**
`$REPO` below is `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR`.

> ### ⚠ State of the deployed tree, as of 2026-08-07
>
> The Picasso checkout is at **2ff0050 plus six files rsynced by hand** to run the
> cluster smokes: `slurm/c2_smoke/{worker,launcher,chunk_smoke}.sh`,
> `slurm/c2_campaign/submit_paced.sh`, `experiments/scripts/c2_{task_spec,slot_plan}.py`.
> That is a mixed state and **must not be what the campaign runs.** Step 1 replaces it.

### Step 0 — from the workstation, on the commit to be tagged

```bash
cd ~/research/code/IsalSR
git status --porcelain            # MUST be empty; deploy.sh refuses a dirty tree
python -m pytest tests/unit -q    # 7,611 pass
bash slurm/c2_smoke/mock_chunk_test.sh          # 26/26, ~10 min, real payload
```

If the tree carries another agent's uncommitted work, deploy from a clean clone of the
branch tip. **Never commit someone else's files** (defect 14).

### Step 1 — deploy

```bash
bash slurm/c2_smoke/deploy.sh     # rsync incl. .git, verify SP-1, rebuild, verify SP-2
```

`deploy.sh` is the only path: Picasso has no outbound SSH, so `git pull` cannot work
there (defect 13). Verify the rebuild actually happened — `--no-build-isolation` fails
silently and keeps loading the stale `.so`:

```bash
ssh picasso 'cd $REPO && python -c "from isalsr.core import _native; print(_native.__file__)"'
ssh picasso 'stat -c "%y" <that path>'          # must post-date the last C++ edit
```

### Step 2 — prove the copy-back on the deployed tree

**Do not skip this.** It is 20 minutes and it is the only thing standing between a
two-day campaign and a tree with holes in it. It runs the real `worker.sh` twice, staged
and direct, and diffs the results file by file.

```bash
ssh picasso "cd \$REPO && bash slurm/c2_smoke/chunk_smoke.sh"
# wait for the 12 arrays (24 tasks) to drain, then:
ssh picasso "cd \$REPO && python slurm/c2_smoke/chunk_smoke_verify.py \
             \$FSCRATCH/results/isalsr/c2_chunk_smoke"
# expect: SMOKE OK -- localscratch loses nothing
```

### Step 3 — the Stage F gate

```bash
bash slurm/c2_campaign/stage_f_preflight.sh
```

Fails closed; submits nothing. G9 asserts 42 arrays / 12,600 **cells**, G9b asserts SCBI's
2 h floor and a positive deadline on every array, G12 decodes through the campaign's real
bundle and asserts the chunks partition the array exactly.

### Step 4 — preview

```bash
python -m experiments.scripts.c2_slot_plan --seeds 1-30 --table    # plan + totals
bash slurm/c2_campaign/launch.sh --dry-run                         # the 42 arrays
ssh picasso "cd \$REPO && C2_PROFILE=campaign bash slurm/c2_smoke/launcher.sh --test-only"
```

`--test-only` must report **42/42 accepted** and an inode projection inside the live
headroom. It validates a *resource request* and never runs the worker — which is why
Step 2 exists.

### Step 5 — submit, PACED

> ### 🔴 `C2_MIN_JOBID` is MANDATORY here (found 2026-08-07)
>
> `submit_paced.sh` skips arrays whose **job name** already appears in
> `sacct -S today`, scoped by `C2_MIN_JOBID`, which **defaults to 0**. The smoke
> and campaign profiles build the *same* 42 job names — both come from the one
> launcher, `c2s_${METHOD:0:1}${ARM:0:1}_${SUITE}`.
>
> **Step 2 above requires Stage C on this very commit**, so those 42 names are
> always in today's `sacct` by the time you reach Step 5. With the default, the
> submission prints `SKIP (already submitted)` forty-two times, submits **zero**
> arrays, attaches the aggregation `afterany` to the **smoke** wave, writes the
> smoke ids to `job_ids.txt`, and **exits 0** — a silent no-op that reads as
> success. Same family as 2026-08-06, where `--test-only` reported 42/42 while
> every task was about to abort.

```bash
# Compute the floor FIRST -- every campaign array is submitted after this point.
MIN=$(ssh picasso "sacct -S today -n -P -X -o JobID | cut -d_ -f1 | sort -n | tail -1")
MIN=$(( MIN + 1 ))

ssh picasso "cd \$REPO && C2_MIN_JOBID=${MIN} bash slurm/c2_campaign/submit_paced.sh --dry-run"
#   the dry run MUST report:  already present (job id >= ${MIN}): 0
#   and list all 42 arrays.  Any SKIP line means the floor is too low.

ssh picasso "cd \$REPO && C2_MIN_JOBID=${MIN} bash slurm/c2_campaign/submit_paced.sh"
```

> ### 🔴 Then submit the sweep arrays — `submit_paced.sh` does not
>
> `launcher.sh` submits a `c2w_*` sweep per array (`afterany` on the mains) and
> extends the aggregation dependency to include them. `submit_paced.sh` has **no
> sweep block**, so on its own it never recovers cells a task's deadline refused
> to start (5–20 of 12,600 by the launcher's own simulation), aggregates a tree
> missing them, and breaks the commitment made to SCBI in §6 above.
>
> Submit them as a second paced pass, then repoint the aggregation:
>
> ```bash
> scontrol update JobId=<agg> Dependency=afterany:<42 mains>:<42 sweeps>
> scontrol show job <agg> | tr ' ' '\n' | grep '^Dependency='   # expect 84 entries
> ```
>
> Submitting them minutes later is equivalent to submitting them together: they
> are `afterany` on the mains either way, and resume makes them additive.

> 🔴 **Use `submit_paced.sh`, never `launch.sh`, for the real campaign.** `launch.sh`
> submits in a tight loop; on 2026-08-06 that hit
> `Slurm temporarily unable to accept job` after 29 of 42 and aborted *before* writing
> `job_ids.txt`, leaving 29 untracked arrays on the cluster. `submit_paced.sh` paces at
> 20 s (`C2_SLEEP` to raise it) and is **idempotent** — it skips arrays that already
> exist, so re-running after an abort completes the set instead of duplicating it.
>
> **Your first act after any submission error is `squeue`.** Assume jobs exist until
> proven otherwise.

### Step 6 — the first six hours

```bash
ssh picasso 'squeue'                                        # NOT squeue -u (Lua wrapper rejects it)
ssh picasso 'sacct -X -S today -o JobID,State -P -n | awk -F"|" "{print \$2}" | sort | uniq -c'
ssh picasso 'grep -c "^Copy-back:.*verified" $FSCRATCH/execs/isalsr/c2_3arm/logs/*.out | tail'
```

1. Record the 42 job ids in `EXECUTION-PLAN` §11.3 — the launcher wrote them to
   `<logs>/job_ids.txt` in submission order, which is §11.3's row order.
2. **Watch achieved concurrency.** The plan assumes 2,016 slots; the `short` QOS granted
   934, and C2 runs at 4.1× lower priority with fairshare eroding as it burns. Lowering
   `C2_SLOT_BUDGET` mid-campaign is safe — it touches no config and no deployed file.
3. **Watch for `[FATAL] copy-back`** in the logs. It means a task's results are still on a
   compute node and it deliberately did not delete them; recover before the node's scratch
   is reclaimed.
4. After ~24 h, read the measured Bingo wall clock and re-apportion:
   ```bash
   python -m experiments.scripts.c2_slot_plan --bingo-hours <measured> \
          --rebalance <logs>/job_ids.txt        # emits 42 scontrol lines
   ```
   `scontrol update JobId=<id> ArrayTaskThrottle=<n>` re-apportions a **running** array.
   It touches no config and no deployed file, so it is **not** defect 10.

### Step 7 — do not touch the tree while it runs

**Never deploy while an array runs.** A mid-wave redeploy splits provenance across two
HEADs and marks every subsequent cell `-dirty` (defect 10; it cost v4 161 of its 1,260
cells).

### Rollback

```bash
ssh picasso 'scancel --name=c2s_*'     # or each id in job_ids.txt
```

The orchestrator's resume makes a re-launch additive: a completed `(method, arm, problem,
seed)` is skipped, so re-running the same arrays costs only the missing cells. **Do not
delete the campaign root to "start clean"** — the status ledger is what makes cell
completeness provable (§5.5), and a partially completed triple is worse than a missing one
because it silently unbalances the paired test.

---

## 8. Escape hatches

| Variable / flag | Effect |
|---|---|
| `--no-chunk` | One cell per task, the pre-2026-08-07 shape. A/B only: it restores the 12,600-task submission SCBI asked us to stop sending, and the planner says so in its summary |
| `C2_USE_LOCALSCRATCH=0` | Write straight to FSCRATCH. Exists so the staging claim stays falsifiable, not because it is ever better |
| `C2_SWEEP=0` | Do not submit the dependent sweep arrays. Deferred cells then need a manual re-run |
| `C2_SLOT_BUDGET=<n>` | Total concurrent slots. Safe to lower mid-campaign |
| `C2_SLEEP=<s>` | Pacing between array submissions (default 20 s) |
| `C2_MAX_BUNDLE=<n>` | Ceiling on cells per task; the Stage C smoke uses 4 to stay inside the 2 h `short` QOS |
| `C2_PROBLEMS=<a:b:c>` | Problem subset for probes. **Colon-separated** — `--export` splits on commas |
