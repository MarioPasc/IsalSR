## SP-1…SP-6 (AC-4b)

| # | Property | Cells passing | Verdict |
|---|---|---|---|
| SP-1 | Provenance — running the commit we think we are | 40/40 | **PASS** |
| SP-2 | Installation freshness — the .so is the code we edited | 40/40 | **PASS** |
| SP-3 | Engine native, with the forced-Python negative control | 40/40 | **PASS** |
| SP-4 | Alphabet — no Sub/Div, no '-'/'/' in any canonical string | 40/40 | **PASS** |
| SP-5 | Both hosts — UDFS and Bingo | 40/40 | **PASS** |
| SP-6 | T06 fallback ledger importable, five paths exposed | 40/40 | **PASS** |
| SP-3′ | *negative control* — forced Python actually reports `python` | 40/40 | **PASS** |


## Stage C readiness — fields that must exist BEFORE launch

| Stage C criterion | Needs | Present in RunLog | Checkable? |
|---|---|---|---|
| **C1.9** | the five T06 fallback rates | 0/40 | **NO** |
| **C1.14** | `engine == native` | 0/40 | **NO** |


## Per-run summary

| Host | Arm | Problem | R² test | ρ | unique | wall (s) | engine | NaN |
|---|---|---|---|---|---|---|---|---|
| bingo | isalsr | I.12.2 | 1.0000 | 1.7756 | 200616 | 522 | — | — |
| bingo | isalsr | II.34.29a | 1.0000 | 1.7633 | 28696 | 67 | — | — |
| bingo | isalsr | II.34.29b | 1.0000 | 1.7695 | 71067 | 180 | — | — |
| bingo | isalsr | III.19.51 | 0.9567 | 1.7776 | 444753 | 1469 | — | — |
| bingo | isalsr | III.4.32 | 1.0000 | 1.7736 | 413826 | 1470 | — | — |
| bingo | isalsr | test_4 | 0.9959 | 1.7870 | 413239 | 1469 | — | — |
| bingo | isalsr | Strogatz-bacres1 | 1.0000 | 1.7950 | 360329 | 1002 | — | — |
| bingo | isalsr | Strogatz-bacres2 | 1.0000 | 1.7931 | 134395 | 294 | — | — |
| bingo | isalsr | Strogatz-barmag1 | 0.9824 | 1.8048 | 562135 | 1469 | — | — |
| bingo | isalsr | Strogatz-barmag2 | 1.0000 | 1.7957 | 527598 | 1258 | — | — |
| bingo | isalsr | Strogatz-glider1 | 1.0000 | 1.7907 | 39448 | 61 | — | — |
| bingo | isalsr | Strogatz-glider2 | 1.0000 | 1.7934 | 19834 | 30 | — | — |
| bingo | isalsr | Strogatz-lv1 | 1.0000 | 1.7703 | 31412 | 47 | — | — |
| bingo | isalsr | Strogatz-lv2 | 1.0000 | 1.7874 | 137625 | 267 | — | — |
| bingo | isalsr | Strogatz-predprey1 | 0.9999 | 1.7918 | 595012 | 1470 | — | — |
| bingo | isalsr | Strogatz-predprey2 | 0.9982 | 1.8046 | 585217 | 1470 | — | — |
| bingo | isalsr | Strogatz-shearflow1 | 1.0000 | 1.8057 | 599570 | 1471 | — | — |
| bingo | isalsr | Strogatz-shearflow2 | 1.0000 | 1.7981 | 412638 | 953 | — | — |
| bingo | isalsr | Strogatz-vdp1 | 1.0000 | 1.7969 | 248428 | 540 | — | — |
| bingo | isalsr | Strogatz-vdp2 | 1.0000 | 1.1710 | 427 | 0 | — | — |
| udfs | isalsr | I.12.2 | 0.3087 | 1.3696 | 1721 | 1501 | — | — |
| udfs | isalsr | II.34.29a | 0.6928 | 1.3920 | 2079 | 1500 | — | — |
| udfs | isalsr | II.34.29b | 0.4023 | 1.3364 | 2054 | 1501 | — | — |
| udfs | isalsr | III.19.51 | 0.1258 | 1.3622 | 2485 | 1501 | — | — |
| udfs | isalsr | III.4.32 | 0.4253 | 1.3592 | 2514 | 1500 | — | — |
| udfs | isalsr | test_4 | 0.5660 | 1.3591 | 2506 | 1500 | — | — |
| udfs | isalsr | Strogatz-bacres1 | 0.3359 | 2.1886 | 6962 | 1500 | — | — |
| udfs | isalsr | Strogatz-bacres2 | 0.8800 | 2.1833 | 6993 | 1500 | — | — |
| udfs | isalsr | Strogatz-barmag1 | 0.7891 | 2.1832 | 6994 | 1500 | — | — |
| udfs | isalsr | Strogatz-barmag2 | 0.8181 | 2.1860 | 6977 | 1500 | — | — |
| udfs | isalsr | Strogatz-glider1 | 0.8610 | 2.1853 | 5991 | 1500 | — | — |
| udfs | isalsr | Strogatz-glider2 | 0.8446 | 2.1787 | 6061 | 1500 | — | — |
| udfs | isalsr | Strogatz-lv1 | 0.8810 | 2.1701 | 6115 | 1500 | — | — |
| udfs | isalsr | Strogatz-lv2 | 0.6625 | 2.1755 | 6057 | 1500 | — | — |
| udfs | isalsr | Strogatz-predprey1 | 0.8691 | 2.1745 | 6063 | 1500 | — | — |
| udfs | isalsr | Strogatz-predprey2 | 0.7761 | 2.1753 | 6051 | 1500 | — | — |
| udfs | isalsr | Strogatz-shearflow1 | 0.4673 | 2.1881 | 6019 | 1500 | — | — |
| udfs | isalsr | Strogatz-shearflow2 | 0.6197 | 2.2018 | 6403 | 1500 | — | — |
| udfs | isalsr | Strogatz-vdp1 | 0.1275 | 2.1804 | 6103 | 1500 | — | — |
| udfs | isalsr | Strogatz-vdp2 | 1.0000 | 1.1429 | 35 | 7 | — | — |


## Aggregate

- runs: **40**  ({'bingo/isalsr': 20, 'udfs/isalsr': 20})
- cells with NaN/inf or unparsable: **0**
- dedup-arm cells with ρ ≤ 1 (C1.6): **0** 
- baseline cells with ρ ≠ 1 (C1.8): **0** 
- ρ bingo (dedup arms): mean **1.7572**, range [1.1710, 1.8057]
- ρ udfs (dedup arms): mean **1.8846**, range [1.1429, 2.2018]
