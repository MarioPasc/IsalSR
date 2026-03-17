Which token encoding should we use for node-insertion instructions (V/v + label)?                                                                                             
                                                                                                                                                                                              
❯ 1. Two-char compound            ┌──────────────────────────────────────────┐
    (Recommended)                 │ Alphabet Sigma_SR (21 tokens):           │
  2. Single-char flat alphabet    │                                          │
                                  │ Movement (7 single-char):                │
                                  │   N P n p C c W                          │
                                  │                                          │
                                  │ Labeled insertion (14 two-char):         │
                                  │   V+  V*  Vs  Vc  Ve  Vl  Vk             │
                                  │   v+  v*  vs  vc  ve  vl  vk             │
                                  │                                          │
                                  │ Example string for sin(x+y):             │
                                  │   'V+NnncVsC'                            │
                                  │                                          │
                                  │ Tokenization: V/v consume next char      │
                                  │ as label; all others are single-char.    │
                                  └──────────────────────────────────────────┘
  1. Two-char compound            ┌──────────────────────────────────────────┐
    (Recommended)                 │ Alphabet Sigma_SR (21 single-char):      │
❯ 2. Single-char flat alphabet    │                                          │
                                  │ Movement:  N P n p C c W                 │
                                  │ Primary:   A(+) M() S(sin) O(cos)        │
                                  │            E(exp) L(log) K(const)        │
                                  │ Secondary: a(+) m() s(sin) o(cos)        │
                                  │            e(exp) l(log) k(const)        │
                                  │                                          │
                                  │ Example string for sin(x+y):             │
                                  │   'ANnncSC'                              │
                                  │                                          │
                                  │ Tokenization: every char is one token.   │
                                  └──────────────────────────────────────────┘
