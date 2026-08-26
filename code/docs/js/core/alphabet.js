/**
 * IsalSR — Token Alphabet and Tokenizer
 *
 * The alphabet Sigma_SR of Definition 3.2 of the paper: 7 single-character
 * tokens and 2 x 12 = 24 compound tokens, 31 in total.
 *
 * The label set L carries one label per mathematical operation, so subtraction
 * and division get none: they are emitted as Add(x, Neg(y)) and Mul(x, Inv(y))
 * at the conversion boundary (Section 3.1), which leaves Pow as the only
 * non-commutative operation.
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  // Single-character tokens
  var SINGLE_TOKENS = {
    'N': { category: 'movement', description: 'Primary pointer next', css: 'token-movement-fwd' },
    'P': { category: 'movement', description: 'Primary pointer prev', css: 'token-movement-bwd' },
    'n': { category: 'movement', description: 'Secondary pointer next', css: 'token-movement-fwd' },
    'p': { category: 'movement', description: 'Secondary pointer prev', css: 'token-movement-bwd' },
    'C': { category: 'edge', description: 'Edge primary→secondary', css: 'token-edge' },
    'c': { category: 'edge', description: 'Edge secondary→primary', css: 'token-edge' },
    'W': { category: 'noop', description: 'No-op', css: 'token-noop' }
  };

  // The twelve label characters of L (Table 1 of the paper), in the order the
  // paper lists them. `guarded` marks the numerically protected evaluations.
  var LABEL_CHARS = {
    '+': { type: 'ADD',   arity: 'variadic', display: '+',      css: 'token-variadic', guarded: false },
    '*': { type: 'MUL',   arity: 'variadic', display: '×', css: 'token-variadic', guarded: false },
    'g': { type: 'NEG',   arity: 'unary',    display: '−x', css: 'token-unary',   guarded: false },
    'i': { type: 'INV',   arity: 'unary',    display: '1/x',    css: 'token-unary',    guarded: true },
    's': { type: 'SIN',   arity: 'unary',    display: 'sin',    css: 'token-unary',    guarded: false },
    'c': { type: 'COS',   arity: 'unary',    display: 'cos',    css: 'token-unary',    guarded: false },
    'e': { type: 'EXP',   arity: 'unary',    display: 'exp',    css: 'token-unary',    guarded: true },
    'l': { type: 'LOG',   arity: 'unary',    display: 'log',    css: 'token-unary',    guarded: true },
    'r': { type: 'SQRT',  arity: 'unary',    display: '√', css: 'token-unary',    guarded: true },
    'a': { type: 'ABS',   arity: 'unary',    display: '|x|',    css: 'token-unary',    guarded: false },
    '^': { type: 'POW',   arity: 'binary',   display: '^',      css: 'token-binary',   guarded: true },
    'k': { type: 'CONST', arity: 'leaf',     display: 'k',      css: 'token-leaf',     guarded: false }
  };

  // Labels a reader might reach for that L deliberately does not carry.
  var DECOMPOSED_LABELS = {
    '-': 'Σ_SR has no Sub label — write x − y as Add(x, Neg(y)), i.e. Vg then an edge.',
    '/': 'Σ_SR has no Div label — write x / y as Mul(x, Inv(y)), i.e. Vi then an edge.'
  };

  // Pow is the only operation whose operands must be ordered (Table 1).
  var ORDERED_BINARY = { 'POW': true };

  /**
   * Tokenize an IsalSR instruction string.
   * @param {string} str - Raw instruction string.
   * @returns {Array<{token: string, category: string, css: string, label?: Object}>}
   * @throws {Error} on invalid characters.
   */
  IsalSR.tokenize = function (str) {
    var tokens = [];
    var i = 0;
    while (i < str.length) {
      var ch = str[i];
      if (ch === 'V' || ch === 'v') {
        if (i + 1 >= str.length) {
          throw new Error(ch + ' at position ' + i + ' missing label character.');
        }
        var labelCh = str[i + 1];
        var labelInfo = LABEL_CHARS[labelCh];
        if (!labelInfo) {
          if (DECOMPOSED_LABELS[labelCh]) {
            throw new Error(DECOMPOSED_LABELS[labelCh]);
          }
          throw new Error('Invalid label "' + labelCh + '" after ' + ch + ' at position ' + i + '.');
        }
        tokens.push({
          token: ch + labelCh,
          category: 'insert',
          pointer: ch === 'V' ? 'primary' : 'secondary',
          css: labelInfo.css,
          label: labelInfo
        });
        i += 2;
      } else if (SINGLE_TOKENS[ch]) {
        var info = SINGLE_TOKENS[ch];
        tokens.push({
          token: ch,
          category: info.category,
          css: info.css,
          description: info.description
        });
        i++;
      } else {
        throw new Error('Unknown character "' + ch + '" at position ' + i + '.');
      }
    }
    return tokens;
  };

  /**
   * Get the CSS class for a token string.
   */
  IsalSR.tokenCssClass = function (tok) {
    if (tok.length === 1 && SINGLE_TOKENS[tok]) return SINGLE_TOKENS[tok].css;
    if (tok.length === 2 && (tok[0] === 'V' || tok[0] === 'v')) {
      var lbl = LABEL_CHARS[tok[1]];
      if (lbl) return lbl.css;
    }
    return 'token-noop';
  };

  /**
   * Get label info for a label character.
   */
  IsalSR.getLabelInfo = function (labelCh) {
    return LABEL_CHARS[labelCh] || null;
  };

  /**
   * Label character of a NodeType, i.e. the NODE_TYPE_TO_LABEL table.
   */
  IsalSR.labelCharOf = function (nodeType) {
    for (var ch in LABEL_CHARS) {
      if (LABEL_CHARS[ch].type === nodeType) return ch;
    }
    return null;
  };

  // Expose registries
  IsalSR.SINGLE_TOKENS = SINGLE_TOKENS;
  IsalSR.LABEL_CHARS = LABEL_CHARS;
  IsalSR.ORDERED_BINARY = ORDERED_BINARY;
})();
