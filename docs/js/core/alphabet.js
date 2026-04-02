/**
 * IsalSR — Token Alphabet and Tokenizer
 *
 * Defines the 31-token IsalSR alphabet, label registry, and tokenizer.
 * Two-tier encoding: single-char movement/edge tokens + two-char compound tokens.
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
    'C': { category: 'edge', description: 'Edge primary\u2192secondary', css: 'token-edge' },
    'c': { category: 'edge', description: 'Edge secondary\u2192primary', css: 'token-edge' },
    'W': { category: 'noop', description: 'No-op', css: 'token-noop' }
  };

  // Label characters for compound tokens (V/v + label)
  var LABEL_CHARS = {
    '+': { type: 'ADD',   arity: 'variadic', display: '+',    css: 'token-variadic' },
    '*': { type: 'MUL',   arity: 'variadic', display: '\u00D7', css: 'token-variadic' },
    '-': { type: 'SUB',   arity: 'binary',   display: '\u2212', css: 'token-binary' },
    '/': { type: 'DIV',   arity: 'binary',   display: '\u00F7', css: 'token-binary' },
    '^': { type: 'POW',   arity: 'binary',   display: '^',    css: 'token-binary' },
    's': { type: 'SIN',   arity: 'unary',    display: 'sin',  css: 'token-unary' },
    'c': { type: 'COS',   arity: 'unary',    display: 'cos',  css: 'token-unary' },
    'e': { type: 'EXP',   arity: 'unary',    display: 'exp',  css: 'token-unary' },
    'l': { type: 'LOG',   arity: 'unary',    display: 'log',  css: 'token-unary' },
    'r': { type: 'SQRT',  arity: 'unary',    display: '\u221A', css: 'token-unary' },
    'a': { type: 'ABS',   arity: 'unary',    display: '|x|',  css: 'token-unary' },
    'g': { type: 'NEG',   arity: 'unary',    display: '\u2212x', css: 'token-unary' },
    'i': { type: 'INV',   arity: 'unary',    display: '1/x',  css: 'token-unary' },
    'k': { type: 'CONST', arity: 'leaf',     display: 'k',    css: 'token-leaf' }
  };

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

  // Expose registries
  IsalSR.SINGLE_TOKENS = SINGLE_TOKENS;
  IsalSR.LABEL_CHARS = LABEL_CHARS;
})();
