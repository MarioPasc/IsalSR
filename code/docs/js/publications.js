/**
 * IsalSR — Publications page interactivity
 * BibTeX toggle and clipboard copy.
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  IsalSR.toggleBibtex = function (id) {
    var el = document.getElementById(id);
    if (el) {
      el.classList.toggle('open');
    }
  };

  IsalSR.copyBibtex = function (id) {
    var el = document.getElementById(id);
    if (!el) return;
    var pre = el.querySelector('pre');
    if (!pre) return;

    var text = pre.textContent;
    navigator.clipboard.writeText(text).then(function () {
      var feedback = document.getElementById(id + '-feedback');
      if (feedback) {
        feedback.textContent = 'Copied!';
        feedback.classList.add('show');
        setTimeout(function () {
          feedback.classList.remove('show');
          feedback.textContent = '';
        }, 2000);
      }
    }).catch(function () {
      var range = document.createRange();
      range.selectNodeContents(pre);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    });
  };
})();
