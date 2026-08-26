/**
 * IsalSR — Theme toggle (dark/light mode)
 * Persists preference in localStorage, respects prefers-color-scheme.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'isalsr-theme';

  function getPreferred() {
    var stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      return 'light';
    }
    return 'dark';
  }

  function apply(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
  }

  apply(getPreferred());

  window.IsalSR = window.IsalSR || {};
  window.IsalSR.toggleTheme = function () {
    var current = document.documentElement.getAttribute('data-theme') || 'dark';
    apply(current === 'dark' ? 'light' : 'dark');
  };
})();
