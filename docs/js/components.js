/**
 * IsalSR — Shared nav + footer injection
 * Reads body[data-page] for active link highlighting.
 * Navigation: Learn (How It Works, Math, Results) | Publications | Interactive (Playground) | Team
 */
(function () {
  'use strict';

  var page = document.body.getAttribute('data-page') || '';

  function activeClass(id) {
    return id === page ? ' active' : '';
  }

  function mobileActiveClass(id) {
    return id === page ? ' active' : '';
  }

  function dropdownActive(pages) {
    for (var i = 0; i < pages.length; i++) {
      if (pages[i] === page) return ' active';
    }
    return '';
  }

  // ---- Navigation ----
  var NAV_HTML =
    '<nav class="site-nav" role="navigation" aria-label="Main navigation">' +
      '<div class="site-nav__inner">' +
        '<a href="index.html" class="site-nav__logo" aria-label="IsalSR Home">' +
          '<svg class="site-nav__logo-icon" viewBox="0 0 32 32" aria-hidden="true">' +
            '<circle cx="22" cy="10" r="6" fill="#8b5cf6" opacity="0.9"/>' +
            '<text x="22" y="10" text-anchor="middle" dominant-baseline="central" font-family="monospace" font-size="8" font-weight="bold" fill="#fff">+</text>' +
            '<circle cx="10" cy="22" r="4.5" fill="#f59e0b" opacity="0.9"/>' +
            '<text x="10" y="22" text-anchor="middle" dominant-baseline="central" font-family="monospace" font-size="6" font-weight="bold" fill="#fff">x</text>' +
            '<line x1="13" y1="19" x2="18.5" y2="13.5" stroke="#10b981" stroke-width="2" stroke-linecap="round"/>' +
            '<polygon points="19,13 17,15 20,14.5" fill="#10b981"/>' +
          '</svg>' +
          '<span>IsalSR</span>' +
        '</a>' +
        '<div class="site-nav__links">' +
          // Learn dropdown
          '<div class="site-nav__dropdown">' +
            '<button class="site-nav__dropdown-toggle' + dropdownActive(['how-it-works', 'math', 'results']) + '" aria-haspopup="true" aria-expanded="false">' +
              'Learn <svg class="site-nav__chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5L6 7.5L9 4.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
            '</button>' +
            '<div class="site-nav__dropdown-menu">' +
              '<a href="how-it-works.html" class="site-nav__dropdown-item' + activeClass('how-it-works') + '">How It Works</a>' +
              '<a href="math.html" class="site-nav__dropdown-item' + activeClass('math') + '">Math Foundations</a>' +
              '<a href="results.html" class="site-nav__dropdown-item' + activeClass('results') + '">Results</a>' +
            '</div>' +
          '</div>' +
          '<a href="publications.html" class="site-nav__link' + activeClass('publications') + '">Publications</a>' +
          // Interactive dropdown
          '<div class="site-nav__dropdown">' +
            '<button class="site-nav__dropdown-toggle' + dropdownActive(['playground']) + '" aria-haspopup="true" aria-expanded="false">' +
              'Interactive <svg class="site-nav__chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5L6 7.5L9 4.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
            '</button>' +
            '<div class="site-nav__dropdown-menu">' +
              '<a href="playground.html" class="site-nav__dropdown-item' + activeClass('playground') + '">Playground</a>' +
            '</div>' +
          '</div>' +
          '<a href="team.html" class="site-nav__link' + activeClass('team') + '">Team</a>' +
        '</div>' +
        '<div class="site-nav__actions">' +
          '<button class="theme-toggle" onclick="IsalSR.toggleTheme()" aria-label="Toggle theme">' +
            '<svg class="theme-toggle__icon--dark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
              '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>' +
            '</svg>' +
            '<svg class="theme-toggle__icon--light" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
              '<circle cx="12" cy="12" r="5"/>' +
              '<line x1="12" y1="1" x2="12" y2="3"/>' +
              '<line x1="12" y1="21" x2="12" y2="23"/>' +
              '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>' +
              '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>' +
              '<line x1="1" y1="12" x2="3" y2="12"/>' +
              '<line x1="21" y1="12" x2="23" y2="12"/>' +
              '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>' +
              '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>' +
            '</svg>' +
          '</button>' +
          '<a href="https://github.com/MarioPasc/IsalSR" class="site-nav__github" target="_blank" rel="noopener noreferrer" aria-label="GitHub Repository">' +
            '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">' +
              '<path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>' +
            '</svg>' +
          '</a>' +
          '<button class="hamburger" onclick="IsalSR.toggleMobile()" aria-label="Toggle menu" aria-expanded="false">' +
            '<span class="hamburger__line"></span>' +
            '<span class="hamburger__line"></span>' +
            '<span class="hamburger__line"></span>' +
          '</button>' +
        '</div>' +
      '</div>' +
    '</nav>' +
    '<nav class="mobile-menu" id="mobile-menu" role="navigation" aria-label="Mobile navigation">' +
      '<a href="index.html" class="mobile-menu__link' + mobileActiveClass('home') + '">Home</a>' +
      '<a href="how-it-works.html" class="mobile-menu__link' + mobileActiveClass('how-it-works') + '">How It Works</a>' +
      '<a href="math.html" class="mobile-menu__link' + mobileActiveClass('math') + '">Math Foundations</a>' +
      '<a href="results.html" class="mobile-menu__link' + mobileActiveClass('results') + '">Results</a>' +
      '<a href="publications.html" class="mobile-menu__link' + mobileActiveClass('publications') + '">Publications</a>' +
      '<a href="playground.html" class="mobile-menu__link' + mobileActiveClass('playground') + '">Playground</a>' +
      '<a href="team.html" class="mobile-menu__link' + mobileActiveClass('team') + '">Team</a>' +
    '</nav>';

  // ---- Footer ----
  var FOOTER_HTML =
    '<footer class="site-footer">' +
      '<div class="site-footer__inner">' +
        '<div class="site-footer__grid">' +
          '<div class="site-footer__about">' +
            '<h3 class="site-footer__title">IsalSR</h3>' +
            '<p class="site-footer__text">' +
              'Instruction Set and Language for Symbolic Regression. ' +
              'Isomorphism-invariant canonical string representations for expression DAGs.' +
            '</p>' +
          '</div>' +
          '<div class="site-footer__links">' +
            '<h4 class="site-footer__heading">Learn</h4>' +
            '<a href="how-it-works.html">How It Works</a>' +
            '<a href="math.html">Math Foundations</a>' +
            '<a href="results.html">Results</a>' +
          '</div>' +
          '<div class="site-footer__links">' +
            '<h4 class="site-footer__heading">Resources</h4>' +
            '<a href="publications.html">Publications</a>' +
            '<a href="playground.html">Playground</a>' +
            '<a href="https://github.com/MarioPasc/IsalSR" target="_blank" rel="noopener noreferrer">GitHub</a>' +
          '</div>' +
          '<div class="site-footer__links">' +
            '<h4 class="site-footer__heading">Contact</h4>' +
            '<a href="mailto:ezeqlr@lcc.uma.es">ezeqlr@lcc.uma.es</a>' +
            '<a href="mailto:mpascual@uma.es">mpascual@uma.es</a>' +
            '<a href="team.html">Team</a>' +
          '</div>' +
        '</div>' +
        '<div class="site-footer__bottom">' +
          '<p>&copy; 2025&ndash;2026 ICAI Research Group, University of M&aacute;laga. Built for science.</p>' +
        '</div>' +
      '</div>' +
    '</footer>';

  // ---- Injection ----
  function inject() {
    var navTarget = document.getElementById('site-nav');
    if (navTarget) {
      navTarget.innerHTML = NAV_HTML;
    }

    var footerTarget = document.getElementById('site-footer');
    if (footerTarget) {
      footerTarget.innerHTML = FOOTER_HTML;
    }

    // Dropdown hover/focus behaviour
    var dropdowns = document.querySelectorAll('.site-nav__dropdown');
    for (var i = 0; i < dropdowns.length; i++) {
      (function (dropdown) {
        var toggle = dropdown.querySelector('.site-nav__dropdown-toggle');
        var menu = dropdown.querySelector('.site-nav__dropdown-menu');
        if (!toggle || !menu) return;

        dropdown.addEventListener('mouseenter', function () {
          toggle.setAttribute('aria-expanded', 'true');
          menu.style.display = 'block';
        });
        dropdown.addEventListener('mouseleave', function () {
          toggle.setAttribute('aria-expanded', 'false');
          menu.style.display = '';
        });
        toggle.addEventListener('click', function (e) {
          e.preventDefault();
          var expanded = toggle.getAttribute('aria-expanded') === 'true';
          toggle.setAttribute('aria-expanded', !expanded);
          menu.style.display = expanded ? '' : 'block';
        });
      })(dropdowns[i]);
    }
  }

  // ---- Mobile menu toggle ----
  window.IsalSR = window.IsalSR || {};
  IsalSR.toggleMobile = function () {
    var menu = document.getElementById('mobile-menu');
    var btn = document.querySelector('.hamburger');
    if (!menu) return;
    var isOpen = menu.classList.toggle('open');
    if (btn) {
      btn.classList.toggle('open', isOpen);
      btn.setAttribute('aria-expanded', isOpen);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', inject);
  } else {
    inject();
  }
})();
