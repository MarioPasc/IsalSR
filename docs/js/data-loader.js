/**
 * IsalSR — CSV Data Loader
 *
 * Fetches team.csv and publications.csv, parses them (RFC 4180),
 * and dynamically renders the Team and Publications pages.
 *
 * CSV order convention:
 *  - team.csv: display order matches CSV row order (first row = first card)
 *  - publications.csv: oldest first in CSV, displayed newest first (reversed)
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  // ================================================================
  // RFC 4180 CSV Parser
  // ================================================================

  function parseCSV(text) {
    var rows = [];
    var row = [];
    var field = '';
    var inQuotes = false;
    var i = 0;
    var len = text.length;

    while (i < len) {
      var ch = text[i];

      if (inQuotes) {
        if (ch === '"') {
          if (i + 1 < len && text[i + 1] === '"') {
            field += '"';
            i += 2;
          } else {
            inQuotes = false;
            i++;
          }
        } else {
          field += ch;
          i++;
        }
      } else {
        if (ch === '"') {
          inQuotes = true;
          i++;
        } else if (ch === ',') {
          row.push(field);
          field = '';
          i++;
        } else if (ch === '\r') {
          i++;
        } else if (ch === '\n') {
          row.push(field);
          field = '';
          rows.push(row);
          row = [];
          i++;
        } else {
          field += ch;
          i++;
        }
      }
    }

    if (field || row.length > 0) {
      row.push(field);
      rows.push(row);
    }

    while (rows.length > 0 && rows[rows.length - 1].length === 1 && rows[rows.length - 1][0] === '') {
      rows.pop();
    }

    return rows;
  }

  function csvToObjects(text) {
    var rows = parseCSV(text);
    if (rows.length < 2) return [];

    var headers = rows[0].map(function (h) { return h.trim(); });
    var objects = [];

    for (var i = 1; i < rows.length; i++) {
      var obj = {};
      for (var j = 0; j < headers.length; j++) {
        obj[headers[j]] = (j < rows[i].length) ? rows[i][j].trim() : '';
      }
      objects.push(obj);
    }
    return objects;
  }

  // ================================================================
  // Team Page Renderer
  // ================================================================

  function renderTeamPage(members) {
    var container = document.getElementById('team-grid-container');
    if (!container) return;

    var html = '';
    for (var i = 0; i < members.length; i++) {
      var m = members[i];
      html += '<article class="card profile-card">';

      if (m.photo) {
        html += '<img src="' + escapeAttr(m.photo) + '" alt="Photo of ' + escapeHTML(m.name) + '" class="profile-card__photo" width="140" height="140">';
      }

      html += '<h2 class="profile-card__name">' + escapeHTML(m.name) + '</h2>';

      if (m.role) {
        html += '<p class="profile-card__role">' + escapeHTML(m.role) + '</p>';
      }

      if (m.title) {
        var affiliationParts = m.title.split(',').map(function (s) { return s.trim(); });
        html += '<p class="profile-card__affiliation">' + affiliationParts.map(escapeHTML).join('<br>') + '</p>';
      }

      if (m.bio) {
        html += '<p class="profile-card__bio">' + escapeHTML(m.bio) + '</p>';
      }

      html += '<div class="profile-card__links">';

      if (m.email) {
        html += '<a href="mailto:' + escapeAttr(m.email) + '" class="btn btn-sm btn-ghost">Email</a>';
      }
      if (m.scholar) {
        html += '<a href="' + escapeAttr(m.scholar) + '" class="btn btn-sm btn-ghost" target="_blank" rel="noopener noreferrer">Scholar</a>';
      }
      if (m.github) {
        html += '<a href="' + escapeAttr(m.github) + '" class="btn btn-sm btn-ghost" target="_blank" rel="noopener noreferrer">GitHub</a>';
      }
      if (m.orcid) {
        html += '<a href="https://orcid.org/' + escapeAttr(m.orcid) + '" class="btn btn-sm btn-ghost orcid-link" target="_blank" rel="noopener noreferrer">';
        html += '<img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" alt="" width="16" height="16" class="orcid-icon">';
        html += ' ORCID</a>';
      }

      html += '</div>';
      html += '</article>';
    }

    container.innerHTML = html;
  }

  // ================================================================
  // Publications Page Renderer
  // ================================================================

  function renderPublicationsPage(papers) {
    var container = document.getElementById('paper-list-container');
    if (!container) return;

    var reversed = papers.slice().reverse();
    var schemaData = [];

    var html = '';
    for (var i = 0; i < reversed.length; i++) {
      var p = reversed[i];
      var bibId = p.id || ('bib-' + i);

      var yearBadgeColor = 'blue';
      if (parseInt(p.year, 10) <= new Date().getFullYear()) {
        yearBadgeColor = 'purple';
      }

      var typeBadgeColor = p.badge_color || 'green';

      html += '<article class="card paper-card">';
      html += '<div class="paper-card__meta">';
      html += '<span class="badge badge--' + yearBadgeColor + '">' + escapeHTML(p.year) + '</span>';
      html += '<span class="badge badge--' + escapeAttr(typeBadgeColor) + '">' + escapeHTML(p.type) + '</span>';
      html += '</div>';

      html += '<h2 class="paper-card__title">' + escapeHTML(p.title) + '</h2>';
      html += '<p class="paper-card__authors">' + escapeHTML(p.authors) + '</p>';

      if (p.abstract) {
        html += '<div class="paper-card__abstract"><p>' + escapeHTML(p.abstract) + '</p></div>';
      }

      html += '<div class="paper-card__actions">';
      if (p.arxiv) {
        html += '<a href="' + escapeAttr(p.arxiv) + '" class="btn btn-sm btn-primary" target="_blank" rel="noopener noreferrer">arXiv</a>';
      }
      if (p.doi) {
        html += '<a href="' + escapeAttr(p.doi) + '" class="btn btn-sm btn-primary" target="_blank" rel="noopener noreferrer">DOI</a>';
      }
      if (p.pdf) {
        html += '<a href="' + escapeAttr(p.pdf) + '" class="btn btn-sm btn-outline" target="_blank" rel="noopener noreferrer">PDF</a>';
      }
      if (p.bibtex) {
        html += '<button class="btn btn-sm btn-ghost" onclick="IsalSR.toggleBibtex(\'' + escapeAttr(bibId) + '\')">';
        html += 'BibTeX <span id="' + escapeAttr(bibId) + '-feedback" class="copy-feedback"></span>';
        html += '</button>';
      }
      html += '</div>';

      if (p.bibtex) {
        html += '<div id="' + escapeAttr(bibId) + '" class="bibtex-content expandable">';
        html += '<div class="expandable__content" style="display: block;">';
        html += '<pre>' + escapeHTML(p.bibtex) + '</pre>';
        html += '<button class="btn btn-sm btn-outline" onclick="IsalSR.copyBibtex(\'' + escapeAttr(bibId) + '\')" style="margin-top: var(--space-sm);">';
        html += 'Copy to clipboard</button>';
        html += '</div></div>';
      }

      html += '</article>';

      var schemaEntry = {
        '@context': 'https://schema.org',
        '@type': 'ScholarlyArticle',
        'name': p.title,
        'author': p.authors.split(',').map(function (a) {
          return { '@type': 'Person', 'name': a.trim() };
        }),
        'datePublished': p.year
      };
      if (p.arxiv) schemaEntry.url = p.arxiv;
      else if (p.doi) schemaEntry.url = p.doi;
      schemaData.push(schemaEntry);
    }

    container.innerHTML = html;

    var existingSchema = document.getElementById('schema-publications');
    if (existingSchema) existingSchema.remove();
    var script = document.createElement('script');
    script.type = 'application/ld+json';
    script.id = 'schema-publications';
    script.textContent = JSON.stringify(schemaData);
    document.head.appendChild(script);
  }

  // ================================================================
  // HTML escaping utilities
  // ================================================================

  function escapeHTML(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function escapeAttr(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // ================================================================
  // Data fetching
  // ================================================================

  IsalSR.loadTeamData = function () {
    fetch('data/team.csv')
      .then(function (response) {
        if (!response.ok) throw new Error('Failed to load team.csv');
        return response.text();
      })
      .then(function (text) {
        var members = csvToObjects(text);
        renderTeamPage(members);
      })
      .catch(function (err) {
        var container = document.getElementById('team-grid-container');
        if (container) {
          container.innerHTML = '<p style="color: #ef4444;">Error loading team data: ' + escapeHTML(err.message) + '</p>';
        }
      });
  };

  IsalSR.loadPublicationsData = function () {
    fetch('data/publications.csv')
      .then(function (response) {
        if (!response.ok) throw new Error('Failed to load publications.csv');
        return response.text();
      })
      .then(function (text) {
        var papers = csvToObjects(text);
        renderPublicationsPage(papers);
      })
      .catch(function (err) {
        var container = document.getElementById('paper-list-container');
        if (container) {
          container.innerHTML = '<p style="color: #ef4444;">Error loading publications data: ' + escapeHTML(err.message) + '</p>';
        }
      });
  };

  IsalSR._parseCSV = parseCSV;
  IsalSR._csvToObjects = csvToObjects;
})();
