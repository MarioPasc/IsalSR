/**
 * IsalSR — Step-by-Step Player
 *
 * Transport controls for stepping through S2D traces.
 * Adapted from IsalGraph with compound token support.
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  /**
   * Create a step player bound to UI elements.
   * @param {Object} config
   * @param {Array} config.traceSteps - Array of trace step objects from S2D
   * @param {string} config.dagSvgId - SVG ID for DAG rendering
   * @param {string} config.stepInfoId - ID for step counter text
   * @param {string} config.stepLogId - ID for step log container
   * @param {string} config.stringDisplayId - ID for string display container
   * @param {Array} config.tokens - Parsed token array from tokenize()
   */
  IsalSR.StepPlayer = function (config) {
    this.steps = config.traceSteps || [];
    this.dagSvgId = config.dagSvgId;
    this.stepInfoId = config.stepInfoId;
    this.stepLogId = config.stepLogId;
    this.stringDisplayId = config.stringDisplayId;
    this.tokens = config.tokens || [];
    this.currentStep = 0;
    this.playing = false;
    this.timer = null;
    this.speed = 800;
  };

  IsalSR.StepPlayer.prototype.totalSteps = function () {
    return this.steps.length;
  };

  IsalSR.StepPlayer.prototype.goToStep = function (idx) {
    if (idx < 0) idx = 0;
    if (idx >= this.steps.length) idx = this.steps.length - 1;
    this.currentStep = idx;
    this.renderCurrentStep();
  };

  IsalSR.StepPlayer.prototype.next = function () {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
      this.renderCurrentStep();
    } else {
      this.pause();
    }
  };

  IsalSR.StepPlayer.prototype.prev = function () {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.renderCurrentStep();
    }
  };

  IsalSR.StepPlayer.prototype.play = function () {
    if (this.playing) return;
    this.playing = true;
    var self = this;
    this.timer = setInterval(function () {
      if (self.currentStep < self.steps.length - 1) {
        self.next();
      } else {
        self.pause();
      }
    }, this.speed);
  };

  IsalSR.StepPlayer.prototype.pause = function () {
    this.playing = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  };

  IsalSR.StepPlayer.prototype.reset = function () {
    this.pause();
    this.currentStep = 0;
    this.renderCurrentStep();
  };

  IsalSR.StepPlayer.prototype.setSpeed = function (ms) {
    this.speed = ms;
    if (this.playing) {
      this.pause();
      this.play();
    }
  };

  IsalSR.StepPlayer.prototype.renderCurrentStep = function () {
    var step = this.steps[this.currentStep];
    if (!step) return;

    // Update step counter
    var infoEl = document.getElementById(this.stepInfoId);
    if (infoEl) {
      infoEl.textContent = 'Step ' + this.currentStep + ' / ' + (this.steps.length - 1);
    }

    // Render DAG
    if (step.dag && typeof IsalSR.renderDAG === 'function') {
      var d3Data = step.dag.toD3Data();
      var cdll = step.cdll;
      IsalSR.renderDAG(this.dagSvgId, d3Data, {
        primaryNode: cdll.size() > 0 ? cdll.getValue(step.primaryPtr) : undefined,
        secondaryNode: cdll.size() > 0 ? cdll.getValue(step.secondaryPtr) : undefined
      });
    }

    // Render token string display with current position
    var strEl = document.getElementById(this.stringDisplayId);
    if (strEl && this.tokens.length > 0) {
      var html = '';
      for (var i = 0; i < this.tokens.length; i++) {
        var tok = this.tokens[i];
        var css = tok.css || 'token-noop';
        var cls = 'string-display__char ' + css;
        if (i < this.currentStep - 1) cls += ' done';
        if (i === this.currentStep - 1) cls += ' current';
        html += '<span class="' + cls + '">' + tok.token + '</span>';
        if (i < this.tokens.length - 1) html += ' ';
      }
      strEl.innerHTML = html;
    }

    // Render step log
    var logEl = document.getElementById(this.stepLogId);
    if (logEl) {
      var logHtml = '';
      for (var s = 0; s <= this.currentStep && s < this.steps.length; s++) {
        var entry = this.steps[s];
        var tokenStr = '';
        if (s > 0 && this.tokens[s - 1]) {
          var t = this.tokens[s - 1];
          tokenStr = '<span class="' + (t.css || '') + '" style="font-weight:700;">' + t.token + '</span> \u2014 ';
        }
        var active = s === this.currentStep ? ' active' : '';
        logHtml += '<div class="step-log__entry' + active + '">';
        if (s === 0) {
          logHtml += entry.description;
        } else {
          logHtml += tokenStr + entry.description;
        }
        logHtml += '</div>';
      }
      logEl.innerHTML = logHtml;
      logEl.scrollTop = logEl.scrollHeight;
    }

    // Update transport button states
    this._updateTransportUI();
  };

  IsalSR.StepPlayer.prototype._updateTransportUI = function () {
    var prevBtn = document.getElementById('sp-prev');
    var nextBtn = document.getElementById('sp-next');
    var playBtn = document.getElementById('sp-play');

    if (prevBtn) prevBtn.disabled = this.currentStep <= 0;
    if (nextBtn) nextBtn.disabled = this.currentStep >= this.steps.length - 1;
    if (playBtn) {
      playBtn.textContent = this.playing ? '\u23F8' : '\u25B6';
      playBtn.title = this.playing ? 'Pause' : 'Play';
    }
  };
})();
