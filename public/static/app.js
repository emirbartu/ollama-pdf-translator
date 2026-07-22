/*  Shared module for Ollama PDF Translator
 *  Handles: pdf.js validation, dropzone UX, NDJSON streaming, copy-btn feedback.
 *  Safe to include on any page — form wiring is guarded.
 */

import * as pdfjsLib from 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.7.76/build/pdf.min.mjs';
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.7.76/build/pdf.worker.min.mjs';

const MAX_BYTES = 1 * 1024 * 1024;
const MAX_PAGES = 3;

/* ------------------------------------------------------------------ */
/*  Copy buttons (works on every page)                                */
/* ------------------------------------------------------------------ */

function copyFeedback(btn) {
  var original = btn.textContent;
  btn.textContent = 'Copied';
  btn.classList.add('copied');
  setTimeout(function () {
    btn.textContent = original;
    btn.classList.remove('copied');
  }, 1500);
}

function wireCopyButtons() {
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var text = btn.getAttribute('data-copy');
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { copyFeedback(btn); });
      } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); copyFeedback(btn); } catch (_e) { /* noop */ }
        document.body.removeChild(ta);
      }
    });
  });
}

/* ------------------------------------------------------------------ */
/*  Translate-form logic (wired only when #upload-form exists)        */
/* ------------------------------------------------------------------ */

function initTranslateForm() {
  var form = document.getElementById('upload-form');
  if (!form) return;

  var dz = document.getElementById('dropzone');
  var input = document.getElementById('file-input');
  var dzText = document.getElementById('dz-text');
  var statusEl = document.getElementById('status');

  var dzDefaultText = dzText.innerHTML;

  function showName() {
    if (input.files.length) {
      dzText.textContent = input.files[0].name;
      dz.classList.add('has-file');
    }
  }

  function resetFile() {
    input.value = '';
    dzText.innerHTML = dzDefaultText;
    dz.classList.remove('has-file');
  }

  async function validateFile(file) {
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      return 'Only PDF files are supported.';
    }
    if (file.size > MAX_BYTES) {
      return 'File too large — 1 MB max.';
    }
    try {
      var doc = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
      var pages = doc.numPages;
      doc.destroy();
      if (pages > MAX_PAGES) return 'PDF too long — ' + MAX_PAGES + ' pages max for this demo.';
    } catch (_e) {
      return 'Could not read this PDF.';
    }
    return null;
  }

  async function checkSelected() {
    if (!input.files.length) return;
    var err = await validateFile(input.files[0]);
    if (err) {
      resetFile();
      showError(err);
    }
  }

  function showError(msg) {
    statusEl.innerHTML = '<div class="status error"><div class="status-row">' +
      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>' +
      '<span class="status-title"></span></div></div>';
    statusEl.querySelector('.status-title').textContent = msg;
  }

  function runningCard() {
    return '<div class="status running">' +
      '<div class="status-row"><span class="spinner"></span>' +
      '<span class="status-title">Translating ' + (input.files[0] ? input.files[0].name : 'PDF') + '</span></div>' +
      '<div class="progress"><div class="progress-fill" id="pfill" style="width: 0%"></div></div>' +
      '<div class="status-sub" id="psub">Starting…</div></div>';
  }

  function handleEvent(ev) {
    if (ev.type === 'progress') {
      var fill = document.getElementById('pfill');
      var sub = document.getElementById('psub');
      if (fill) fill.style.width = ev.percent + '%';
      if (sub) sub.textContent = 'Page ' + ev.page + ' of ' + ev.total + ' · ' + ev.percent + '%';
    } else if (ev.type === 'done') {
      var bin = atob(ev.pdf_b64);
      var bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      var url = URL.createObjectURL(new Blob([bytes], { type: 'application/pdf' }));
      statusEl.innerHTML = '<div class="status done"><div class="status-row">' +
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>' +
        '<span class="status-title">Translation complete</span></div>' +
        '<a class="download" id="dl" href="' + url + '" download="' + ev.filename + '">' +
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>' +
        'Download translated PDF</a></div>';
      document.getElementById('dl').click();
    } else if (ev.type === 'error') {
      showError(ev.message);
    }
  }

  async function startTranslation(e) {
    e.preventDefault();
    if (!input.files.length) return false;
    var validationError = await validateFile(input.files[0]);
    if (validationError) {
      resetFile();
      showError(validationError);
      return false;
    }
    statusEl.innerHTML = runningCard();
    try {
      var res = await fetch('/translate', { method: 'POST', body: new FormData(form) });
      if (res.status === 429) {
        showError('Rate limit reached — max 5 translations per hour. Please try again later.');
        return false;
      }
      if (!res.ok) {
        showError('Translation failed. Please try again.');
        return false;
      }
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';
      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
          var line = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 1);
          if (line) handleEvent(JSON.parse(line));
        }
      }
      var tail = buffer.trim();
      if (tail) handleEvent(JSON.parse(tail));
    } catch (_err) {
      showError('Network error — please try again.');
    }
    return false;
  }

  /* Wire events */
  input.addEventListener('change', function () { showName(); checkSelected(); });

  ['dragenter', 'dragover'].forEach(function (e) {
    dz.addEventListener(e, function (ev) { ev.preventDefault(); dz.classList.add('dragover'); });
  });
  ['dragleave', 'drop'].forEach(function (e) {
    dz.addEventListener(e, function (ev) { ev.preventDefault(); dz.classList.remove('dragover'); });
  });
  dz.addEventListener('drop', function (ev) {
    if (ev.dataTransfer.files.length) {
      input.files = ev.dataTransfer.files;
      showName();
      checkSelected();
    }
  });

  /* Expose to onsubmit attribute */
  window.startTranslation = startTranslation;
}

/* ------------------------------------------------------------------ */
/*  Boot                                                               */
/* ------------------------------------------------------------------ */

function init() {
  wireCopyButtons();
  initTranslateForm();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
