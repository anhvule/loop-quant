/* Stock Risk Outlook — renders the /api/predict payload.
   Design rule: ranges, odds and warnings lead; medians are always labelled as an
   assumption, never as a target. */
(function () {
  "use strict";

  var form = document.getElementById("form");
  var input = document.getElementById("symbols");
  var monthsSel = document.getElementById("months");
  var rfSel = document.getElementById("rf");
  var go = document.getElementById("go");
  var statusEl = document.getElementById("status");
  var out = document.getElementById("out");

  function money(v) {
    if (v == null || !isFinite(v)) return "—";
    var d = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 2 : 3;
    return "$" + v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function pct(x, dp) { return (x == null || !isFinite(x)) ? "—" : (x * 100).toFixed(dp == null ? 0 : dp) + "%"; }
  function signed(x) { return (x == null || !isFinite(x)) ? "—" : (x >= 0 ? "+" : "") + (x * 100).toFixed(0) + "%"; }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function el(html) { var d = document.createElement("div"); d.innerHTML = html.trim(); return d.firstChild; }

  // A dead server surfaces as a bare "Failed to fetch", which reads like a bug in the
  // page. Name the actual cause and the fix -- the usual reason is that serve.py is not
  // running (or was restarted while this tab stayed open).
  function netMsg(e, fallback) {
    var m = (e && e.message) ? e.message : "";
    if (/failed to fetch|networkerror|load failed|network request failed/i.test(m)) {
      return "Could not reach the server. Start it with " +
             "`python web/serve.py` in the web/ folder, then reload this page.";
    }
    return m || fallback;
  }

  // Rebuilds only when the text node is missing, so ticking the elapsed counter does
  // not restart the spinner animation every second. Clearing also wipes the text --
  // a hidden box holding a stale "Analysing…" is a trap the next error message.
  function setStatus(msg, isErr) {
    if (!msg) { statusEl.hidden = true; statusEl.textContent = ""; return; }
    var txt = statusEl.querySelector(".stxt");
    if (!txt) {
      statusEl.innerHTML =
        '<span class="spin" aria-hidden="true"></span><span class="stxt"></span>';
      txt = statusEl.querySelector(".stxt");
    }
    statusEl.className = "status" + (isErr ? " err" : "");
    statusEl.hidden = false;
    txt.textContent = msg;
  }

  /* ---- cone chart (inline SVG, no libraries) ---- */
  function cone(d) {
    var W = 720, H = 260, PL = 52, PR = 16, PT = 14, PB = 30;
    var pts = [{ d: 0, p5: d.spot, p50: d.spot, p95: d.spot, label: "now" }].concat(
      d.months.map(function (m) {
        return { d: m.trading_day, p5: m.p5, p50: m.p50, p95: m.p95, label: m.label.split(" ")[0] };
      }));
    var lo = Math.min.apply(null, pts.map(function (p) { return p.p5; }));
    var hi = Math.max.apply(null, pts.map(function (p) { return p.p95; }));
    lo = Math.max(lo * 0.9, 1e-6); hi = hi * 1.05;
    var maxD = pts[pts.length - 1].d || 1;
    var x = function (v) { return PL + (v / maxD) * (W - PL - PR); };
    var lg = function (v) { return Math.log(Math.max(v, 1e-6)); };
    var y = function (v) { return PT + (lg(hi) - lg(v)) / (lg(hi) - lg(lo)) * (H - PT - PB); };

    var up = pts.map(function (p) { return x(p.d) + "," + y(p.p95); }).join(" ");
    var down = pts.slice().reverse().map(function (p) { return x(p.d) + "," + y(p.p5); }).join(" ");
    var mid = pts.map(function (p, i) { return (i ? "L" : "M") + x(p.d) + " " + y(p.p50); }).join(" ");

    var ticks = [lo, Math.sqrt(lo * hi), hi].map(function (v) {
      return '<g><line x1="' + PL + '" y1="' + y(v) + '" x2="' + (W - PR) + '" y2="' + y(v) +
        '" stroke="var(--grid)"/><text x="' + (PL - 6) + '" y="' + (y(v) + 4) +
        '" text-anchor="end" font-size="10" fill="var(--axis-ink)">' + money(v) + "</text></g>";
    }).join("");

    var labels = pts.map(function (p, i) {
      if (!i) return "";
      return '<text x="' + x(p.d) + '" y="' + (H - 9) + '" text-anchor="middle" font-size="10" ' +
        'fill="var(--axis-ink)">' + esc(p.label) + "</text>";
    }).join("");

    // Hover layer: one crosshair per milestone. Milestones are few, so nearest-x is a
    // plain scan rather than a scale inversion.
    var hot = pts.map(function (p, i) {
      if (!i) return "";
      return '<rect class="hot" x="' + (x(p.d) - 16) + '" y="' + PT + '" width="32" ' +
        'height="' + (H - PT - PB) + '" fill="transparent" data-i="' + i + '"/>';
    }).join("");

    var svg = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Forecast range cone">' +
      ticks +
      '<polygon points="' + up + " " + down + '" fill="var(--band)" opacity=".14"/>' +
      '<path d="' + mid + '" fill="none" stroke="var(--band)" stroke-width="2"/>' +
      '<line x1="' + PL + '" y1="' + y(d.spot) + '" x2="' + (W - PR) + '" y2="' + y(d.spot) +
      '" stroke="var(--axis-ink)" stroke-dasharray="4 4"/>' +
      '<line class="xh" x1="0" y1="' + PT + '" x2="0" y2="' + (H - PB) +
      '" stroke="var(--axis-ink)" stroke-width="1" opacity="0"/>' +
      labels + hot + "</svg>";

    // Serialised alongside the markup so the hover handler needs no closure over `d`.
    conePoints = pts.map(function (p) {
      return { x: x(p.d), label: p.label, p5: p.p5, p50: p.p50, p95: p.p95 };
    });
    return '<div class="figure" id="cone-fig">' + svg + '<div class="tip"></div></div>';
  }

  var conePoints = [];

  /* Wire the cone crosshair after the SVG is in the document. */
  function wireCone(root) {
    var fig = root.querySelector("#cone-fig");
    if (!fig || !conePoints.length) return;
    var svg = fig.querySelector("svg");
    var tip = fig.querySelector(".tip");
    var xh = fig.querySelector(".xh");
    var pts = conePoints.slice();

    function show(i) {
      var p = pts[i];
      if (!p) return;
      xh.setAttribute("x1", p.x); xh.setAttribute("x2", p.x);
      xh.setAttribute("opacity", "1");
      tip.innerHTML = "<b>" + esc(p.label) + "</b>" +
        '<div class="r"><span>Best 5%</span><span>' + money(p.p95) + "</span></div>" +
        '<div class="r"><span>Median</span><span>' + money(p.p50) + "</span></div>" +
        '<div class="r"><span>Worst 5%</span><span>' + money(p.p5) + "</span></div>";
      // viewBox units -> CSS pixels, so the tooltip tracks the rendered width.
      var scale = fig.clientWidth / 720;
      var left = p.x * scale;
      tip.classList.add("on");
      var tw = tip.offsetWidth;
      tip.style.left = Math.max(4, Math.min(left - tw / 2, fig.clientWidth - tw - 4)) + "px";
      tip.style.top = "6px";
    }
    function hide() { tip.classList.remove("on"); xh.setAttribute("opacity", "0"); }

    Array.prototype.forEach.call(svg.querySelectorAll(".hot"), function (r) {
      var i = parseInt(r.getAttribute("data-i"), 10);
      r.addEventListener("mouseenter", function () { show(i); });
      r.addEventListener("focus", function () { show(i); });
    });
    svg.addEventListener("mouseleave", hide);
    svg.addEventListener("blur", hide, true);
  }

  /* ---- sections ---- */
  function metric(label, value, big) {
    return '<div class="metric"><span class="mlabel">' + esc(label) + "</span>" +
      '<span class="v' + (big ? " big" : "") + '">' + value + "</span></div>";
  }

  function header(d) {
    var w = d.warnings.map(function (t) {
      return '<div class="warn">' + esc(t) + "</div>";
    }).join("");
    return el('<section class="card">' +
      '<div class="head">' +
      "<div><h2>" + esc(d.symbol) + " — " + esc(d.info.name) + "</h2>" +
      '<p class="hint" style="margin-bottom:0">as of ' + esc(d.as_of) + " · " +
      d.bars + " shared daily bars</p></div>" +
      '<div class="metrics">' +
      metric("Last close", money(d.spot) + " " + esc(d.info.currency), true) +
      metric("Beta", d.beta.toFixed(2) + " ± " + d.beta_se.toFixed(2)) +
      metric("Daily vol", pct(d.vol_daily, 1)) +
      metric("Market r²", pct(d.r2)) +
      "</div></div>" + w + "</section>");
  }

  function tile(label, value, delta, cls) {
    var d = delta == null ? "" :
      '<span class="d ' + (delta >= 0 ? "pos" : "neg") + '">' + signed(delta) + "</span>";
    return '<div class="tile ' + (cls || "") + '"><span class="mlabel">' + esc(label) +
      '</span><span class="v">' + value + "</span>" + d + "</div>";
  }

  function headline(d) {
    var a = d.anchored_range, t = d.terminal, last = d.months[d.months.length - 1];
    return el('<section class="card"><h2>Where it could be by ' + esc(last.label) + "</h2>" +
      '<p class="hint">The <em>range</em> is the output. The median is what the model ' +
      "<em>assumes</em> — beta × market drift, never this stock's own momentum — and is " +
      "not a prediction.</p>" +
      '<div class="tiles">' +
      tile("Worst 5%", money(a.p5), a.p5 / d.spot - 1, "lo") +
      tile("Median (assumption)", money(a.median), a.median / d.spot - 1) +
      tile("Best 5%", money(a.p95), a.p95 / d.spot - 1, "hi") +
      tile("Chance above today", pct(t.p_up), null) +
      "</div>" + cone(d) + "</section>");
  }

  /* ---- wave chart (inline SVG): price window + labelled counts + key levels ---- */
  function waveChart(ch) {
    if (!ch || !ch.available || !ch.series || ch.series.length < 2) return "";
    var W = 760, H = 330, PL = 8, PR = 96, PT = 16, PB = 26;
    var xs = ch.series.map(function (s) { return s.i; });
    var ps = ch.series.map(function (s) { return s.p; });
    var counts = [ch.best, ch.rival].filter(Boolean);
    counts.forEach(function (c) { c.points.forEach(function (p) { ps.push(p.p); }); });
    (ch.levels || []).forEach(function (l) { ps.push(l.price); });

    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var lo = Math.min.apply(null, ps), hi = Math.max.apply(null, ps);
    var padv = (hi - lo) * 0.08 || 1;
    lo -= padv; hi += padv;
    var X = function (i) { return PL + ((i - x0) / Math.max(1, x1 - x0)) * (W - PL - PR); };
    var Y = function (p) { return PT + ((hi - p) / Math.max(1e-9, hi - lo)) * (H - PT - PB); };

    var out = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" ' +
      'aria-label="Elliott wave structure chart">';

    (ch.levels || []).forEach(function (l) {
      out += '<line x1="' + PL + '" y1="' + Y(l.price) + '" x2="' + (W - PR) + '" y2="' +
        Y(l.price) + '" stroke="var(--good)" stroke-opacity=".55" stroke-width="1"/>' +
        '<text x="' + (W - PR + 4) + '" y="' + (Y(l.price) + 3.5) + '" font-size="9" ' +
        'fill="var(--good-text)">★ ' + esc(l.label) + " " + money(l.price) + "</text>";
    });
    out += '<line x1="' + PL + '" y1="' + Y(ch.spot) + '" x2="' + (W - PR) + '" y2="' +
      Y(ch.spot) + '" stroke="var(--axis-ink)" stroke-dasharray="4 4"/>';

    out += '<polyline fill="none" stroke="currentColor" stroke-opacity=".75" ' +
      'stroke-width="1.2" points="' +
      ch.series.map(function (s) { return X(s.i) + "," + Y(s.p); }).join(" ") + '"/>';

    function drawCount(c, color, width, fs, badge) {
      if (!c) return "";
      var s = '<polyline fill="none" stroke="' + color + '" stroke-width="' + width +
        '" stroke-linejoin="round" points="' +
        c.points.map(function (p) { return X(p.i) + "," + Y(p.p); }).join(" ") + '"/>';
      c.points.forEach(function (p) {
        var cx = X(p.i), cy = Y(p.p);
        s += '<circle cx="' + cx + '" cy="' + cy + '" r="' + (badge ? 10 : 3.5) +
          '" fill="' + (badge ? "var(--card)" : color) + '" stroke="' + color +
          '" stroke-width="' + (badge ? 1.6 : 1) + '"/>';
        if (p.l) {
          s += '<text x="' + cx + '" y="' + (cy + (badge ? 3.5 : -8)) +
            '" text-anchor="middle" font-size="' + fs + '" font-weight="bold" fill="' +
            color + '">' + esc(p.l) + "</text>";
        }
      });
      return s;
    }
    out += drawCount(ch.rival, "var(--serious)", 1.8, 8, false);
    out += drawCount(ch.best, "var(--bad)", 2.8, 9, true);
    out += "</svg>";

    var legend = '<p class="hint" style="margin:.35rem 0 0">' +
      '<span style="font-weight:600"><i class="lg" style="background:var(--bad)"></i>best count</span>' +
      (ch.rival ? ' · <span style="font-weight:600"><i class="lg" style="background:var(--serious)"></i>rival count</span>' : "") +
      ' · <span style="font-weight:600"><i class="lg" style="background:var(--good)"></i>★ key fib levels</span>' +
      ' · dashed = today. Numbers mark the pivot each wave <em>ends</em> at; the last ' +
      "point is the wave still in progress.</p>";
    return out + legend;
  }

  function waves(d) {
    var w = d.waves;
    if (!w || !w.available) {
      return el('<section class="card"><h2>Wave structure</h2><p class="hint">' +
        esc((w && w.reason) || "not available for this ticker") + "</p></section>");
    }
    var counts = (w.counts || []).map(function (c, i) {
      var dir = c.rising ? '<span class="hi">▲ rising</span>'
                         : '<span class="lo">▼ falling</span>';
      return "<tr><td>" + (i === 0 ? "<strong>best</strong>" : "rival") + "</td><td>" +
        esc(c.label || c.kind) + "</td><td>" + dir + "</td><td>" + esc(c.stage) +
        "</td><td>" + c.score.toFixed(2) + "</td></tr>";
    }).join("");
    if (!counts) counts = '<tr><td colspan="5">No rule-valid structure in recent swings.</td></tr>';

    function lvRows(list) {
      if (!list || !list.length) return '<tr><td colspan="4">(none)</td></tr>';
      return list.map(function (l) {
        var tags = [];
        if (l.confluent) tags.push("confluent");
        if (l.near_pivot) tags.push("at prior pivot");
        var note = tags.length ? tags.join(", ") : "floats in empty space";
        var cls = tags.length ? "" : ' class="flag"';
        return "<tr><td>" + money(l.price) + "</td><td>" + signed(l.price / d.spot - 1) +
          "</td><td>" + esc(l.label) + "</td><td" + cls + ">" + esc(note) + "</td></tr>";
      }).join("");
    }

    var amb = w.ambiguous
      ? '<div class="warn">⚠ AMBIGUOUS — the top readings are within 15% of each other. ' +
        "The stage is a coin-flip between them; do not act on the label.</div>"
      : "";

    return el('<section class="card"><h2>Wave structure <span class="meta">(descriptive — ' +
      'not a forecast)</span></h2>' +
      '<p class="hint">Elliott counts are subjective in practice and have no established ' +
      "predictive validity. These are computed mechanically (ATR-scaled swings + the three " +
      "hard Elliott rules) so they are at least reproducible and testable.</p>" +
      '<div class="warn">' + esc(w.verdict) + "</div>" + amb +
      '<p class="hint">swing filter ' + (w.threshold * 100).toFixed(1) + "% (ATR " +
      (w.atr_pct * 100).toFixed(1) + "%/day) · " + w.n_pivots + " pivots, " +
      w.n_robust + " robust across thresholds</p>" +
      waveChart(w.chart) +
      '<div class="scroll"><table><thead><tr><th>Reading</th><th>Structure</th>' +
      "<th>Direction</th><th>Stage</th><th>Score</th></tr></thead><tbody>" + counts +
      "</tbody></table></div>" +
      '<p class="hint">“Rising/falling” is the direction the structure <em>travels</em>. ' +
      "A falling correction means price has been going down. The “→ … expected” in the " +
      "stage is what Elliott theory says comes next structurally — it is a label, " +
      "<strong>not</strong> a validated forecast (the stage test found no usable signal " +
      "in forward returns).</p>" +
      '<h2 style="margin-top:1rem;font-size:.98rem">Fib levels below (support candidates)</h2>' +
      '<div class="scroll"><table><thead><tr><th>Price</th><th>vs spot</th><th>Level</th>' +
      "<th>Support?</th></tr></thead><tbody>" + lvRows(w.fib_below) + "</tbody></table></div>" +
      '<h2 style="margin-top:1rem;font-size:.98rem">Fib levels above (targets)</h2>' +
      '<div class="scroll"><table><thead><tr><th>Price</th><th>vs spot</th><th>Level</th>' +
      "<th>Resistance?</th></tr></thead><tbody>" + lvRows(w.fib_above) + "</tbody></table></div>" +
      '<p class="hint" style="margin-top:.7rem">Only levels marked <em>confluent</em> ' +
      "(agreeing across separate swings) or sitting at a prior pivot have even anecdotal " +
      "support. Ones floating in empty space are arithmetic on a chart. And note what the " +
      "validation does <em>not</em> show: turning points landing near these levels " +
      "afterwards is not the same as price reversing at them next time — wave stage " +
      "carried no usable signal for forward returns.</p></section>");
  }

  function disclaimer(d) {
    return el('<section class="card"><h2>Before you use any of this</h2>' +
      '<p class="hint" style="margin:0">' + esc(d.disclaimer) +
      " A model built on price history cannot represent a company failing outright — the " +
      "true downside for a small, cash-burning company is lower than any number shown here. " +
      "For decisions about real money, speak to a licensed adviser who knows your full " +
      "situation.</p></section>");
  }

  /* ------------------------------------------------------------------ *
   * One analysis, two readings.
   *
   * The scorecard and the verdict are shown side by side and deliberately NOT
   * merged into a single number. The scorecard measures risk and refuses to rank
   * on returns; the verdict additionally asks whether a name was paid for its beta.
   * Where they disagree, that is the finding -- averaging them would hide it.
   * ------------------------------------------------------------------ */

  var PRESETS = {
    "load-hb": "WYFI,BMNR,AAOI,CIFR,SMR,RGTI,OKLO,IREN,SMCI,NBIS,CRDO,CLSK,MRVL,SOUN,MU,ASTS,ONDS,TE,EOSE,BBAI",
    "load-nuke": "OKLO,SMR,LEU,NSLR,EOSE,ENPH,FSLR",
    "load-ai": "NBIS,CRWV,IREN,CIFR,CLSK,AMD,MU,MRVL,CRDO,SMCI,AAOI"
  };
  // Weights are the USER's value judgement; the composite is a preference ranking.
  var WKEYS = [
    ["beta", "True high-beta"], ["verifiable", "Verifiable"],
    ["survivable", "Survivable"], ["redundancy", "Adds a new bet"],
    ["liquidity", "Liquid"], ["vol_regime", "Vol regime"]
  ];
  // Gate columns. `regime` is a property of the market, not of any name, so it lives
  // in the regime card instead of repeating identically down every row.
  var CGATES = [["adv", "Liquid"], ["beta", "High beta"], ["treynor", "Treynor"],
                ["idio", "Mkt-driven"], ["survivable", "Survivable"],
                ["own_trend", "Trend"], ["dilution", "Dilution"],
                ["runway", "Runway"], ["revenue", "Revenue"]];
  // Scorecard criteria the gates do not already cover.
  var SEXTRA = [["verifiable", "Verifiable"], ["redundancy", "Adds a bet"],
                ["vol_regime", "Vol now"]];
  var VCLASS = {
    investable: "pass", reckless: "fail", mixed: "warn", stand_aside: "warn",
    excluded_low_beta: "info", excluded_illiquid: "info",
    insufficient_history: "info", error: "info"
  };

  var wState = {};
  WKEYS.forEach(function (k) { wState[k[0]] = 1; });
  var last = null;

  function gradeLabel(g) {
    return g === "pass" ? "OK" : g === "warn" ? "WARN" : g === "fail" ? "FAIL" : "n/a";
  }
  function verdictPill(v) {
    return '<span class="badge ' + (VCLASS[v] || "info") + '">' +
      esc(String(v).replace(/_/g, " ")) + "</span>";
  }
  function verdictCount(v, n) {
    return '<span class="badge ' + (VCLASS[v] || "info") + '">' +
      esc(String(v).replace(/_/g, " ")) + ' <span class="n">' + n + "</span></span>";
  }
  function num(v, dp, sign) {
    if (v == null || !isFinite(v)) return "—";
    return (sign && v >= 0 ? "+" : "") + v.toFixed(dp);
  }
  function pointsOf(c) { return c.grade === "pass" ? 1 : c.grade === "warn" ? 0.5 : 0; }
  function compositeOf(row) {
    var num_ = 0, den = 0;
    (row.criteria || []).forEach(function (c) {
      if (c.grade === "info") return;
      var w = wState[c.key] == null ? 1 : wState[c.key];
      num_ += w * pointsOf(c); den += w;
    });
    return den > 0 ? num_ / den : 0;
  }
  function findBy(list, key) {
    return (list || []).filter(function (x) { return x.key === key; })[0];
  }
  function pillCell(cr) {
    if (!cr) return '<td><span class="pill info">n/a</span></td>';
    return '<td><span class="pill ' + cr.grade + '" title="' + esc(cr.display) +
      " — " + esc(cr.threshold) + '">' + gradeLabel(cr.grade) + "</span></td>";
  }

  /* ---- cards ---- */

  function regimeCard(d) {
    var g = d.regime || {};
    var gap = (g.value != null && isFinite(g.value))
      ? '<span class="delta ' + (g.value >= 0 ? "up" : "down") + '">' +
        (g.value >= 0 ? "+" : "") + (g.value * 100).toFixed(1) + "%</span>"
      : "";
    // The gate's own display already reads "SPY 743.29 vs 200d SMA 696.69 (+6.7%)";
    // the delta chip carries the number, so strip the parenthetical to avoid echoing it.
    var fact = String(g.display || "").replace(/\s*\([^)]*\)\s*$/, "");

    var mode = d.quartile_mode === "quartile"
      ? "Treynor was ranked against the " + d.n_eligible + " eligible tickers you submitted."
      : "Fewer than 8 tickers cleared the hard gates, so the Treynor gate used its " +
        "absolute rule (excess return above zero) rather than a top-quartile rank. " +
        "A real quartile needs a wider universe — the command line runs this over the " +
        "whole S&amp;P 500.";
    var standAside = !d.risk_on
      ? '<div class="warn">The market is below its 200-day average, so nothing is ' +
        "labelled <em>investable</em> — the best a name can score is " +
        verdictPill("stand_aside") + ". A weak tape does not make a good name bad; " +
        "it makes leverage to that tape a poor trade.</div>"
      : "";
    var noFunda = d.fundamentals_available === false
      ? "<p>Dilution, runway and revenue read <em>unverified</em> here: the web app " +
        "fetches prices only. Run <code>python -m scripts.classify</code> for the " +
        "fundamentals-backed verdict.</p>"
      : "";

    return '<section class="card">' +
      '<div class="regime">' +
      '<span class="rstate ' + (d.risk_on ? "on" : "off") + '">' +
      '<span class="dot"></span>' + (d.risk_on ? "Risk-on" : "Risk-off") + "</span>" +
      '<span class="rfact num">' + esc(fact) + "</span>" + gap +
      '<span class="rmeta">' + esc(d.quartile_mode || "") + " basis · " +
      d.n_eligible + " eligible · rf " +
      (d.rf != null ? (d.rf * 100).toFixed(1) + "%" : "—") + "</span></div>" +
      '<p class="lede">' + esc(d.banner || "") + "</p>" + standAside +
      '<details class="fine"><summary>Methodology</summary>' +
      "<p>" + esc(g.threshold || "") + ". " + mode + "</p>" + noFunda + "</details>" +
      "</section>";
  }

  function sliders() {
    var html = WKEYS.map(function (k) {
      return '<label><span class="wname"><span>' + esc(k[1]) + '</span>' +
        '<b id="wv-' + k[0] + '">' + wState[k[0]] + "</b></span>" +
        '<input type="range" min="0" max="3" step="1" value="' + wState[k[0]] +
        '" data-k="' + k[0] + '"></label>';
    }).join("");
    return '<section class="card"><details open><summary class="shead">' +
      "<h2>Scorecard weights</h2></summary>" +
      '<p class="hint">These sliders weight the <strong>scorecard</strong> only. The ' +
      "score column is <em>your</em> weighted preference — not a forecast, and it does " +
      "not affect the verdict, which uses fixed published thresholds.</p>" +
      '<div class="sliders">' + html + "</div></details></section>";
  }

  function table(d) {
    var rows = d.rows.slice().sort(function (a, b) {
      var ra = d.verdict_order.indexOf(a.verdict), rb = d.verdict_order.indexOf(b.verdict);
      if (ra !== rb) return ra - rb;
      return compositeOf(b) - compositeOf(a);
    });

    var counts = {};
    d.rows.forEach(function (r) { counts[r.verdict] = (counts[r.verdict] || 0) + 1; });
    var summary = d.verdict_order.filter(function (v) { return counts[v]; })
      .map(function (v) { return verdictCount(v, counts[v]); }).join(" ");

    var head = "<tr><th>Ticker</th><th>Verdict</th><th>Score</th>" +
      CGATES.map(function (g) { return "<th>" + esc(g[1]) + "</th>"; }).join("") +
      SEXTRA.map(function (g) { return "<th>" + esc(g[1]) + "</th>"; }).join("") +
      "<th>β 252d</th><th>Treynor</th></tr>";

    var prev = null;
    var body = rows.map(function (r) {
      var cells = CGATES.map(function (g) { return pillCell(findBy(r.gates, g[0])); }).join("") +
        SEXTRA.map(function (g) { return pillCell(findBy(r.criteria, g[0])); }).join("");
      var m = r.metrics || {};
      var t = m.treynor;
      // Rule between verdict blocks; the sort already grouped them.
      var grp = (prev !== null && r.verdict !== prev) ? ' class="grp"' : "";
      prev = r.verdict;
      return "<tr" + grp + '><td><span class="tick">' + esc(r.symbol) + "</span>" +
        '<span class="sub-px">' + money(r.spot) + "</span></td>" +
        "<td>" + verdictPill(r.verdict) + "</td>" +
        '<td class="num"><strong>' + compositeOf(r).toFixed(2) + "</strong></td>" + cells +
        '<td class="num">' + num(m.beta_252, 2) + "</td>" +
        '<td class="num ' + (t == null || !isFinite(t) ? "" : t >= 0 ? "pos" : "neg") +
        '">' + num(t, 2, true) + "</td></tr>";
    }).join("");

    var fails = d.failures && d.failures.length
      ? '<div class="warn">Could not analyse: ' +
        d.failures.map(function (f) { return esc(f.symbol) + " (" + esc(f.reason) + ")"; })
          .join("; ") + "</div>"
      : "";
    var trunc = d.truncated && d.truncated.length
      ? '<div class="warn">⚠ Only the first ' + d.max_symbols + " tickers were analysed. " +
        "NOT analysed: " + d.truncated.map(esc).join(", ") + "</div>"
      : "";

    function critList(list) {
      return (list || []).map(function (cr) {
        return '<span class="pill ' + cr.grade + '">' + gradeLabel(cr.grade) +
          "</span><span><strong>" + esc(cr.title) + "</strong> — " + esc(cr.display) +
          '<br><span class="thr">threshold: ' + esc(cr.threshold) +
          (cr.note ? " · " + esc(cr.note) : "") + "</span></span>";
      }).join("");
    }

    var details = rows.map(function (r) {
      var gateRows = critList(r.gates);
      return '<details class="det"><summary><span class="tick">' + esc(r.symbol) +
        '</span><span class="nm">' + esc(r.name || r.symbol) + "</span>" +
        verdictPill(r.verdict) + "</summary>" +
        '<div class="dgrid">' +
        '<div><span class="mlabel">Verdict gates</span><div class="crit">' +
        (gateRows || '<span></span><span class="thr">not gated — see the reason above</span>') +
        "</div></div>" +
        '<div><span class="mlabel">Risk scorecard</span><div class="crit">' +
        critList(r.criteria) + "</div></div>" +
        "</div></details>";
    }).join("");

    return '<section class="card"><h2>Cohort — ' + d.n_analysed + " of " +
      d.n_requested + " tickers</h2>" +
      '<p class="hint" style="display:flex;gap:.35rem;flex-wrap:wrap">' + summary +
      "</p>" + trunc + fails +
      '<div class="scroll"><table><thead>' + head + "</thead><tbody>" + body +
      "</tbody></table></div>" +
      '<p class="hint">Hover any pill for its value and threshold. <strong>Excluded</strong> ' +
      "is not a criticism — an illiquid or low-beta name is simply outside what the " +
      "verdict judges, though its scorecard still applies. Open a row for every " +
      "threshold behind both readings.</p>" +
      '<h2 style="margin-top:1.2rem;font-size:1rem">Per-ticker detail</h2>' + details +
      "</section>";
  }

  /* ---- render ---- */

  function render(d) {
    last = d;
    out.innerHTML = "";
    out.appendChild(el(regimeCard(d)));
    out.appendChild(el(sliders()));
    var tbl = el(table(d));
    tbl.id = "cohort";
    out.appendChild(tbl);

    // A cone is only readable for one name, so the forecast rides along with a lone
    // ticker rather than being a separate mode.
    if (d.forecast) {
      [header, headline, waves, disclaimer].forEach(function (f) {
        out.appendChild(f(d.forecast));
      });
      wireCone(out);        // needs the SVG in the document to measure widths
    }

    Array.prototype.forEach.call(out.querySelectorAll('input[type=range]'), function (r) {
      r.addEventListener("input", function () {
        wState[r.getAttribute("data-k")] = parseFloat(r.value);
        var v = document.getElementById("wv-" + r.getAttribute("data-k"));
        if (v) v.textContent = r.value;
        // Re-render only the table so slider focus is preserved.
        var old = document.getElementById("cohort");
        if (old) {
          var next = el(table(last));
          next.id = "cohort";
          old.parentNode.replaceChild(next, old);
        }
      });
    });
    out.hidden = false;
  }

  // Upstream price requests can stall (Yahoo throttles). Without a deadline the
  // spinner runs forever with no feedback and no way out, which reads as a frozen
  // app rather than a slow one. Bound it, show the clock, and fail with advice.
  var REQUEST_TIMEOUT_MS = 120000;

  function run(list) {
    var raw = (list || input.value || "").trim();
    if (!raw) return;
    input.value = raw;
    var syms = raw.replace(/\s+/g, ",").split(",").filter(Boolean);
    out.hidden = true;
    go.disabled = true;

    var base = syms.length === 1
      ? "Analysing " + syms[0].toUpperCase() + " and running 8,000 simulations…"
      : "Analysing " + syms.length + " tickers…";
    setStatus(base);

    var t0 = Date.now();
    // Only start counting aloud after 3s, so quick runs stay quiet.
    var tick = setInterval(function () {
      var s = Math.round((Date.now() - t0) / 1000);
      if (s >= 3 && !statusEl.classList.contains("err")) {
        setStatus(base + "  " + s + "s");
      }
    }, 1000);

    var ctrl = (typeof AbortController !== "undefined") ? new AbortController() : null;
    var killer = setTimeout(function () { if (ctrl) ctrl.abort(); }, REQUEST_TIMEOUT_MS);
    function done() { clearInterval(tick); clearTimeout(killer); go.disabled = false; }

    var url = "/api/analyse?symbols=" + encodeURIComponent(syms.join(",")) +
      "&months=" + encodeURIComponent(monthsSel.value) +
      "&rf=" + encodeURIComponent(rfSel.value);
    fetch(url, ctrl ? { signal: ctrl.signal } : undefined).then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, body: j }; });
    }).then(function (res) {
      if (!res.ok || res.body.error) throw new Error(res.body.error || "Request failed");
      setStatus("");
      render(res.body);
      try {
        history.replaceState(null, "", "?symbols=" + encodeURIComponent(syms.join(",")));
      } catch (e) { /* file:// and sandboxed frames reject this; harmless */ }
    }).catch(function (e) {
      if (e && e.name === "AbortError") {
        setStatus("Gave up after " + Math.round(REQUEST_TIMEOUT_MS / 1000) +
          "s — the price source is not responding. It rate-limits bursts, so wait a " +
          "moment and try again, or analyse fewer tickers.", true);
      } else {
        setStatus(netMsg(e, "Could not analyse those tickers."), true);
      }
    }).then(done);
  }

  /* ------------------------------------------------------------------ *
   * Portfolio import
   *
   * Parsed HERE, in the browser, and never uploaded: the app's standing claim is
   * that nothing leaves your machine except price requests, and a POST of your
   * holdings would break that the moment this is deployed anywhere.
   *
   * These rules MIRROR src/forecast/portfolio.py -- headings, the non-ticker list
   * and the ticker pattern are the same in both. Change one, change the other.
   * Note the dot is preserved: in a portfolio it is an exchange suffix (BHP.AX),
   * not the BRK.B -> BRK-B rewrite that universe.py applies to the index list.
   * ------------------------------------------------------------------ */
  var SYMBOL_HEADINGS = ["symbol", "ticker", "tickersymbol", "symbols", "tickers"];
  var NOT_TICKERS = ["CASH", "$CASH", "$$CASH", "TOTAL", "TOTALS", "SUBTOTAL",
                     "N/A", "NA", "--", "-"];
  var TICKER_RE = /^\^?[A-Z0-9][A-Z0-9.\-=]{0,11}$/;

  function parseCsv(text) {
    var rows = [], row = [], cell = "", quoted = false;
    for (var i = 0; i < text.length; i++) {
      var c = text.charAt(i);
      if (quoted) {
        if (c === '"') {
          if (text.charAt(i + 1) === '"') { cell += '"'; i++; } else { quoted = false; }
        } else { cell += c; }
      } else if (c === '"') { quoted = true; }
      else if (c === ",") { row.push(cell); cell = ""; }
      else if (c === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
      else if (c !== "\r") { cell += c; }
    }
    if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
    return rows;
  }

  function normHead(s) { return String(s).toLowerCase().replace(/[^a-z]/g, ""); }
  function isTicker(v) {
    v = String(v).trim().toUpperCase();
    return !!v && NOT_TICKERS.indexOf(v) < 0 && TICKER_RE.test(v);
  }

  function symbolsFromCsvText(text) {
    if (text.charAt(0) === "﻿") text = text.slice(1);      // Excel/Yahoo BOM
    var rows = parseCsv(text).filter(function (r) {
      return r.some(function (c) { return String(c).trim() !== ""; });
    });
    if (!rows.length) return { symbols: [], skipped: [] };

    var head = rows[0].map(normHead), col = -1;
    for (var i = 0; i < head.length; i++) {
      if (SYMBOL_HEADINGS.indexOf(head[i]) >= 0) { col = i; break; }
    }
    var body;
    if (col < 0) {
      if (!isTicker(rows[0][0])) {
        throw new Error("No ticker column found. Expected a header containing " +
          "Symbol or Ticker — or a file whose first column is just ticker symbols.");
      }
      col = 0; body = rows;                    // headerless: every row is data
    } else {
      body = rows.slice(1);
    }

    var out = [], seen = {}, skipped = [];
    body.forEach(function (r) {
      var raw = String(col < r.length ? r[col] : "").trim();
      if (!raw) return;
      var v = raw.toUpperCase();
      if (NOT_TICKERS.indexOf(v) >= 0) {
        skipped.push({ value: raw, reason: "not a tradable instrument" }); return;
      }
      if (!TICKER_RE.test(v)) {
        skipped.push({ value: raw, reason: "does not look like a ticker" }); return;
      }
      if (seen[v]) return;                     // extra lots of the same holding
      seen[v] = 1; out.push(v);
    });
    return { symbols: out, skipped: skipped };
  }

  var fileIn = document.getElementById("pfile");
  var importNote = document.getElementById("import-note");

  function showImport(html, isErr) {
    importNote.hidden = false;
    importNote.innerHTML = html;
    importNote.classList.toggle("drop", !!isErr);
  }

  if (fileIn) {
    fileIn.addEventListener("change", function () {
      var f = fileIn.files && fileIn.files[0];
      if (!f) return;
      var reader = new FileReader();
      reader.onerror = function () {
        showImport("Could not read <b>" + esc(f.name) + "</b>.", true);
      };
      reader.onload = function () {
        var res;
        try {
          res = symbolsFromCsvText(String(reader.result || ""));
        } catch (e) {
          showImport(esc(e.message || "Could not parse that file."), true);
          return;
        }
        if (!res.symbols.length) {
          showImport("No tickers found in <b>" + esc(f.name) + "</b>.", true);
          return;
        }
        var msg = "Loaded <b>" + res.symbols.length + "</b> ticker" +
          (res.symbols.length === 1 ? "" : "s") + " from <b>" + esc(f.name) + "</b>";
        if (res.skipped.length) {
          msg += ' · <span class="drop">ignored ' + res.skipped.length + ": " +
            res.skipped.slice(0, 4).map(function (s) { return esc(s.value); }).join(", ") +
            (res.skipped.length > 4 ? " …" : "") + "</span>";
        }
        // No client-side cap warning: the server owns the limit and reports any
        // truncation authoritatively in the response. Mirroring the number here would
        // just be a constant waiting to drift out of sync with serve.py.
        showImport(msg);
        run(res.symbols.join(","));
      };
      reader.readAsText(f);
      fileIn.value = "";        // let the same file be re-picked after an edit
    });
  }

  form.addEventListener("submit", function (e) { e.preventDefault(); run(); });
  Array.prototype.forEach.call(document.querySelectorAll("#examples .chip"), function (b) {
    if (b.id === "go-report") return;        // owned by report.js, not a cohort preset
    var preset = PRESETS[b.id];
    b.addEventListener("click", function () {
      run(preset || b.getAttribute("data-s"));
    });
  });

  // Shared with report.js, which renders a different view into the same #out and
  // must format numbers and report failures identically. Exposed rather than
  // duplicated so the two views cannot drift apart.
  window.LQ = { setStatus: setStatus, esc: esc, money: money, pct: pct, netMsg: netMsg };

  var reportBtn = document.getElementById("go-report");
  if (reportBtn) {
    reportBtn.addEventListener("click", function () {
      if (window.LQReport) window.LQReport.run();
    });
  }

  var q = new URLSearchParams(location.search);
  // ?report= is handled by report.js, which loads after this file and owns that view.
  // Deferring to it from here with a timer is a race: the timer can fire before
  // report.js has finished downloading, and the call is silently lost.
  var initial = q.get("report") ? null : (q.get("symbols") || q.get("symbol"));
  if (initial) run(initial);
})();
