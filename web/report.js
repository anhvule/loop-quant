/* Company report — renders the /api/company payload.

   Design rule, inherited from app.js: every score is a count of NAMED checks with a
   stated threshold, shown beside the value it saw. No composite, no rating, no
   number that cannot be traced back to a rule the reader can disagree with. */
(function () {
  "use strict";

  var LQ = window.LQ || {};
  var esc = LQ.esc, money = LQ.money, pct = LQ.pct, setStatus = LQ.setStatus;
  var out = document.getElementById("out");
  var input = document.getElementById("symbols");

  var AXES = [
    { key: "value", label: "Valuation", short: "Value" },
    { key: "future", label: "Future growth", short: "Future" },
    { key: "past", label: "Past performance", short: "Past" },
    { key: "health", label: "Financial health", short: "Health" },
    { key: "dividend", label: "Dividend", short: "Dividend" }
  ];

  function el(html) { var d = document.createElement("div"); d.innerHTML = html.trim(); return d.firstChild; }
  function n(v, dp) { return (v == null || !isFinite(v)) ? "—" : v.toFixed(dp == null ? 2 : dp); }

  /* Big money: market caps and revenue read as 3.24T, not $3,240,000,000,000. */
  function big(v) {
    if (v == null || !isFinite(v)) return "—";
    var a = Math.abs(v), u = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "k"]];
    for (var i = 0; i < u.length; i++) {
      if (a >= u[i][0]) return (v / u[i][0]).toFixed(2) + u[i][1];
    }
    return v.toFixed(0);
  }
  function bigMoney(v) { return v == null || !isFinite(v) ? "—" : "$" + big(v); }
  function count(v) { return (v == null || !isFinite(v)) ? "—" : Math.round(v).toLocaleString(); }

  function scoreClass(s) { return s <= 2 ? "bad" : s <= 4 ? "mid" : "good"; }
  function gradeLabel(g) { return g === "pass" ? "PASS" : g === "fail" ? "FAIL" : "n/a"; }

  /* ---- snowflake radar (inline SVG, no libraries) ----
     Five axes, six rings — one ring per check, so the shape's radius IS the score.
     Clockwise from the top in the same order the sections are listed below. */
  function snowflake(sn) {
    var W = 320, H = 300, CX = 160, CY = 148, R = 96, MAX = 6;
    var axes = sn.axes || [];
    function pt(i, r) {
      var a = -Math.PI / 2 + i * (2 * Math.PI / axes.length);
      return [CX + Math.cos(a) * r, CY + Math.sin(a) * r];
    }
    function ring(r) {
      return axes.map(function (_, i) { return pt(i, r).join(","); }).join(" ");
    }

    var svg = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' +
      esc("Snowflake score " + sn.total + " of " + sn.max) + '">';
    for (var k = 1; k <= MAX; k++) {
      svg += '<polygon points="' + ring(R * k / MAX) + '" fill="none" ' +
        'stroke="var(--grid)" stroke-width="' + (k === MAX ? 1.2 : .7) + '"/>';
    }
    axes.forEach(function (_, i) {
      var p = pt(i, R);
      svg += '<line x1="' + CX + '" y1="' + CY + '" x2="' + p[0] + '" y2="' + p[1] +
        '" stroke="var(--axis)" stroke-width=".8"/>';
    });

    // Score polygon. A zero axis would collapse to the centre and read as missing,
    // so every vertex keeps a visible floor.
    var poly = axes.map(function (a, i) {
      return pt(i, Math.max(R * a.score / MAX, 3)).join(",");
    }).join(" ");
    svg += '<polygon points="' + poly + '" fill="var(--band)" fill-opacity=".26" ' +
      'stroke="var(--band)" stroke-width="2" stroke-linejoin="round"/>';
    axes.forEach(function (a, i) {
      var p = pt(i, Math.max(R * a.score / MAX, 3));
      svg += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="3.2" fill="var(--band)"/>';
    });

    axes.forEach(function (a, i) {
      var p = pt(i, R + 24), anchor = "middle";
      if (p[0] > CX + 12) anchor = "start";
      else if (p[0] < CX - 12) anchor = "end";
      var label = (AXES.filter(function (x) { return x.key === a.key; })[0] || {}).short
        || a.label;
      svg += '<text x="' + p[0] + '" y="' + (p[1] + 1) + '" text-anchor="' + anchor +
        '" font-size="11" font-weight="600" fill="var(--ink-2)">' + esc(label) +
        "</text>" +
        '<text x="' + p[0] + '" y="' + (p[1] + 14) + '" text-anchor="' + anchor +
        '" font-size="10" fill="var(--axis-ink)">' + a.score + "/" + MAX + "</text>";
    });
    return svg + "</svg>";
  }

  function snowCard(d) {
    var sn = d.snowflake;
    var bars = sn.axes.map(function (a) {
      var label = (AXES.filter(function (x) { return x.key === a.key; })[0] || {}).label
        || a.key;
      return '<div class="axrow"><span class="axname">' + esc(label) + "</span>" +
        '<span class="axbar"><i class="' + scoreClass(a.score) + '" style="width:' +
        (a.score / a.max * 100) + '%"></i></span>' +
        '<span class="axnum">' + a.score + "/" + a.max + "</span></div>";
    }).join("");

    var warn = (d.warnings || []).map(function (w) {
      return '<div class="warn">' + esc(w) + "</div>";
    }).join("");

    return '<section class="card"><h2>Snowflake <span class="meta">— ' + sn.total +
      " of " + sn.max + " checks passed</span></h2>" +
      '<p class="hint">Each axis is a count of the six checks below it that passed. ' +
      "A low score can mean the company failed the check <em>or</em> that the data to " +
      "judge it was missing — open a section to see which, per check.</p>" +
      '<div class="snowflake"><div class="snowfig">' + snowflake(sn) + "</div>" +
      '<div class="snowside"><p class="lede">' + esc(sn.summary) + "</p>" +
      '<div class="axlist">' + bars + "</div></div></div>" + warn + "</section>";
  }

  /* ---- header ---- */
  function headerCard(d) {
    var o = d.overview, meta = [o.sector, o.industry, o.exchange]
      .filter(Boolean).map(esc).join(" · ");
    var sum = o.summary
      ? '<details class="fine"><summary>What the company does</summary><p>' +
        esc(o.summary) + "</p></details>"
      : "";
    function m(label, v, big_) {
      return '<div class="metric"><span class="mlabel">' + esc(label) + "</span>" +
        '<span class="v' + (big_ ? " big" : "") + '">' + v + "</span></div>";
    }
    return '<section class="card"><div class="head">' +
      "<div><h2>" + esc(d.symbol) + " — " + esc(o.name || d.symbol) + "</h2>" +
      '<p class="hint" style="margin-bottom:0">' + meta + " · as of " + esc(d.as_of) +
      (d.cached ? " (cached)" : "") + "</p></div>" +
      '<div class="metrics">' +
      m("Price", money(o.price) + " " + esc(o.currency || ""), true) +
      m("Market cap", bigMoney(o.market_cap)) +
      m("Employees", count(o.employees)) +
      "</div></div>" + sum + "</section>";
  }

  /* ---- shared chart helpers ---- */
  function priceChart(series) {
    if (!series || series.length < 2) return "";
    var W = 720, H = 170, PL = 54, PR = 12, PT = 12, PB = 22;
    var ps = series.map(function (s) { return s.p; });
    var lo = Math.min.apply(null, ps), hi = Math.max.apply(null, ps);
    // Floored at zero: padding a range that starts near the axis produced a
    // "-$5.91" tick on the price chart, which is not a price.
    var pad = (hi - lo) * .08 || 1; lo = Math.max(0, lo - pad); hi += pad;
    var X = function (i) { return PL + (i / (series.length - 1)) * (W - PL - PR); };
    var Y = function (p) { return PT + ((hi - p) / (hi - lo)) * (H - PT - PB); };

    // money() renders sub-$1 values to 3 decimals, which turns a floored axis into
    // "$0.000". A zero tick is just zero.
    var tick = function (v) { return v === 0 ? "$0" : money(v); };
    var ticks = [lo, (lo + hi) / 2, hi].map(function (v) {
      return '<line x1="' + PL + '" y1="' + Y(v) + '" x2="' + (W - PR) + '" y2="' +
        Y(v) + '" stroke="var(--grid)"/><text x="' + (PL - 6) + '" y="' + (Y(v) + 4) +
        '" text-anchor="end" font-size="10" fill="var(--axis-ink)">' + tick(v) +
        "</text>";
    }).join("");
    var line = series.map(function (s, i) { return X(i) + "," + Y(s.p); }).join(" ");
    var ends = '<text x="' + PL + '" y="' + (H - 6) + '" font-size="10" ' +
      'fill="var(--axis-ink)">' + esc(series[0].d) + "</text>" +
      '<text x="' + (W - PR) + '" y="' + (H - 6) + '" text-anchor="end" font-size="10" ' +
      'fill="var(--axis-ink)">' + esc(series[series.length - 1].d) + "</text>";
    return '<svg viewBox="0 0 ' + W + " " + H + '" role="img" ' +
      'aria-label="Five-year price history">' + ticks +
      '<polyline fill="none" stroke="var(--band)" stroke-width="1.6" points="' +
      line + '"/>' + ends + "</svg>";
  }

  /* Grouped annual bars. `series` is [{label, values:[], color}], years newest-first
     from the API and reversed here so time reads left to right. */
  function barChart(years, series, fmt) {
    if (!years || !years.length) return "";
    var yrs = years.slice().reverse();
    var sets = series.map(function (s) {
      return { label: s.label, color: s.color, values: (s.values || []).slice().reverse() };
    }).filter(function (s) { return s.values.length; });
    if (!sets.length) return "";

    var W = 720, H = 190, PL = 58, PR = 12, PT = 14, PB = 30;
    var all = [];
    sets.forEach(function (s) { s.values.forEach(function (v) { if (v != null) all.push(v); }); });
    if (!all.length) return "";
    var hi = Math.max.apply(null, all.concat([0]));
    var lo = Math.min.apply(null, all.concat([0]));
    var span = (hi - lo) || 1;
    var Y = function (v) { return PT + ((hi - v) / span) * (H - PT - PB); };
    var zero = Y(0);
    var slot = (W - PL - PR) / yrs.length;
    var bw = Math.min(30, slot / (sets.length + 1));

    var bars = "", labels = "";
    yrs.forEach(function (yr, i) {
      var x0 = PL + i * slot + (slot - bw * sets.length) / 2;
      sets.forEach(function (s, j) {
        var v = s.values[i];
        if (v == null) return;
        var y = Y(Math.max(v, 0)), h = Math.abs(Y(v) - zero);
        var color = typeof s.color === "function" ? s.color(v) : s.color;
        bars += '<rect x="' + (x0 + j * bw) + '" y="' + y + '" width="' + (bw - 3) +
          '" height="' + Math.max(h, 1) + '" fill="' + color + '" rx="1.5"><title>' +
          esc(s.label + " " + yr + ": " + (fmt || bigMoney)(v)) + "</title></rect>";
      });
      labels += '<text x="' + (PL + i * slot + slot / 2) + '" y="' + (H - 10) +
        '" text-anchor="middle" font-size="10" fill="var(--axis-ink)">' + esc(yr) +
        "</text>";
    });

    var axis = '<line x1="' + PL + '" y1="' + zero + '" x2="' + (W - PR) + '" y2="' +
      zero + '" stroke="var(--axis)"/>' +
      '<text x="' + (PL - 6) + '" y="' + (Y(hi) + 8) + '" text-anchor="end" ' +
      'font-size="10" fill="var(--axis-ink)">' + (fmt || bigMoney)(hi) + "</text>";
    var legend = '<div class="legend">' + sets.map(function (s) {
      var c = typeof s.color === "function" ? s.color(1) : s.color;
      return '<span><i style="background:' + c + '"></i>' + esc(s.label) + "</span>";
    }).join("") + "</div>";

    return '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' +
      esc(sets.map(function (s) { return s.label; }).join(" and ") + " by year") +
      '">' + axis + bars + labels + "</svg>" + legend;
  }

  /* Valuation gauge: price against fair value and the analyst target range, on one
     number line. Two independent opinions about the same number, side by side. */
  function valuationGauge(st, dcf) {
    var price = st.price, fv = st.fair_value;
    var pts = [price, fv, st.target_low, st.target_mean, st.target_high]
      .filter(function (v) { return v != null && isFinite(v) && v > 0; });
    if (pts.length < 2 || price == null) return "";
    var lo = Math.min.apply(null, pts), hi = Math.max.apply(null, pts);
    var pad = (hi - lo) * .18 || hi * .1; lo = Math.max(0, lo - pad); hi += pad;

    var W = 720, H = 108, PL = 16, PR = 16, AX = 66;
    var X = function (v) { return PL + ((v - lo) / (hi - lo)) * (W - PL - PR); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" ' +
      'aria-label="Price against fair value and analyst targets">';

    if (st.target_low != null && st.target_high != null) {
      s += '<rect x="' + X(st.target_low) + '" y="' + (AX - 11) + '" width="' +
        Math.max(X(st.target_high) - X(st.target_low), 1) + '" height="22" ' +
        'fill="var(--band)" fill-opacity=".14" rx="3"/>' +
        '<text x="' + X(st.target_low) + '" y="' + (AX + 27) + '" font-size="9.5" ' +
        'text-anchor="middle" fill="var(--axis-ink)">low ' + money(st.target_low) +
        "</text>" +
        '<text x="' + X(st.target_high) + '" y="' + (AX + 27) + '" font-size="9.5" ' +
        'text-anchor="middle" fill="var(--axis-ink)">high ' + money(st.target_high) +
        "</text>";
    }
    s += '<line x1="' + PL + '" y1="' + AX + '" x2="' + (W - PR) + '" y2="' + AX +
      '" stroke="var(--axis)"/>';

    function mark(v, color, label, above) {
      if (v == null || !isFinite(v)) return "";
      var x = X(v), y = above ? AX - 14 : AX + 14;
      return '<line x1="' + x + '" y1="' + (AX - 11) + '" x2="' + x + '" y2="' +
        (AX + 11) + '" stroke="' + color + '" stroke-width="2.5"/>' +
        '<text x="' + x + '" y="' + (above ? y - 12 : y + 10) + '" text-anchor="middle" ' +
        'font-size="11" font-weight="650" fill="' + color + '">' + money(v) + "</text>" +
        '<text x="' + x + '" y="' + (above ? y : y + 0) + '" text-anchor="middle" ' +
        'font-size="9.5" fill="var(--muted)">' + esc(label) + "</text>";
    }
    s += mark(st.target_mean, "var(--axis-ink)", "analyst mean", false);
    s += mark(st.fair_value, "var(--good)", "DCF fair value", true);
    s += mark(price, "var(--ink)", "price now", true);
    s += "</svg>";

    if (dcf) {
      s += '<p class="hint" style="margin:.5rem 0 0">Fair value assumes free cash flow ' +
        "of " + bigMoney(dcf.fcf_base) +
        (dcf.fcf_basis ? " (" + esc(dcf.fcf_basis) + ")" : "") + " growing " +
        pct(dcf.growth_used, 1) +
        " a year for five years (" + esc(dcf.growth_source) + "), fading to " +
        pct(dcf.terminal_growth, 1) + ", discounted at " + pct(dcf.discount_rate, 1) +
        ". Change any one of those and the number moves a lot — it is a sanity check " +
        "on the price, <strong>not</strong> a target.</p>";
    }
    return s;
  }

  /* ---- section card ---- */
  function checkList(checks) {
    return '<div class="crit">' + (checks || []).map(function (c) {
      return '<span class="pill ' + c.grade + '">' + gradeLabel(c.grade) + "</span>" +
        "<span><strong>" + esc(c.title) + "</strong> — " + esc(c.display) +
        '<br><span class="thr">threshold: ' + esc(c.threshold) + "</span></span>";
    }).join("") + "</div>";
  }

  function tiles(list) {
    var cells = list.filter(function (t) { return t[1] !== "—"; }).map(function (t) {
      return '<div class="tile"><span class="mlabel">' + esc(t[0]) +
        '</span><span class="v">' + t[1] + "</span></div>";
    }).join("");
    return cells ? '<div class="tiles">' + cells + "</div>" : "";
  }

  function sectionCard(key, title, sec, body, hint) {
    var badge = '<span class="axis-badge ' + scoreClass(sec.score) + '">' +
      sec.score + "/" + (sec.max || 6) + "</span>";
    var unscored = sec.n_evaluable < (sec.max || 6)
      ? '<p class="hint">' + ((sec.max || 6) - sec.n_evaluable) +
        " of " + (sec.max || 6) + " checks could not be evaluated — shown as " +
        "<span class=\"pill info\">n/a</span> below, and counted as neither a pass " +
        "nor a failure.</p>"
      : "";
    return '<section class="card" id="sec-' + key + '"><div class="shead2"><h2>' +
      esc(title) + "</h2>" + badge + "</div>" +
      (hint ? '<p class="hint">' + hint + "</p>" : "") +
      (body || "") + unscored + checkList(sec.checks) + "</section>";
  }

  /* ---- the seven sections ---- */
  function valuationCard(d) {
    var s = d.sections.value, st = s.stats;
    return sectionCard("value", "1 · Valuation", s,
      tiles([["Price", money(st.price)], ["DCF fair value", money(st.fair_value)],
             ["P/E", n(st.pe, 1)], ["P/B", n(st.pb, 2)], ["PEG", n(st.peg, 2)],
             ["Analyst mean", money(st.target_mean)]]) +
      valuationGauge(st, s.dcf));
  }

  function futureCard(d) {
    var s = d.sections.future, st = s.stats;
    // Revenue only. EPS shares no scale with revenue -- plotting both put dollars
    // against billions of dollars and rendered the EPS bars as invisible slivers.
    // The EPS estimates are in the tiles above, where they are legible.
    var chart = barChart(["next year", "this year"], [
      { label: "Revenue estimate", color: "var(--band)",
        values: [st.rev_next, st.rev_now] }
    ]);
    return sectionCard("future", "2 · Future growth", s,
      tiles([["Forecast EPS growth", pct(st.eps_growth_1y, 1)],
             ["Forecast revenue growth", pct(st.rev_growth_1y, 1)],
             ["EPS this year", n(st.eps_now)], ["EPS next year", n(st.eps_next)]]) +
      chart,
      "Analyst consensus forecasts, as collected by the data vendor. Consensus is " +
      "frequently wrong and tends to be optimistic — it describes expectations, not " +
      "outcomes.");
  }

  function pastCard(d) {
    var s = d.sections.past, st = s.stats, se = s.series;
    var chart = barChart(se.years, [
      { label: "Revenue", color: "var(--band)", values: se.revenue },
      { label: "Net income", color: function (v) { return v < 0 ? "var(--bad)" : "var(--good)"; },
        values: se.net_income }
    ]);
    return sectionCard("past", "3 · Past performance", s,
      tiles([["Revenue", bigMoney(st.revenue)], ["Net income", bigMoney(st.net_income)],
             ["ROE", pct(st.roe, 1)], ["ROA", pct(st.roa, 1)],
             ["ROCE", pct(st.roce, 1)]]) + chart);
  }

  function healthCard(d) {
    var s = d.sections.health, st = s.stats;
    return sectionCard("health", "4 · Financial health", s,
      tiles([["Debt / equity", pct(st.debt_to_equity, 0)],
             ["Total debt", bigMoney(st.total_debt)], ["Cash", bigMoney(st.cash)],
             ["Current ratio", n(st.current_ratio)],
             ["Interest cover", st.interest_coverage == null ? "—"
               : n(st.interest_coverage, 1) + "×"]]));
  }

  function dividendCard(d) {
    var s = d.sections.dividend, st = s.stats, se = s.series;
    if (!s.pays_dividend) {
      return '<section class="card" id="sec-dividend"><div class="shead2">' +
        "<h2>5 · Dividend</h2>" +
        '<span class="axis-badge bad">0/6</span></div>' +
        '<p class="hint" style="margin-bottom:0">' + esc(d.symbol) +
        " pays no dividend, so all six dividend checks fail by definition. That is " +
        "not a criticism — a company reinvesting everything can be the better " +
        "business. It just scores zero on an axis built to measure income.</p>" +
        "</section>";
    }
    var chart = barChart(se.years.slice().reverse(), [
      { label: "Dividend per share", color: "var(--accent)",
        values: se.dps.slice().reverse() }
    ], function (v) { return "$" + n(v); });
    return sectionCard("dividend", "5 · Dividend", s,
      tiles([["Yield", pct(st.yield, 2)], ["Payout ratio", pct(st.payout_ratio, 0)],
             ["Latest DPS", st.dps_latest == null ? "—" : "$" + n(st.dps_latest)]]) +
      chart);
  }

  function managementCard(d) {
    var m = d.sections.management;
    if (!m.officers.length) {
      return '<section class="card"><h2>6 · Management</h2>' +
        '<p class="hint" style="margin-bottom:0">No officer data published for this ' +
        "ticker.</p></section>";
    }
    var rows = m.officers.map(function (o) {
      return "<tr><td>" + esc(o.name) + "</td><td>" + esc(o.title || "—") +
        '</td><td class="num">' + (o.age == null ? "—" : Math.round(o.age)) +
        '</td><td class="num">' + (o.pay == null ? "—" : bigMoney(o.pay)) +
        "</td></tr>";
    }).join("");
    return '<section class="card"><h2>6 · Management</h2>' +
      '<p class="hint">Officers and reported total pay. Tenure is not published in ' +
      "this feed; compensation is the most recent figure the vendor holds and may lag " +
      "a year.</p>" +
      '<div class="scroll"><table><thead><tr><th>Name</th><th>Role</th><th>Age</th>' +
      "<th>Total pay</th></tr></thead><tbody>" + rows + "</tbody></table></div></section>";
  }

  function ownershipCard(d) {
    var o = d.sections.ownership;
    var rows = (o.top_institutions || []).map(function (h) {
      return "<tr><td>" + esc(h.holder) + '</td><td class="num">' + pct(h.pct, 2) +
        '</td><td class="num">' + bigMoney(h.value) + "</td></tr>";
    }).join("");
    var table = rows
      ? '<div class="scroll"><table><thead><tr><th>Institution</th><th>% held</th>' +
        "<th>Value</th></tr></thead><tbody>" + rows + "</tbody></table></div>"
      : '<p class="hint">No institutional holdings published.</p>';

    var insider = (o.insider_buys_12m == null && o.insider_sells_12m == null)
      ? '<p class="hint" style="margin-bottom:0">No insider transactions published.</p>'
      : '<p class="hint" style="margin-bottom:0">Over the last 12 months insiders ' +
        "made <strong>" + o.insider_buys_12m + "</strong> purchase" +
        (o.insider_buys_12m === 1 ? "" : "s") + " and <strong>" +
        o.insider_sells_12m + "</strong> sale" +
        (o.insider_sells_12m === 1 ? "" : "s") + ", a net of <strong>" +
        count(o.insider_net_shares_12m) + "</strong> shares. Insider selling has many " +
        "innocent explanations (tax, diversification, scheduled plans); insider " +
        "<em>buying</em> is the more informative direction.</p>";

    return '<section class="card"><h2>7 · Ownership</h2>' +
      tiles([["Insiders", pct(o.insider_pct, 2)],
             ["Institutions", pct(o.institution_pct, 1)]]) +
      table + insider + "</section>";
  }

  function disclaimerCard(d) {
    return '<section class="card"><h2>Before you use any of this</h2>' +
      '<p class="hint" style="margin:0">' + esc(d.disclaimer) + "</p></section>";
  }

  /* ---- render ---- */
  function render(d) {
    out.innerHTML = "";
    [headerCard, snowCard, valuationCard, futureCard, pastCard, healthCard,
     dividendCard, managementCard, ownershipCard, disclaimerCard]
      .forEach(function (f) { out.appendChild(el(f(d))); });
    var px = priceChart(d.overview.price_series);
    if (px) {
      out.firstChild.appendChild(el('<div class="figure">' + px + "</div>"));
    }
    out.hidden = false;
  }

  var REQUEST_TIMEOUT_MS = 120000;

  function run(symbol) {
    var sym = String(symbol || (input && input.value) || "")
      .replace(/[\s,]+/g, ",").split(",").filter(Boolean)[0];
    if (!sym) {
      // Silently doing nothing reads as a broken button. Say what is missing.
      setStatus("Enter a ticker first, then choose Company report.", true);
      if (input) input.focus();
      return;
    }
    sym = sym.toUpperCase();
    if (input) input.value = sym;
    out.hidden = true;

    var base = "Building the company report for " + sym + "…";
    setStatus(base);
    var t0 = Date.now();
    var tick = setInterval(function () {
      var s = Math.round((Date.now() - t0) / 1000);
      if (s >= 3 && !document.getElementById("status").classList.contains("err")) {
        setStatus(base + "  " + s + "s");
      }
    }, 1000);

    var ctrl = (typeof AbortController !== "undefined") ? new AbortController() : null;
    var killer = setTimeout(function () { if (ctrl) ctrl.abort(); }, REQUEST_TIMEOUT_MS);
    function done() { clearInterval(tick); clearTimeout(killer); }

    fetch("/api/company?symbol=" + encodeURIComponent(sym),
          ctrl ? { signal: ctrl.signal } : undefined)
      .then(function (r) {
        // The deployed build has no serverless twin for this route, so a 404 here is
        // the expected answer rather than a bug. Say which, and how to fix it.
        if (r.status === 404) {
          throw new Error("Company reports run locally only. Start the local server " +
            "with `python web/serve.py` and open the page it prints.");
        }
        return r.json().then(function (j) { return { ok: r.ok, body: j }; });
      })
      .then(function (res) {
        if (!res.ok || res.body.error) {
          throw new Error(res.body.error || "Request failed");
        }
        setStatus("");
        render(res.body);
        try {
          history.replaceState(null, "", "?report=" + encodeURIComponent(sym));
        } catch (e) { /* file:// and sandboxed frames reject this; harmless */ }
      })
      .catch(function (e) {
        if (e && e.name === "AbortError") {
          setStatus("Gave up after " + Math.round(REQUEST_TIMEOUT_MS / 1000) +
            "s — the data source is not responding. It rate-limits bursts, so wait a " +
            "moment and try again.", true);
        } else {
          setStatus((LQ.netMsg ? LQ.netMsg(e, "Could not build that report.")
                               : (e.message || "Could not build that report.")), true);
        }
      })
      .then(done);
  }

  window.LQReport = { run: run };

  // This file loads last, so the deep-link is read here rather than handed over from
  // app.js — by the time this line runs, everything it needs is defined.
  var deep = new URLSearchParams(location.search).get("report");
  if (deep) run(deep);
})();
