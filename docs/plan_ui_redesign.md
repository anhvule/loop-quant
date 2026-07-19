# Plan: Professional Financial UI for the Web App

## Context — why and what "professional" means here

The app just became a single unified analysis (one input → scorecard + verdict + optional
forecast), but the UI still looks like a hobby tool: generic cards, default-blue accent,
plain tables, emoji warnings. The goal is the look and feel of an **institutional
research terminal** — the register of Koyfin / a broker research portal: dense but
readable data tables, tabular numerals, a restrained palette where color only ever means
something, strong typographic hierarchy, and dark mode as a first-class selected theme
rather than an automatic inversion.

Hard constraints (all already properties of this repo — the redesign must not break them):

- **No build step, no frameworks, no external requests.** `web/README.md` promises "no
  external JS/CSS/font requests". System font stack only; every asset inline or local.
- **Zero logic changes.** This is `styles.css` + markup strings in `index.html`/`app.js`
  render functions. The `/api/analyse` payload, the vendored engines, and all 542 tests
  are untouched. If a change needs an engine edit, it is out of scope.
- **The honesty rails are the brand.** Banners ("NOT INVESTMENT ADVICE", "not merged",
  thresholds printed beside grades) must become *more* prominent through design, never
  styled away. A professional financial UI that hides its disclaimers is a worse UI.
- **Encoding discipline:** all file edits via Write/Edit tools, never PowerShell
  `Get-Content`/`Set-Content` (that corrupted `app.js` once already — see the em-dash
  mojibake incident). `web/app.js` contains literal `—`, `·`, `“ ”`, `→`, `★`, `⚠`.

## Design tokens (drop-in `:root` block for styles.css)

Adopt the validated reference palette (light + dark are both *selected*, not flipped).
Replace the current `:root` variables wholesale:

```css
:root{
  /* planes */
  --bg:#f9f9f7; --card:#fcfcfb; --line:#e1e0d9; --hairline:rgba(11,11,11,.10);
  /* ink */
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  /* accent (one, used sparingly: primary action, focus, links, cone band) */
  --accent:#2a78d6; --accent-ink:#1c5cab;
  /* status -- reserved for grades/verdicts, never decorative */
  --good:#0ca30c; --good-text:#006300; --warn-s:#fab219; --serious:#ec835a;
  --bad:#d03b3b;
  /* charts */
  --band:#2a78d6; --grid:#e1e0d9; --axis:#c3c2b7;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d0d0d; --card:#1a1a19; --line:#2c2c2a; --hairline:rgba(255,255,255,.10);
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --accent:#3987e5; --accent-ink:#6da7ec;
    --good:#0ca30c; --good-text:#0ca30c; --warn-s:#fab219; --serious:#ec835a;
    --bad:#d03b3b;
    --band:#3987e5; --grid:#2c2c2a; --axis:#383835;
  }
}
```

Wave-chart colors are currently hardcoded in `app.js` (`#54a24b`, `#3f8438`, `#b3232c`,
`#e39146`) — swap to `var(--good-text)`, `var(--bad)`, `var(--serious)` so both themes
work (today those greens/reds are invisible-ish on dark).

Status colors on the light surface: `--warn-s` and `--serious` are sub-3:1 **by design**
— every pill already carries a text label (OK / warn / FAIL), so color never stands
alone. Keep that pairing; never introduce a color-only indicator.

Typography: system stack stays (`-apple-system, "Segoe UI", Roboto, …`). Add a scale —
page title 1.35rem/650, card titles 0.95rem/650, body 0.92rem, table 0.86rem, micro
labels 0.72rem uppercase with `letter-spacing:.06em` in `--muted`. `font-variant-numeric:
tabular-nums` on every table cell, metric, and axis tick (already partially present —
make it universal via a `.num` utility and table defaults).

## Section-by-section spec

### 1. Top bar (index.html)

Replace the plain `<header>` with a slim sticky top bar: left — a wordmark
("LOOP QUANT" in micro-caps over "Stock Risk Outlook"); right — a compact
"research tool · not investment advice" micro-label. Height ~52px, `--card`
background, hairline bottom border. The existing `.banner` paragraph moves to a
single-line **notice strip** directly under the top bar (amber-tinted background,
⚠ replaced by a small inline SVG triangle icon so it renders identically on all
platforms), with a `<details>` for the full text so the rail is always visible but
never shouty.

### 2. Command bar (the form)

Style the unified form as a terminal command bar in a card: ticker input grows,
monospace-feel uppercase input (`text-transform:uppercase` exists; add
`font-variant-numeric` and wider letter-spacing), selects restyled with a shared
`.field` wrapper that puts a micro-label ("HORIZON", "RISK-FREE") *above* each control —
selects lose their inline "Risk-free 4%" hack and become labeled controls. Primary
button: `--accent` fill, white text, subtle 1px darker border, `:focus-visible` ring
(2px `--accent` outline offset 2px — add globally; keyboard focus is currently browser
default). Preset chips: ghost buttons with hairline borders that fill on hover;
group the two chip clusters with micro-labels "ONE NAME" / "COHORTS".

### 3. Status / loading

Replace the plain status box with: (a) an inline spinner (pure CSS, 16px, `--accent`)
next to the message while fetching; (b) error state keeps the red border but gains the
triangle icon. No skeletons — the payload arrives all at once; a honest spinner with the
"running 8,000 simulations" message is better than fake placeholders.

### 4. Regime strip (app.js `regimeCard`)

Convert from a generic card into a **market status strip**: a horizontal row —
big label "RISK-ON" / "RISK-OFF" (good/bad status color, 0.95rem, 700, with a filled
dot), then "SPY 743.29 vs 200d SMA 696.69" in tabular nums, then the gap as a signed
delta chip (`+6.7%` in `--good-text`), then right-aligned "rank basis · rf 4.0%" in
muted micro text. The long methodology sentences (quartile fallback, CLI note,
stand-aside explanation) collapse into a `<details class="fine">` labeled
"methodology" inside the strip. The two-readings banner stays visible as one bold
sentence, not a wall.

### 5. The cohort table (app.js `table`)

This is the centerpiece; make it read like a terminal grid:

- Sticky header (`position:sticky; top:0`) with `--card` background, micro-cap column
  labels, hairline bottom rule. The `.scroll` wrapper gets `max-height:70vh` so long
  cohorts scroll under the frozen header.
- Row hover: faint accent wash (`color-mix(in srgb, var(--accent) 6%, transparent)`).
- First column: ticker bold + price beneath in muted 0.78rem (exists — keep).
- **Verdict column**: replace the text pill with a stronger badge — 3px left border in
  the status color + tinted background + 0.72rem caps text. `stand_aside` and
  `borderline` amber; `investable` green; `reckless` red; excluded/insufficient gray.
- **Gate/criteria pills**: shrink to compact glyph chips — `OK` `w` `✗` `–` (still text,
  still title-tooltips with value + threshold). Column count is 17; compact chips are
  what keeps it scannable. Numeric columns (β, Treynor, Score) right-aligned tabular.
- Group separators: a 2px `--line` rule between verdict groups (investable block,
  borderline block, …) — the sort already groups them; add `data-verdict` on rows and
  draw the rule when it changes.
- Verdict summary chips above the table become count badges in the same badge style.

### 6. Detail expanders

Keep `<details>`, restyle: summary row gets ticker (bold) + name (muted) + verdict badge
right-aligned; open state shows the two labeled columns ("VERDICT GATES" /
"RISK SCORECARD" micro-labels) in a two-column grid on wide screens (`grid-template-
columns:1fr 1fr; gap:1.2rem` above 900px, stacked below). Threshold lines stay muted
0.8rem — that's the honesty rail, keep every word.

### 7. Sliders card

Retitle "Scorecard weights" with the existing hint. Range inputs: styled track
(4px hairline, `--accent` filled portion via `accent-color:var(--accent)` — one line,
native, no JS) and the value shown in a small mono chip. Collapse the whole card into a
`<details>` open-by-default so cohort scanners can fold it away.

### 8. Forecast block (single ticker)

- `header`: merge into a **hero strip**: ticker + name left; right side a metric row —
  last close (1.5rem, 650), then beta ±se, daily vol, bars as labeled micro-metrics.
- `headline`: the four range figures become **stat tiles** in a responsive grid
  (label micro-caps, value 1.3rem tabular, delta chip signed and colored good/bad).
  "median (assumption)" keeps that exact label — it is load-bearing.
- Cone chart: gridlines to `--grid` at 1px, axis text `--muted` 10px, band fill
  `var(--band)` at 14% opacity with a 2px `var(--band)` median line, today-line
  hairline dashed `--axis`. Add a hover crosshair: a vertical line + tooltip showing
  month / p5 / median / p95 at the nearest milestone (pure JS on the existing SVG,
  ~30 lines, no library — per the dataviz interaction spec; milestones are few so
  snap-to-nearest is trivial).
- Wave chart: token swap as above; star glyphs keep their `★` + label (identity never
  color-alone); legend line uses real chips instead of colored `■` characters.

### 9. Footer

Two-column: method summary left (existing text, tightened), data-source + disclaimer
right, both 0.8rem `--muted`, hairline top rule. Add "SPY benchmark · Yahoo Finance
daily closes · local analysis, nothing leaves your machine except price requests".

### 10. Responsive

- ≤900px: detail grids stack; stat tiles wrap 2-up; command bar wraps (input full
  width, controls in a row below).
- The 17-column table does NOT reflow — it scrolls horizontally inside `.scroll`
  (already true). Add `scroll-shadow` affordances: left/right inset gradients via
  `background-attachment:local` trick (pure CSS) so truncation is visible.
- Print: `@media print` — hide form/sliders, black-on-white, keep table + banners.

## Implementation order

1. `styles.css` — full rewrite around the token block (sections 1–3, 5–7, 9, 10 are
   mostly CSS). Keep every existing class name that app.js emits; add new ones rather
   than renaming where possible.
2. `index.html` — top bar, notice strip, labeled command bar, chip group labels.
3. `app.js` — markup-string edits only: regime strip, verdict badges, compact gate
   chips, group separators, stat tiles, hero strip, wave-chart color tokens, cone
   crosshair (the one genuinely new JS, ~30 lines, isolated in `cone()`).
4. Palette validation: `node scripts/validate_palette.js` (dataviz skill,
   `C:\Users\anhvu\AppData\Local\Temp\claude\bundled-skills\...\dataviz\scripts\`) on
   the wave/status set against both surfaces — or skip if node is unavailable and note
   that the reference palette ships pre-validated (it does; only re-run if values are
   changed from the tables above).
5. Verification (below), then update the `web/README.md` layout section (it still
   describes the three-mode UI — rewrite the "Three modes" section to "One analysis"
   while in there).

## Verification

- `python -m pytest -q` — must stay 542 green (nothing here touches Python, so any
  failure means scope creep; stop and investigate).
- Live drive via the browser pane (`preview_start` name "web", stop/start to defeat
  none — cache is already `no-store`):
  - Single ticker (NVDA): hero strip, stat tiles, cone with working crosshair
    tooltip, wave chart legible in BOTH themes.
  - Cohort (the 20-name high-beta preset): sticky header while scrolling, group
    separators between verdict blocks, hover wash, badge colors, slider re-render
    still preserves focus.
  - Error path: bogus ticker → styled error; dead server → the netMsg guidance.
  - Both themes: `resize_window` with `colorScheme:"light"` then `"dark"` — verify
    ink contrast, chart gridlines, badges in each; check `read_console_messages`
    clean after every render.
  - Mojibake scan after all edits: grep `web/` for `â€|Ã‚|�` (the session's encoding
    canary) — must be zero.
- Eyeball pass per the dataviz procedure step 7: label collisions in the cone axis,
  table header truncation at 1280px and at ~portrait width.

## Out of scope (say no if tempted)

- Any JS framework, bundler, icon font, or webfont download.
- Merging scorecard and verdict into one visual score (explicitly refused upstream).
- New API fields or engine changes (if a design wants data the payload lacks, cut the
  design element, not the parity).
- Removing/toning down any disclaimer or threshold text.
