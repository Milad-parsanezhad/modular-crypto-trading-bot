import postgres from "postgres";

const EXPERIMENT = "v0.51";
const POLICY = "SIMULTANEOUS_MAX_EXPECTED_R_NO_PREEMPTION";
const PREREG = "d8ee4576aaf55750dd5910cc0d3b2efcbba3f5b2";
const CALENDAR = "6a8fa49de2d1befa9c17049aa61e13da20a028eb";
const IDENTITY_AMENDMENT = "4838c0d98c408d358929b0767676a5ac024bd8dd";
const EVIDENCE_ADDENDUM = "da5dcf5cd3568a92c75871016f000917c9380955";
const REST_FALLBACK_DOC = "be25c38293e25477beb68708ccb40c03a1ec2040";
const PREDICTOR_POLICY = "V47_C1_LATEST_CANONICAL_FOLD_PER_ASSET";
const PROSPECTIVE_START_MS = Date.parse("2026-09-13T12:00:00Z");
const PROSPECTIVE_END_MS = PROSPECTIVE_START_MS + 150 * 24 * 60 * 60 * 1000;
const MAX_CAPTURE_LAG_MINUTES = 60;
const BAR_MS = 4 * 60 * 60 * 1000;
const ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"];
const VENUES = ["coinex", "okx", "kucoin"];
const SOURCE_COMMIT = Bun.env.V51_SOURCE_COMMIT || "UNSET";
const DATABASE_URL = Bun.env.SHARED_DATABASE_URL || Bun.env.DATABASE_URL;

if (!DATABASE_URL) throw new Error("SHARED_DATABASE_URL/DATABASE_URL is required");
for (const name of ["LIVE_EXECUTION", "PAPER_EXECUTION", "KRAKEN_ENABLED"]) {
  if ((Bun.env[name] || "false").toLowerCase() === "true") {
    throw new Error(`${name}=true is forbidden for v0.51 research collector`);
  }
}

const sql = postgres(DATABASE_URL, {
  max: 1,
  idle_timeout: 10,
  connect_timeout: 15,
  onnotice: () => {},
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const iso = (ms) => new Date(ms).toISOString();
const canonicalSymbol = (asset) => `${asset}USDT`;

function finiteNumber(value, label) {
  const n = Number(value);
  if (!Number.isFinite(n)) throw new Error(`non-finite ${label}`);
  return n;
}

function normalizeBar({ venue, asset, openMs, open, high, low, close, volume, capturedMs, source }) {
  const o = finiteNumber(open, "open");
  const h = finiteNumber(high, "high");
  const l = finiteNumber(low, "low");
  const c = finiteNumber(close, "close");
  const v = finiteNumber(volume, "volume");
  const opened = finiteNumber(openMs, "open timestamp");
  const closed = opened + BAR_MS;
  if (closed > capturedMs) return null;
  if (Math.min(o, h, l, c) <= 0 || v < 0 || h < Math.max(o, c, l) || l > Math.min(o, c, h)) {
    throw new Error(`invalid OHLCV geometry ${venue} ${asset} ${iso(opened)}`);
  }
  const lag = (capturedMs - closed) / 60000;
  const eligible = closed >= PROSPECTIVE_START_MS && closed < PROSPECTIVE_END_MS && lag >= 0 && lag <= MAX_CAPTURE_LAG_MINUTES;
  return {
    first_seen_at: iso(capturedMs),
    venue,
    symbol: canonicalSymbol(asset),
    bar_open_time: iso(opened),
    bar_close_time: iso(closed),
    capture_lag_minutes: lag,
    prospective_eligible_v51: eligible,
    open: o,
    high: h,
    low: l,
    close: c,
    volume: v,
    source,
  };
}

async function fetchJson(url, attempts = 3) {
  let last;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      const response = await fetch(url, {
        method: "GET",
        headers: { "User-Agent": "modular-crypto-research-bot/v0.51" },
        signal: AbortSignal.timeout(20000),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      last = error;
      if (attempt < attempts) await sleep(750 * attempt);
    }
  }
  throw last;
}

async function fetchCoinEx(asset, capturedMs) {
  const url = new URL("https://api.coinex.com/v2/spot/kline");
  url.searchParams.set("market", canonicalSymbol(asset));
  url.searchParams.set("period", "4hour");
  url.searchParams.set("limit", "4");
  const payload = await fetchJson(url);
  if (Number(payload.code) !== 0 || !Array.isArray(payload.data)) throw new Error(`CoinEx response code ${payload.code}`);
  return payload.data
    .map((d) => normalizeBar({
      venue: "coinex",
      asset,
      openMs: d.created_at,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
      capturedMs,
      source: "coinex_v2_spot_kline_public",
    }))
    .filter(Boolean);
}

async function fetchOkx(asset, capturedMs) {
  const url = new URL("https://www.okx.com/api/v5/market/candles");
  url.searchParams.set("instId", `${asset}-USDT`);
  url.searchParams.set("bar", "4H");
  url.searchParams.set("limit", "4");
  const payload = await fetchJson(url);
  if (String(payload.code) !== "0" || !Array.isArray(payload.data)) throw new Error(`OKX response code ${payload.code}`);
  return payload.data
    .filter((row) => String(row[8]) === "1")
    .map((row) => normalizeBar({
      venue: "okx",
      asset,
      openMs: row[0],
      open: row[1],
      high: row[2],
      low: row[3],
      close: row[4],
      volume: row[5],
      capturedMs,
      source: "okx_v5_market_candles_public_confirmed",
    }))
    .filter(Boolean);
}

async function fetchKucoin(asset, capturedMs) {
  const url = new URL("https://api.kucoin.com/api/v1/market/candles");
  url.searchParams.set("symbol", `${asset}-USDT`);
  url.searchParams.set("type", "4hour");
  url.searchParams.set("startAt", String(Math.floor((capturedMs - 16 * 60 * 60 * 1000) / 1000)));
  url.searchParams.set("endAt", String(Math.floor(capturedMs / 1000)));
  const payload = await fetchJson(url);
  if (String(payload.code) !== "200000" || !Array.isArray(payload.data)) throw new Error(`KuCoin response code ${payload.code}`);
  return payload.data
    .map((row) => normalizeBar({
      venue: "kucoin",
      asset,
      openMs: Number(row[0]) * 1000,
      open: row[1],
      close: row[2],
      high: row[3],
      low: row[4],
      volume: row[5],
      capturedMs,
      source: "kucoin_v1_market_candles_public",
    }))
    .filter(Boolean);
}

async function fetchSeries(venue, asset, capturedMs) {
  if (venue === "coinex") return fetchCoinEx(asset, capturedMs);
  if (venue === "okx") return fetchOkx(asset, capturedMs);
  if (venue === "kucoin") return fetchKucoin(asset, capturedMs);
  throw new Error(`forbidden venue ${venue}`);
}

function valuesMatch(existing, row) {
  const eps = 1e-12;
  for (const col of ["open", "high", "low", "close", "volume"]) {
    if (Math.abs(Number(existing[col]) - Number(row[col])) > eps) return false;
  }
  return true;
}

async function initSchema() {
  await sql.unsafe(`CREATE SCHEMA IF NOT EXISTS v51_research`);
  await sql.unsafe(`
    CREATE TABLE IF NOT EXISTS v51_research.raw_ohlcv (
      first_seen_at timestamptz NOT NULL,
      venue text NOT NULL,
      symbol text NOT NULL,
      bar_open_time timestamptz NOT NULL,
      bar_close_time timestamptz NOT NULL,
      capture_lag_minutes double precision NOT NULL,
      prospective_eligible_v51 boolean NOT NULL,
      open double precision NOT NULL,
      high double precision NOT NULL,
      low double precision NOT NULL,
      close double precision NOT NULL,
      volume double precision NOT NULL,
      source text NOT NULL,
      source_commit text NOT NULL,
      PRIMARY KEY (venue, symbol, bar_open_time)
    )
  `);
  await sql.unsafe(`
    CREATE TABLE IF NOT EXISTS v51_research.collection_runs (
      run_id text NOT NULL,
      captured_at timestamptz NOT NULL,
      venue text NOT NULL,
      symbol text NOT NULL,
      status text NOT NULL,
      bars_received integer NOT NULL,
      closed_bars_received integer NOT NULL,
      eligible_bars_received integer NOT NULL,
      latest_closed_bar timestamptz,
      error_type text,
      error_message text,
      source_commit text NOT NULL,
      PRIMARY KEY (run_id, venue, symbol)
    )
  `);
  await sql.unsafe(`
    CREATE TABLE IF NOT EXISTS v51_research.run_manifests (
      run_id text PRIMARY KEY,
      captured_at timestamptz NOT NULL,
      manifest jsonb NOT NULL
    )
  `);
}

async function persistBar(row) {
  const existing = await sql`
    SELECT first_seen_at, capture_lag_minutes, prospective_eligible_v51,
           open, high, low, close, volume, source, source_commit
    FROM v51_research.raw_ohlcv
    WHERE venue = ${row.venue}
      AND symbol = ${row.symbol}
      AND bar_open_time = ${row.bar_open_time}::timestamptz
    LIMIT 1
  `;
  if (existing.length) {
    if (!valuesMatch(existing[0], row)) {
      throw new Error(`CLOSED_BAR_REVISION ${row.venue} ${row.symbol} ${row.bar_open_time}`);
    }
    return "existing";
  }
  await sql`
    INSERT INTO v51_research.raw_ohlcv (
      first_seen_at, venue, symbol, bar_open_time, bar_close_time,
      capture_lag_minutes, prospective_eligible_v51,
      open, high, low, close, volume, source, source_commit
    ) VALUES (
      ${row.first_seen_at}::timestamptz, ${row.venue}, ${row.symbol},
      ${row.bar_open_time}::timestamptz, ${row.bar_close_time}::timestamptz,
      ${row.capture_lag_minutes}, ${row.prospective_eligible_v51},
      ${row.open}, ${row.high}, ${row.low}, ${row.close}, ${row.volume},
      ${row.source}, ${SOURCE_COMMIT}
    )
    ON CONFLICT (venue, symbol, bar_open_time) DO NOTHING
  `;
  return "inserted";
}

async function persistAudit(audit) {
  await sql`
    INSERT INTO v51_research.collection_runs (
      run_id, captured_at, venue, symbol, status, bars_received,
      closed_bars_received, eligible_bars_received, latest_closed_bar,
      error_type, error_message, source_commit
    ) VALUES (
      ${audit.run_id}, ${audit.captured_at}::timestamptz, ${audit.venue}, ${audit.symbol},
      ${audit.status}, ${audit.bars_received}, ${audit.closed_bars_received},
      ${audit.eligible_bars_received}, ${audit.latest_closed_bar || null}::timestamptz,
      ${audit.error_type || null}, ${audit.error_message || null}, ${SOURCE_COMMIT}
    )
    ON CONFLICT (run_id, venue, symbol) DO NOTHING
  `;
}

async function sha256Text(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function evidenceDigest() {
  const rows = await sql`
    SELECT first_seen_at, venue, symbol, bar_open_time, bar_close_time,
           capture_lag_minutes, prospective_eligible_v51,
           open, high, low, close, volume, source, source_commit
    FROM v51_research.raw_ohlcv
    ORDER BY venue, symbol, bar_open_time
  `;
  const canonical = rows.map((row) => ({
    first_seen_at: new Date(row.first_seen_at).toISOString(),
    venue: row.venue,
    symbol: row.symbol,
    bar_open_time: new Date(row.bar_open_time).toISOString(),
    bar_close_time: new Date(row.bar_close_time).toISOString(),
    capture_lag_minutes: Number(row.capture_lag_minutes),
    prospective_eligible_v51: Boolean(row.prospective_eligible_v51),
    open: Number(row.open),
    high: Number(row.high),
    low: Number(row.low),
    close: Number(row.close),
    volume: Number(row.volume),
    source: row.source,
    source_commit: row.source_commit,
  }));
  return sha256Text(JSON.stringify(canonical));
}

async function main() {
  const capturedMs = Date.now();
  const capturedAt = iso(capturedMs);
  if (capturedMs < PROSPECTIVE_START_MS) {
    console.log(JSON.stringify({ status: "PRESTART_NO_COLLECTION", captured_at: capturedAt }));
    return;
  }
  if (capturedMs >= PROSPECTIVE_END_MS) {
    console.log(JSON.stringify({ status: "WINDOW_COMPLETE_NO_COLLECTION", captured_at: capturedAt }));
    return;
  }
  if (SOURCE_COMMIT === "UNSET") throw new Error("V51_SOURCE_COMMIT must be pinned");

  await initSchema();
  const runId = `v51-${capturedAt.replace(/[^0-9]/g, "")}-${Bun.env.RAILWAY_DEPLOYMENT_ID || "cron"}`;
  const audits = [];

  for (const venue of VENUES) {
    for (const asset of ASSETS) {
      const symbol = canonicalSymbol(asset);
      const audit = {
        run_id: runId,
        captured_at: capturedAt,
        venue,
        symbol,
        status: "UNKNOWN",
        bars_received: 0,
        closed_bars_received: 0,
        eligible_bars_received: 0,
        latest_closed_bar: null,
        error_type: null,
        error_message: null,
      };
      try {
        const rows = await fetchSeries(venue, asset, capturedMs);
        audit.bars_received = rows.length;
        audit.closed_bars_received = rows.length;
        audit.eligible_bars_received = rows.filter((r) => r.prospective_eligible_v51).length;
        audit.latest_closed_bar = rows.length
          ? rows.map((r) => r.bar_close_time).sort().at(-1)
          : null;
        for (const row of rows) await persistBar(row);
        audit.status = rows.length ? "OK" : "NO_CLOSED_BARS";
      } catch (error) {
        audit.status = String(error?.message || "").startsWith("CLOSED_BAR_REVISION")
          ? "REVISION_ERROR"
          : "FETCH_OR_PERSIST_ERROR";
        audit.error_type = error?.constructor?.name || "Error";
        audit.error_message = String(error?.message || error).slice(0, 500);
      }
      await persistAudit(audit);
      audits.push(audit);
    }
  }

  const counts = await sql`
    SELECT
      COUNT(*)::int AS raw_rows_total,
      COUNT(*) FILTER (WHERE prospective_eligible_v51)::int AS raw_rows_prospective_eligible,
      COUNT(DISTINCT (venue, symbol)) FILTER (WHERE prospective_eligible_v51)::int AS eligible_series
    FROM v51_research.raw_ohlcv
  `;
  const okSeries = audits.filter((a) => a.status === "OK").length;
  const manifest = {
    experiment: EXPERIMENT,
    role: "raw_public_market_evidence_only",
    transport: "railway_bun_direct_rest_postgres",
    arbitration_policy: POLICY,
    preregistration_commit: PREREG,
    calendar_commit: CALENDAR,
    predictor_identity_amendment_commit: IDENTITY_AMENDMENT,
    evidence_integrity_addendum_commit: EVIDENCE_ADDENDUM,
    railway_rest_fallback_doc_commit: REST_FALLBACK_DOC,
    source_commit: SOURCE_COMMIT,
    predictor_identity_policy: PREDICTOR_POLICY,
    prospective_start: iso(PROSPECTIVE_START_MS),
    prospective_end: iso(PROSPECTIVE_END_MS),
    max_capture_lag_minutes: MAX_CAPTURE_LAG_MINUTES,
    captured_at: capturedAt,
    run_id: runId,
    venues: VENUES,
    assets: ASSETS,
    successful_series_this_run: okSeries,
    expected_series_per_run: VENUES.length * ASSETS.length,
    raw_rows_total: Number(counts[0]?.raw_rows_total || 0),
    raw_rows_prospective_eligible: Number(counts[0]?.raw_rows_prospective_eligible || 0),
    eligible_series: Number(counts[0]?.eligible_series || 0),
    raw_table_sha256: await evidenceDigest(),
    kraken_touched: false,
    paper_execution: false,
    live_execution: false,
    scoring_performed: false,
    arbitration_performed: false,
  };
  await sql`
    INSERT INTO v51_research.run_manifests (run_id, captured_at, manifest)
    VALUES (${runId}, ${capturedAt}::timestamptz, ${JSON.stringify(manifest)}::jsonb)
    ON CONFLICT (run_id) DO NOTHING
  `;
  console.log(JSON.stringify(manifest));
}

try {
  await main();
} finally {
  await sql.end({ timeout: 5 });
}
