# v0.20 Phase-Q Automation Map

`v19-forward-microstructure` (every 30 min target)
→ public CoinEx/OKX/KuCoin fixed-window snapshot
→ Phase-Q protocol + CI provenance stamp
→ final payload SHA-256
→ immutable `v19-forward-microstructure-<run_id>` artifact

`v20-phase-q-monitor` (every 6 h)
→ list retained GitHub Actions artifacts
→ download eligible v0.19 forward artifacts
→ reconstruct evidence directory
→ validate protocol/provenance/hash
→ exclude manual/PR/branch/pre-start/tampered evidence
→ deduplicate 30-min slots
→ compute 168h/336-opportunity maturity and BTC/ETH quality statistics
→ create evidence-ledger SHA-256
→ upload JSON + Markdown + harvest manifest

No arrow in this map reaches PAPER strategy replacement, testnet or LIVE execution.
