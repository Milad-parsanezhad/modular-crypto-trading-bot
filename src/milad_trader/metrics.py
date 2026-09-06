"""Economic evaluation with explicit units and dependence-aware intervals."""
import numpy as np


def performance(curve, fills, bars_per_year):
    equity = curve.equity.to_numpy(float)
    ret = equity[1:] / equity[:-1] - 1
    total = equity[-1]/equity[0]-1
    sd = np.std(ret, ddof=1) if len(ret)>1 else 0
    downside = np.sqrt(np.mean(np.minimum(ret, 0)**2))
    annual_return = np.expm1(np.log1p(total)*bars_per_year/max(len(ret),1))
    mdd = float(np.min(equity/np.maximum.accumulate(equity)-1))
    pnl = fills.loc[fills.side == "sell", "pnl"].astype(float).to_numpy() if len(fills) else np.array([])
    wins, losses = pnl[pnl>0].sum(), -pnl[pnl<0].sum()
    q = np.quantile(ret, 0.05)
    return dict(total_return=float(total), annualized_return=float(annual_return),
                sharpe=float(np.mean(ret)/sd*np.sqrt(bars_per_year)) if sd>1e-12 else None,
                sortino=float(np.mean(ret)/downside*np.sqrt(bars_per_year)) if downside>1e-12 else None,
                max_drawdown=mdd, calmar=float(annual_return/abs(mdd)) if mdd<0 else None,
                profit_factor=float(wins/losses) if losses>0 else None,
                completed_trades=int(len(pnl)), win_rate=float(np.mean(pnl>0)) if len(pnl) else None,
                fees=float(fills.fee.sum()) if len(fills) else 0.,
                cvar_95_loss=float(-np.mean(ret[ret<=q])), bars=int(len(ret)),
                turnover_notional=float((fills.price*fills.quantity).sum()/equity[0]) if len(fills) else 0.)


def paired_block_interval(a, b, block=12, draws=1000, seed=42):
    """Circular moving-block bootstrap CI of mean per-bar return DIFFERENCE.

    Descriptive 95% interval; not a multiplicity-adjusted significance claim.
    Call within each contiguous test fold (never across gaps or seeds).
    """
    diff = np.asarray(a)-np.asarray(b)
    if len(diff)<2 or not np.isfinite(diff).all():
        raise ValueError("Need paired finite returns")
    rng = np.random.default_rng(seed)
    n = len(diff)
    block = min(block,n)
    starts = rng.integers(0,n,size=(draws,int(np.ceil(n/block))))
    idx = (starts[...,None]+np.arange(block)) % n
    means = diff[idx.reshape(draws,-1)[:,:n]].mean(axis=1)
    return dict(mean_difference=float(diff.mean()), lower=float(np.quantile(means,.025)),
                upper=float(np.quantile(means,.975)), block_bars=block, draws=draws,
                multiple_testing_adjusted=False)
