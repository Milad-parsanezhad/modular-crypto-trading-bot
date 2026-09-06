"""Summarize a completed market run without selecting a test-set winner."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from milad_trader.data import read_csv
from milad_trader.features import make_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--quality', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['source_kind'] == 'historical'
    raw = read_csv(args.data, manifest['config']['timeframe'])
    frame = make_features(raw)
    quality = json.loads(Path(args.quality).read_text())
    if not quality['passed']:
        raise ValueError('Data quality gate has not passed')
    import hashlib
    if quality['data_sha256'] != hashlib.sha256(Path(args.data).read_bytes()).hexdigest():
        raise ValueError('Quality report refers to different candles')
    shutil.copy2(args.quality, out/'data_quality.json')
    metrics = pd.read_csv(run/'metrics.csv')
    seed_results=[]
    for (model,features), group in metrics.groupby(['model','features']):
        if group.seed.isna().all():
            seed_results.append(dict(model=model,features=features,seed=None,
                compounded_return=float((1+group.total_return).prod()-1),total_trades=int(group.completed_trades.sum())))
        else:
            for seed,g in group.groupby('seed'):
                seed_results.append(dict(model=model,features=features,seed=int(seed),
                    compounded_return=float((1+g.total_return).prod()-1),total_trades=int(g.completed_trades.sum())))
    seeds=pd.DataFrame(seed_results)
    seeds.to_csv(out/'per_seed_summary.csv',index=False)
    summary=seeds.groupby(['model','features']).agg(median_compounded_return=('compounded_return','median'),
                min_compounded_return=('compounded_return','min'),max_compounded_return=('compounded_return','max'),
                median_total_trades=('total_trades','median')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    boundaries=[]
    for fold in manifest['folds']:
        row={'fold':fold['number']}
        for stage in ['train','validation','test']:
            a,b=fold[stage]
            row[stage+'_first']=str(frame.index[a]);row[stage+'_last']=str(frame.index[b-1])
        boundaries.append(row)
    pd.DataFrame(boundaries).to_csv(out/'time_boundaries.csv',index=False)
    for name in ['manifest.json','metrics.csv','cost_sensitivity.csv','regimes.csv','bootstrap_intervals.json']:
        if (run/name).exists():shutil.copy2(run/name,out/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=summary[summary.features!='no_ichi'].sort_values('model')
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    vals=groups.median_compounded_return.to_numpy()*100
    low=vals-groups.min_compounded_return.to_numpy()*100
    high=groups.max_compounded_return.to_numpy()*100-vals
    ax.barh(groups.model,vals,xerr=np.array([low,high]),color=['#167d8d' if v>=0 else '#ae4242' for v in vals],capsize=3)
    ax.axvline(0,color='black',lw=.7)
    ax.set_xlabel('Compounded return over three test folds (%)')
    ax.set_title('BTC/USDT 4h | Real Binance spot data\nMedian across seeds; whiskers = seed range, not confidence intervals')
    ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'market_returns.png',dpi=160)
    print(json.dumps(quality,indent=2))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
