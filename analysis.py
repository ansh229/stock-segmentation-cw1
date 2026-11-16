"""6BUIS017W stock segmentation. Real data only; no fabricated fallback."""
from pathlib import Path
from datetime import datetime, timezone
from io import StringIO
import argparse
import hashlib
import importlib.metadata
import json
import time
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

START, END, SEED = '2021-01-01', '2025-01-01', 42
FEATURES = ['MeanWeeklyReturn', 'WeeklyBeta', 'Recency', 'Frequency', 'Monetary']
VERSION = '1.0.0'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str))


def acquire(root, offline=False):
    """Save separate ticker histories so interrupted downloads can be resumed."""
    raw = root / 'data'
    raw.mkdir(parents=True, exist_ok=True)
    universe_path = raw / 'constituents.csv'
    manifest_path = raw / 'provenance.json'
    if not universe_path.exists():
        if offline:
            raise RuntimeError('Cached constituents.csv is required in offline mode.')
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=45) as response:
            html = response.read().decode('utf-8')
        (raw / 'constituents_source.html').write_text(html)
        universe = next(t for t in pd.read_html(StringIO(html)) if 'Symbol' in t.columns)
        universe['YahooTicker'] = universe['Symbol'].str.replace('.', '-', regex=False)
        universe.to_csv(universe_path, index=False)
        write_json(manifest_path, {'universe_source': url,
                   'universe_retrieved_utc': datetime.now(timezone.utc).isoformat(),
                   'start_inclusive': START, 'end_exclusive': END,
                   'price_source': 'Yahoo Finance via yfinance',
                   'adjustment_policy': 'auto_adjust=False; split-adjusted Close and compatible Volume; Adj Close for returns',
                   'histories': {}})
    if not manifest_path.exists():
        raise RuntimeError('Cached data require provenance.json; do not invent retrieval dates.')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('start_inclusive'), manifest.get('end_exclusive')) != (START, END):
        raise RuntimeError('Cache period does not match the required study period.')
    universe = pd.read_csv(universe_path)
    tickers = ['^GSPC'] + universe['YahooTicker'].tolist()
    failures = []
    for number, ticker in enumerate(tickers, 1):
        path = raw / (ticker + '.csv')
        if path.exists():
            if ticker not in manifest['histories']:
                raise RuntimeError(f'No provenance for cached {ticker}')
            expected = manifest['histories'][ticker].get('sha256')
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Cache checksum mismatch for {ticker}')
            continue
        if offline:
            failures.append({'Share': ticker, 'Reason': 'No cached history'})
            continue
        import yfinance as yf
        for attempt in range(3):
            try:
                d = yf.Ticker(ticker).history(start=START, end=END,
                        auto_adjust=False, actions=True, repair=False, raise_errors=True)
                if d.empty:
                    raise ValueError('Empty history')
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                d.index.name = 'Date'
                d.to_csv(path)
                manifest['histories'][ticker] = {
                    'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                write_json(manifest_path, manifest)
                print(f'{number}/{len(tickers)} {ticker}: saved', flush=True)
                break
            except Exception as error:
                if attempt == 2:
                    failures.append({'Share': ticker, 'Reason': str(error)})
                else:
                    time.sleep(2 ** attempt)
        time.sleep(0.25)
    pd.DataFrame(failures, columns=['Share', 'Reason']).to_csv(raw / 'download_failures.csv', index=False)
    return universe


def read_history(path):
    d = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
    return d.loc[(d.index >= START) & (d.index < END)]


def weekly_prices(prices):
    """Keep full calendar weeks only, including ordinary holiday-shortened weeks."""
    w = prices.resample('W-FRI').last()
    monday = w.index - pd.Timedelta(days=4)
    return w.loc[(monday >= pd.Timestamp(START)) & (w.index < pd.Timestamp(END))]


def activity(d, observation):
    baseline = d['Volume'].shift(1).rolling(20, min_periods=20).mean()
    events = baseline.notna() & (d['Volume'] > 2 * baseline)
    dates = d.index[events]
    recency = (observation - dates[-1]).days if len(dates) else np.nan
    recent = d.tail(60)
    assert len(recent) == 60 and baseline.loc[recent.index].notna().all()
    detail = pd.DataFrame({'Close': d['Close'], 'Volume': d['Volume'],
                           'Previous20Mean': baseline, 'Threshold': 2 * baseline,
                           'HighVolume': events, 'DailyTradedValue': d['Close'] * d['Volume']})
    return recency, int(events.loc[recent.index].sum()), float(detail.loc[recent.index, 'DailyTradedValue'].sum()), detail


def prepare(root, universe):
    out = root / 'outputs'
    out.mkdir(exist_ok=True)
    benchmark = read_history(root / 'data' / '^GSPC.csv')
    if benchmark.index.duplicated().any():
        raise ValueError('Duplicate benchmark dates')
    if benchmark.empty or benchmark.index.min() > pd.Timestamp('2021-01-04') or benchmark.index.max() < pd.Timestamp('2024-12-31'):
        raise ValueError('Benchmark lacks required period endpoints')
    for col in ['Close', 'Adj Close']:
        if col not in benchmark or benchmark[col].isna().any() or (benchmark[col] <= 0).any():
            raise ValueError('Invalid benchmark ' + col)
    sessions = benchmark.index
    observation = sessions[-1]
    market = weekly_prices(benchmark['Adj Close']).pct_change(fill_method=None)
    rows, log, corporate = [], [], []
    for _, share in universe.iterrows():
        ticker = share['YahooTicker']
        try:
            d = read_history(root / 'data' / (ticker + '.csv'))
            if d.index.duplicated().any():
                raise ValueError('Duplicate dates')
            if not d.index.equals(sessions):
                raise ValueError('Unequal or missing daily history; strict full-period coverage rule')
            for col in ['Close', 'Adj Close', 'Volume']:
                if col not in d or not np.isfinite(d[col]).all():
                    raise ValueError('Missing or nonfinite ' + col)
            if (d[['Close', 'Adj Close']] <= 0).any().any():
                raise ValueError('Nonpositive price')
            if (d['Volume'] < 0).any() or (d['Volume'] % 1 != 0).any():
                raise ValueError('Negative or fractional volume')
            if (d['Volume'] == 0).any():
                raise ValueError('Zero volume on benchmark trading session')
            splits = d.get('Stock Splits', pd.Series(0., index=d.index)).fillna(0)
            dividends = d.get('Dividends', pd.Series(0., index=d.index)).fillna(0)
            for date in d.index[(splits != 0) | (dividends != 0)]:
                corporate.append({'Share': ticker, 'Date': date, 'Split': splits.loc[date],
                                  'Dividend': dividends.loc[date], 'Close': d.loc[date, 'Close'],
                                  'AdjClose': d.loc[date, 'Adj Close']})
            # Yahoo Close already accounts for splits: applying split ratios again is incorrect.
            # Large close jumps near split dates are flagged for review, not automatically repaired.
            jumps = d['Close'].pct_change(fill_method=None).abs()
            split_near = (splits != 0).rolling(3, center=True, min_periods=1).max().astype(bool)
            if (split_near & (jumps > .8)).any():
                raise ValueError('Extreme close jump near split; manual adjustment review required')
            returns = weekly_prices(d['Adj Close']).pct_change(fill_method=None)
            pair = pd.concat([returns.rename('Share'), market.rename('Market')], axis=1).dropna()
            if len(pair) != market.notna().sum() or len(pair) < 100 or pair['Market'].var() <= 0:
                raise ValueError('Insufficient or misaligned weekly returns')
            beta = pair['Share'].cov(pair['Market']) / pair['Market'].var()
            r, f, m, detail = activity(d, observation)
            risk_ok = np.isfinite([returns.mean(), beta]).all()
            rows.append({'Share': ticker, 'Security': share['Security'],
                'Sector': share['GICS Sector'], 'MeanWeeklyReturn': pair['Share'].mean(),
                'WeeklyBeta': beta, 'PairedWeeks': len(pair), 'Recency': r,
                'Frequency': f, 'Monetary': m, 'ObservationDate': observation.date()})
            log.append({'Share': ticker, 'RiskIncluded': bool(risk_ok), 'RFMIncluded': pd.notna(r),
                        'Reason': 'Valid' if pd.notna(r) else 'No qualifying event in valid study history'})
            if not (out / 'worked_example_daily.csv').exists() and pd.notna(r):
                detail.to_csv(out / 'worked_example_daily.csv')
                detail.tail(60).to_csv(out / 'worked_example_60_sessions.csv')
                dates = detail.index[detail['HighVolume']]
                write_json(out / 'worked_example.json', {'Share': ticker,
                    'ObservationDate': observation.date(), 'LatestEventDate': dates[-1].date(),
                    'Recency': r, 'Frequency': f, 'Monetary': m,
                    'WindowStart': detail.tail(60).index[0].date()})
        except (ValueError, KeyError, FileNotFoundError) as error:
            log.append({'Share': ticker, 'RiskIncluded': False, 'RFMIncluded': False, 'Reason': str(error)})
    pd.DataFrame(log).to_csv(out / 'inclusion_exclusion_log.csv', index=False)
    pd.DataFrame(corporate).to_csv(out / 'corporate_actions.csv', index=False)
    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError('No valid shares; inspect validation and download logs.')
    features.to_csv(out / 'features_before_common_filter.csv', index=False)
    features = features.dropna(subset=FEATURES).set_index('Share').sort_index()
    if len(features) < 10:
        raise RuntimeError('Fewer than ten common valid shares; resolve data coverage before modelling.')
    features.to_csv(out / 'features.csv')
    features[FEATURES].describe().to_csv(out / 'feature_summary.csv')
    return features


def plot_save(out, name):
    plt.tight_layout()
    plt.savefig(out / (name + '.png'), dpi=160, bbox_inches='tight')
    plt.close()


def cluster(root, features, overrides=None):
    out = root / 'outputs'
    overrides = overrides or {}
    xrfm = features[['Recency', 'Frequency', 'Monetary']].copy()
    xrfm['LogMonetary'] = np.log1p(xrfm.pop('Monetary'))
    xrfm.corr().to_csv(out / 'rfm_pearson_correlations.csv')
    xrfm.corr(method='spearman').to_csv(out / 'rfm_spearman_correlations.csv')
    matrices = {'RiskReturn': features[['MeanWeeklyReturn', 'WeeklyBeta']], 'RFM': xrfm}
    labels, diagnostics, choices, projections = {}, [], {}, {}
    for name, x in matrices.items():
        scaler = StandardScaler()
        z = scaler.fit_transform(x)
        if np.unique(z, axis=0).shape[0] < 9:
            raise ValueError('Too few distinct observations for k=2 to 8')
        pd.DataFrame(z, index=x.index, columns=x.columns).to_csv(out / (name + '_standardised.csv'))
        pd.DataFrame({'Mean': scaler.mean_, 'Scale': scaler.scale_}, index=x.columns).to_csv(out / (name + '_scaler.csv'))
        diagnostic = pd.DataFrame({'Skew': x.skew(), 'Minimum': x.min(), 'Maximum': x.max(),
            'AbsZAbove3': (np.abs(z) > 3).sum(axis=0)}, index=x.columns)
        diagnostic.to_csv(out / (name + '_skew_outliers.csv'))
        x.hist(bins=30, figsize=(10, 4))
        plot_save(out, name + '_feature_distributions')
        tree = linkage(z, method='ward', metric='euclidean')
        plt.figure(figsize=(16, 6))
        dendrogram(tree, labels=x.index.tolist(), leaf_font_size=3, leaf_rotation=90)
        plt.title(name + ' Ward dendrogram')
        plt.xlabel('Share ticker'); plt.ylabel('Ward linkage distance')
        plot_save(out, name + '_dendrogram')
        if name == 'RFM':
            pca = PCA(n_components=2)
            projection = pca.fit_transform(z)
            projections[name] = {'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'loadings': pca.components_.tolist(), 'features': x.columns.tolist()}
            axislabels = [f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', f'PC2 ({pca.explained_variance_ratio_[1]:.1%})']
        else:
            projection = x.to_numpy(); axislabels = ['Arithmetic mean weekly return (decimal)', 'Weekly beta']
        for algorithm in ['Ward', 'KMeans']:
            candidates, candidates_labels = [], {}
            for k in range(2, 9):
                if algorithm == 'Ward':
                    lab = cut_tree(tree, n_clusters=k).ravel()
                    inertia = float(sum(((z[lab == j] - z[lab == j].mean(axis=0)) ** 2).sum() for j in np.unique(lab)))
                else:
                    model = KMeans(n_clusters=k, random_state=SEED, n_init=30)
                    lab = model.fit_predict(z); inertia = float(model.inertia_)
                sizes = np.bincount(lab)
                record = {'Matrix': name, 'Algorithm': algorithm, 'k': k,
                          'Silhouette': silhouette_score(z, lab), 'Inertia': inertia,
                          'MinSize': int(sizes.min()), 'Sizes': json.dumps(sizes.tolist())}
                candidates.append(record); diagnostics.append(record); candidates_labels[k] = lab
                features.assign(Cluster=lab + 1).groupby('Cluster')[FEATURES].mean().to_csv(out / f'{name}_{algorithm}_k{k}_candidate_profiles.csv')
            # Explicit provisional policy. Review candidate profiles before final interpretation.
            acceptable = [r for r in candidates if r['MinSize'] >= max(2, int(np.ceil(.02 * len(x))))]
            best = max(acceptable or candidates, key=lambda r: (r['Silhouette'], -r['k']))
            key = name + '_' + algorithm
            k = int(overrides.get(key, best['k']))
            if k not in candidates_labels:
                raise ValueError('Selected k must be between 2 and 8')
            lab = candidates_labels[k] + 1
            labels[key] = lab
            choices[key] = {'k': k, 'automatic_recommendation': best['k'],
                            'policy': 'Maximum silhouette among k with each cluster at least 2 percent of common shares (minimum 2); fallback all k. Review sizes, dendrograms and profiles.',
                            'human_override': key in overrides}
            profile = features.assign(Cluster=lab).groupby('Cluster')[FEATURES].agg(['count', 'mean', 'median'])
            profile.to_csv(out / (key + '_profiles.csv'))
            sectors = pd.crosstab(pd.Series(lab, index=features.index, name='Cluster'), features['Sector'])
            sectors.to_csv(out / (key + '_sector_counts.csv'))
            sectors.div(sectors.sum(axis=1), axis=0).to_csv(out / (key + '_sector_proportions.csv'))
            sectors.div(sectors.sum(axis=1), axis=0).plot.bar(stacked=True, figsize=(11, 5), colormap='tab20')
            plt.ylabel('Within-cluster sector proportion'); plt.title(key + ' sector composition')
            plt.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
            plot_save(out, key + '_sectors')
            plt.figure(figsize=(10, 6))
            for j in np.unique(lab):
                mask = lab == j
                plt.scatter(projection[mask, 0], projection[mask, 1], label=f'Cluster {j} (n={mask.sum()})', s=18, alpha=.7)
            # Tickers for extreme points; complete labelled coordinates remain in the CSV.
            extreme = np.argsort(np.linalg.norm(z, axis=1))[-12:]
            for i in extreme:
                plt.annotate(x.index[i], projection[i], fontsize=7)
            plt.xlabel(axislabels[0]); plt.ylabel(axislabels[1]); plt.title(key); plt.legend()
            plot_save(out, key + '_scatter')
            pd.DataFrame({'Share': features.index, 'X': projection[:, 0], 'Y': projection[:, 1], 'Cluster': lab}).to_csv(out / (key + '_plot_coordinates.csv'), index=False)
    assignments = features.copy()
    for name, lab in labels.items():
        assignments[name] = lab
    assignments.to_csv(out / 'all_assignments.csv')
    comparisons = pd.DataFrame({a: {b: adjusted_rand_score(labels[a], labels[b]) for b in labels} for a in labels})
    comparisons.to_csv(out / 'baseline_adjusted_rand.csv')
    for matrix in matrices:
        pd.crosstab(labels[matrix + '_Ward'], labels[matrix + '_KMeans']).to_csv(out / (matrix + '_algorithm_membership_crosstab.csv'))
    reduced = StandardScaler().fit_transform(features[['Recency', 'Frequency']])
    k = choices['RFM_Ward']['k']
    reduced_lab = cut_tree(linkage(reduced, method='ward'), n_clusters=k).ravel() + 1
    sensitivity = features.assign(RF_Ward=reduced_lab, RFM_Ward=labels['RFM_Ward'])
    sensitivity.to_csv(out / 'monetary_removal_assignments.csv')
    pd.crosstab(sensitivity['RFM_Ward'], sensitivity['RF_Ward']).to_csv(out / 'monetary_removal_crosstab.csv')
    for group in ['RF_Ward', 'RFM_Ward']:
        sensitivity.groupby(group)[FEATURES].agg(['mean', 'median', 'min', 'max']).to_csv(out / (group + '_sensitivity_profiles.csv'))
    result = {'CommonShares': len(features), 'ObservationDate': str(features['ObservationDate'].iloc[0]),
              'MonetaryRemovalARI': adjusted_rand_score(labels['RFM_Ward'], reduced_lab),
              'SelectedModels': choices, 'PCA': projections, 'CalculationVersion': VERSION}
    write_json(out / 'results_summary.json', result)
    pd.DataFrame(diagnostics).to_csv(out / 'model_comparison.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, group) in zip(axes.ravel(), pd.DataFrame(diagnostics).groupby(['Matrix', 'Algorithm'])):
        ax.plot(group['k'], group['Silhouette'], marker='o'); ax.set_title(' '.join(name)); ax.set_xlabel('k'); ax.set_ylabel('Silhouette')
    plot_save(out, 'silhouette_comparison')
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in pd.DataFrame(diagnostics).query("Algorithm == 'KMeans'").groupby('Matrix'):
        ax.plot(group['k'], group['Inertia'], marker='o', label=name)
    ax.set_xlabel('k'); ax.set_ylabel('Within-cluster sum of squares'); ax.legend()
    plot_save(out, 'kmeans_inertia')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--selection', type=Path, help='JSON mapping e.g. RFM_Ward to reviewed k')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    universe = acquire(args.root, args.offline)
    features = prepare(args.root, universe)
    overrides = json.loads(args.selection.read_text()) if args.selection else None
    result = cluster(args.root, features, overrides)
    versions = {}
    for package in ['pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'yfinance']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    write_json(args.root / 'outputs' / 'package_versions.json', versions)
    write_json(args.root / 'outputs' / 'file_checksums.json', {
        str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.root / 'data').glob('*') if p.is_file()})
    print(json.dumps(result, indent=2, default=str))



if __name__ == "__main__":
    main()
