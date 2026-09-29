"""Создадим CSV из сырых логов perfomance analyzer"""
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results'
rows = []
for file in sorted((root / 'perf').glob('http_*.csv')):
    protocol, n, c = file.stem.split('_')
    n, c = int(n[1:]), int(c[1:])
    log = file.with_suffix('.log').read_text()
    if 'Failed to obtain stable measurement' in log:
        raise RuntimeError(f'Unstable measurement: {file}')
    with file.open() as f:
        raw = next(csv.DictReader(f))
    qps = float(raw['Inferences/Second'])
    rows.append(dict(protocol=protocol,objects_per_request=n,concurrency=c,
                     requests_per_second=qps,objects_per_second=qps*n,
                     p95_latency_ms=float(raw['p95 latency'])/1000,
                     p99_latency_ms=float(raw['p99 latency'])/1000,
                     server_queue_ms=float(raw['Server Queue'])/1000,
                     server_infer_ms=float(raw['Server Compute Infer'])/1000))
if len(rows) != 12:
    raise RuntimeError(f'Expected 12 stable HTTP results, found {len(rows)}')
with (root / 'perf-summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f'Saved {len(rows)} results')
