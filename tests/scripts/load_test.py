import time
import json
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:8000/predict"
REQUEST_FILE = "samples/request.json"

TRAFFIC_LEVELS = [10, 50, 200]
CONCURRENCY = 10
TIMEOUT_S = 10


def send_request(payload):
    start = time.perf_counter()
    try:
        r = requests.post(URL, json=payload, timeout=TIMEOUT_S)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return latency_ms, r.status_code
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return latency_ms, 0  # 0 = exception


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = int((p / 100.0) * (len(sorted_vals) - 1))
    return float(sorted_vals[k])


def run_level(n_requests):
    print(f"\nRunning load test with {n_requests} requests...")

    with open(REQUEST_FILE, "r") as f:
        payload = json.load(f)

    latencies = []
    status_counts = {}

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(n_requests)]
        for fut in as_completed(futures):
            latency_ms, status = fut.result()
            latencies.append(latency_ms)
            status_counts[status] = status_counts.get(status, 0) + 1

    total_s = time.perf_counter() - t_start
    throughput = n_requests / total_s if total_s > 0 else 0.0

    lat_sorted = sorted(latencies)
    avg = float(sum(lat_sorted) / len(lat_sorted)) if lat_sorted else 0.0
    p50 = float(statistics.median(lat_sorted)) if lat_sorted else 0.0
    p95 = percentile(lat_sorted, 95)

    ok = status_counts.get(200, 0)
    failed = n_requests - ok

    print(f"Success: {ok}/{n_requests} | Failed: {failed}")
    if failed:
        # show top non-200 codes
        top = sorted([(k, v) for k, v in status_counts.items() if k != 200], key=lambda x: -x[1])[:5]
        print("Top non-200 statuses:", top)

    print(f"Throughput: {throughput:.2f} req/sec")
    print(f"Average Latency: {avg:.2f} ms")
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P95 Latency: {p95:.2f} ms")

    return {
        "requests": n_requests,
        "concurrency": CONCURRENCY,
        "timeout_s": TIMEOUT_S,
        "success": ok,
        "failed": failed,
        "status_counts": status_counts,
        "throughput_rps": throughput,
        "avg_ms": avg,
        "p50_ms": p50,
        "p95_ms": p95,
    }


if __name__ == "__main__":
    results = []
    for level in TRAFFIC_LEVELS:
        results.append(run_level(level))

    with open("load_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved load_test_results.json")
