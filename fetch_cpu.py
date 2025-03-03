import asyncio
import aiohttp

# List endpoint Prometheus tiap cluster
PROMETHEUS_ENDPOINTS = [
    "http://35.171.190.204:30901/api/v1/query",
    "http://52.55.44.27:30882/api/v1/query",
    "http://3.208.78.108:32028/api/v1/query",
]

# Query untuk CPU usage 15 menit terakhir
QUERY = 'sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (instance)[15m:5m]'

async def fetch_cpu_usage(session, index, url):
    async with session.get(f"{url}?query={QUERY}") as response:
        data = await response.json()
        values = [float(r["value"][1]) for r in data["data"]["result"][0]["values"]]  
        print("data : " +data)
        return index, values  # Nama cluster langsung pakai index

async def get_all_cpu_usage():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_cpu_usage(session, idx, url) for idx, url in enumerate(PROMETHEUS_ENDPOINTS)]
        results = await asyncio.gather(*tasks)
    return dict(results)  # Return dalam format {0: [usage1, usage2, usage3], 1: [...], 2: [...]}

if __name__ == "__main__":
    print(asyncio.run(get_all_cpu_usage()))
