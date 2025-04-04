import asyncio
import aiohttp

# List endpoint Prometheus tiap cluster
PROMETHEUS_ENDPOINTS = [
    "http://3.223.250.176:30007/api/v1/query",
    "http://3.233.22.201:30007/api/v1/query",
    "http://98.84.220.67:30007/api/v1/query",
]


QUERY = 'sum(rate(node_cpu_seconds_total{mode!="idle"}[2m])) by (instance)[30m:2m]'

async def fetch_cpu_usage(session, index, url):
    async with session.get(f"{url}?query={QUERY}") as response:
        data = await response.json()
        values = [float(r[1]) for r in data["data"]["result"][0]["values"]]
        return index, values  # Nama cluster langsung pakai index

async def get_all_cpu_usage():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_cpu_usage(session, idx, url) for idx, url in enumerate(PROMETHEUS_ENDPOINTS)]
        results = await asyncio.gather(*tasks)

    return dict(results)  

if __name__ == "__main__":
    print(asyncio.run(get_all_cpu_usage()))
