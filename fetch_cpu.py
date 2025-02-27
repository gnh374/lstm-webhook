import asyncio
import aiohttp

# List endpoint Prometheus tiap cluster
PROMETHEUS_ENDPOINTS = [
    "http://35.171.190.204:30901/query",
    "http://52.55.44.27:30882/query",
    "http://3.208.78.108:32028/query",
 

]

# Query untuk CPU usage 15 menit terakhir
QUERY = "rate(node_cpu_seconds_total[15m])"

async def fetch_cpu_usage(session, cluster_name, url):
    async with session.get(f"{url}?query={QUERY}") as response:
        data = await response.json()
        values = [float(r["value"][1]) for r in data["data"]["result"]]  # Ambil nilai CPU usage
        return cluster_name, values[-3:]  # Ambil 3 data terakhir (window size = 3)

async def get_all_cpu_usage():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_cpu_usage(session, name, url) for name, url in PROMETHEUS_ENDPOINTS.items()]
        results = await asyncio.gather(*tasks)
    return dict(results)  # Return data dalam format {cluster_name: [usage1, usage2, usage3]}

if __name__ == "__main__":
    asyncio.run(get_all_cpu_usage())
