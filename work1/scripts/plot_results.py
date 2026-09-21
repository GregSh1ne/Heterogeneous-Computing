import json
import plotly.graph_objects as go

def plot_benchmark_results(json_path="results.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    cuda_bench = {}
    eigen_bench = {}

    for b in data["benchmarks"]:
        name = b["name"]
        time_ms = b["real_time"] if b["time_unit"] == "ms" else b["real_time"] / 1e6
        if "BM_CUDA_VectorAdd" in name:
            n = int(name.split("/")[1])
            cuda_bench[n] = time_ms
        elif "BM_Eigen_VectorAdd" in name:
            n = int(name.split("/")[1])
            eigen_bench[n] = time_ms

    n_sizes = sorted(cuda_bench.keys())
    cuda_times = [cuda_bench[n] for n in n_sizes]
    eigen_times = [eigen_bench[n] for n in n_sizes]
    speedup = [eigen_bench[n] / cuda_bench[n] for n in n_sizes]

    # График 1: Real Complexity
    fig_complexity = go.Figure()
    fig_complexity.add_trace(go.Scatter(
        x=n_sizes, y=eigen_times,
        mode='lines+markers',
        name='Eigen Vector Addition (CPU)',
        line=dict(color='royalblue', width=2),
        marker=dict(size=6)
    ))
    fig_complexity.add_trace(go.Scatter(
        x=n_sizes, y=cuda_times,
        mode='lines+markers',
        name='CUDA Vector Addition (GPU)',
        line=dict(color='crimson', width=2),
        marker=dict(size=6)
    ))

    fig_complexity.update_layout(
        title="Real Complexity",
        xaxis_title="N",
        yaxis_title="Time, ms",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    fig_complexity.write_html("real_complexity.html")
    fig_complexity.write_image("real_complexity.png")

    # График 2: Speedup
    fig_speedup = go.Figure()
    fig_speedup.add_trace(go.Scatter(
        x=n_sizes, y=speedup,
        mode='lines+markers',
        name='Speedup GPU vs CPU',
        line=dict(color='darkslateblue', width=2),
        marker=dict(size=6)
    ))
    fig_speedup.add_shape(
        type="line", line=dict(dash='dash', color="gray"),
        x0=min(n_sizes), y0=1, x1=max(n_sizes), y1=1
    )

    fig_speedup.update_layout(
        title="Speedup: CUDA Vector Addition (GPU) vs Eigen Vector Addition (CPU)",
        xaxis_title="N",
        yaxis_title="Speedup",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    fig_speedup.write_html("speedup.html")
    fig_speedup.write_image("speedup.png")

if __name__ == "__main__":
    plot_benchmark_results()