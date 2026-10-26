import json
import plotly.graph_objects as go

def plot_work3_results(json_path="benchmarks.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    naive_data = {}
    shmem_data = {}

    for b in data["benchmarks"]:
        name = b["name"]
        time_ms = b["real_time"] if b["time_unit"] == "ms" else b["real_time"] / 1e6
        n = int(name.split("/")[1])
        if "BM_CUDA_Matmul_Naive" in name:
            naive_data[n] = time_ms
        elif "BM_CUDA_Matmul_SharedMemory" in name:
            shmem_data[n] = time_ms

    sizes = sorted(shmem_data.keys())
    times_naive = [naive_data[n] for n in sizes]
    times_shmem = [shmem_data[n] for n in sizes]
    speedup = [naive_data[n] / shmem_data[n] for n in sizes]

    # График 1: Real Complexity (Naive vs Shared Memory)
    fig_complexity = go.Figure()
    fig_complexity.add_trace(go.Scatter(
        x=sizes, y=times_naive,
        mode='lines+markers',
        name='Naive Matmul (Work #2)',
        line=dict(color='crimson', width=2),
        marker=dict(size=7)
    ))
    fig_complexity.add_trace(go.Scatter(
        x=sizes, y=times_shmem,
        mode='lines+markers',
        name='Shared Memory Matmul (Work #3)',
        line=dict(color='seagreen', width=2),
        marker=dict(size=7)
    ))

    fig_complexity.update_layout(
        title="Real Complexity: Naive vs Shared Memory Matmul",
        xaxis_title="Matrix Size (N x N)",
        yaxis_title="Time (ms)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    fig_complexity.write_html("real_complexity.html")
    fig_complexity.write_image("real_complexity.png")

    # График 2: Speedup (Shared Memory relative to Naive)
    fig_speedup = go.Figure()
    fig_speedup.add_trace(go.Scatter(
        x=sizes, y=speedup,
        mode='lines+markers',
        name='Speedup (T_naive / T_shmem)',
        line=dict(color='royalblue', width=2),
        marker=dict(size=7)
    ))
    fig_speedup.add_shape(
        type="line", line=dict(dash='dash', color="gray"),
        x0=min(sizes), y0=1.0, x1=max(sizes), y1=1.0
    )

    fig_speedup.update_layout(
        title="Speedup of Shared Memory Matmul relative to Naive Implementation",
        xaxis_title="Matrix Size (N x N)",
        yaxis_title="Speedup Factor",
        xaxis_type="log",
        yaxis_type="linear",
        template="plotly_white"
    )
    fig_speedup.write_html("speedup.html")
    fig_speedup.write_image("speedup.png")

if __name__ == "__main__":
    plot_work3_results()