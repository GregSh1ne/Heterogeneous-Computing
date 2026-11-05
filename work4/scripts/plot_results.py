import json
import os
import plotly.graph_objects as go


def plot_work4_results(json_path="benchmarks.json", output_dir="charts"):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    shmem_data = {}
    wmma_data = {}

    for b in data["benchmarks"]:
        name = b["name"]
        time_ms = b["manual_time"] if "manual_time" in b else b["real_time"]
        if b.get("time_unit") == "ns":
            time_ms /= 1e6
        elif b.get("time_unit") == "us":
            time_ms /= 1e3
        elif b.get("time_unit") == "s":
            time_ms *= 1e3

        n = int(name.split("/")[1])
        if "BM_CUDA_Matmul_SharedMemory" in name:
            shmem_data[n] = time_ms
        elif "BM_CUDA_Matmul_WMMA" in name:
            wmma_data[n] = time_ms

    sizes = sorted(wmma_data.keys())
    times_shmem = [shmem_data[n] for n in sizes]
    times_wmma = [wmma_data[n] for n in sizes]
    speedup = [shmem_data[n] / wmma_data[n] for n in sizes]

    # График 1: Реальная вычислительная сложность
    fig_complexity = go.Figure()
    fig_complexity.add_trace(go.Scatter(
        x=sizes, y=times_shmem,
        mode="lines+markers",
        name="Shared Memory Matmul (Work #3)",
        line=dict(color="crimson", width=2),
        marker=dict(size=7)
    ))
    fig_complexity.add_trace(go.Scatter(
        x=sizes, y=times_wmma,
        mode="lines+markers",
        name="WMMA Tensor Cores Matmul (Work #4)",
        line=dict(color="seagreen", width=2),
        marker=dict(size=7)
    ))

    fig_complexity.update_layout(
        title="Real Complexity: Shared Memory vs WMMA Tensor Cores",
        xaxis_title="Matrix Size (N x N)",
        yaxis_title="Time (ms)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    fig_complexity.write_html(os.path.join(output_dir, "real_complexity.html"))
    fig_complexity.write_image(os.path.join(output_dir, "real_complexity.png"))

    # График 2: Ускорение (Speedup)
    fig_speedup = go.Figure()
    fig_speedup.add_trace(go.Scatter(
        x=sizes, y=speedup,
        mode="lines+markers",
        name="Speedup (T_shmem / T_wmma)",
        line=dict(color="royalblue", width=2),
        marker=dict(size=7)
    ))
    fig_speedup.add_shape(
        type="line", line=dict(dash="dash", color="gray"),
        x0=min(sizes), y0=1.0, x1=max(sizes), y1=1.0
    )

    fig_speedup.update_layout(
        title="Speedup of WMMA Implementation relative to Shared Memory",
        xaxis_title="Matrix Size (N x N)",
        yaxis_title="Speedup Factor",
        xaxis_type="log",
        yaxis_type="linear",
        template="plotly_white"
    )
    fig_speedup.write_html(os.path.join(output_dir, "speedup.html"))
    fig_speedup.write_image(os.path.join(output_dir, "speedup.png"))


if __name__ == "__main__":
    plot_work4_results()