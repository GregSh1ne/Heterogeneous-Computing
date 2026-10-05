import json
import plotly.graph_objects as go

def load_data(json_path="benchmarks.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cuda_results = {}
    eigen_results = {}

    for b in data.get("benchmarks", []):
        name = b["name"]
        # Парсим имя и размер N
        parts = name.split("/")
        bench_base = parts[0]
        n = int(parts[1])

        # Время в миллисекундах
        time_ms = b["real_time"]
        if b.get("time_unit") == "ns":
            time_ms /= 1e6
        elif b.get("time_unit") == "us":
            time_ms /= 1e3
        elif b.get("time_unit") == "s":
            time_ms *= 1e3

        if "BM_CUDA_MatMul_Naive" in bench_base:
            cuda_results[n] = time_ms
        elif "BM_Eigen_MatMul" in bench_base:
            eigen_results[n] = time_ms

    sizes = sorted(list(set(cuda_results.keys()) & set(eigen_results.keys())))
    return sizes, cuda_results, eigen_results

def plot_real_complexity(sizes, cuda_results, eigen_results):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sizes,
        y=[eigen_results[n] for n in sizes],
        mode="lines+markers",
        name="Eigen Matrix Multiplication (CPU, float)",
        line=dict(color="#4363d8", width=2),
        marker=dict(symbol="circle", size=8)
    ))

    fig.add_trace(go.Scatter(
        x=sizes,
        y=[cuda_results[n] for n in sizes],
        mode="lines+markers",
        name="CUDA Matrix Multiplication (GPU, Naive, float)",
        line=dict(color="#e6194b", width=2),
        marker=dict(symbol="circle", size=8)
    ))

    fig.update_layout(
        title="Real Complexity",
        xaxis=dict(
            title="N",
            type="log",
            showgrid=True,
            gridcolor="#e0e0e0"
        ),
        yaxis=dict(
            title="Time, ms",
            type="log",
            showgrid=True,
            gridcolor="#e0e0e0"
        ),
        legend=dict(x=0.02, y=0.1, bgcolor="rgba(255,255,255,0.8)"),
        plot_bgcolor="white",
        width=900,
        height=600
    )

    fig.write_html("real_complexity.html")
    try:
        fig.write_image("real_complexity.png", scale=2)
    except Exception:
        print("Note: install kaleido for static png export: pip install kaleido")

def plot_speedup(sizes, cuda_results, eigen_results):
    speedup = [eigen_results[n] / cuda_results[n] for n in sizes]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sizes,
        y=speedup,
        mode="lines+markers",
        name="Speedup: CUDA vs Eigen",
        line=dict(color="#3cb44b", width=2),
        marker=dict(symbol="circle", size=8)
    ))

    # Базовая линия паритета 1.0x
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Baseline (1.0x)")

    fig.update_layout(
        title="Speedup: CUDA Matrix Multiplication (GPU, Naive, float) vs Eigen Matrix Multiplication (CPU, float)",
        xaxis=dict(
            title="N",
            type="log",
            showgrid=True,
            gridcolor="#e0e0e0"
        ),
        yaxis=dict(
            title="Speedup",
            type="log",
            showgrid=True,
            gridcolor="#e0e0e0"
        ),
        plot_bgcolor="white",
        width=900,
        height=600
    )

    fig.write_html("speedup.html")
    try:
        fig.write_image("speedup.png", scale=2)
    except Exception:
        pass

if __name__ == "__main__":
    sizes, cuda_res, eigen_res = load_data()
    plot_real_complexity(sizes, cuda_res, eigen_res)
    plot_speedup(sizes, cuda_res, eigen_res)
    print("Plots generated successfully: real_complexity and speedup (.html/.png)")