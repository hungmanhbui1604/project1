import argparse
import time
import numpy as np

from trt_runner import TensorRTRunner


def benchmark(engine_path, batch_size=1, warmup=50, runs=200):
    engine = TensorRTRunner(engine_path)

    x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        engine.infer(x)

    times = []

    for _ in range(runs):
        start = time.perf_counter()
        engine.infer(x)
        end = time.perf_counter()

        times.append((end - start) * 1000.0)

    times = np.array(times)

    latency_per_batch = times.mean()
    latency_per_image = latency_per_batch / batch_size
    throughput = batch_size * 1000.0 / latency_per_batch

    print("=" * 50)
    print(f"Engine              : {engine_path}")
    print(f"Batch size          : {batch_size}")
    print(f"Latency / batch     : {latency_per_batch:.3f} ms")
    print(f"Latency / image     : {latency_per_image:.3f} ms")
    print(f"Throughput          : {throughput:.2f} img/s")
    print(f"Min latency / batch : {times.min():.3f} ms")
    print(f"Max latency / batch : {times.max():.3f} ms")
    print(f"Std latency         : {times.std():.3f} ms")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="dmv_fp16.engine")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    benchmark(
        engine_path=args.engine,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
    )
