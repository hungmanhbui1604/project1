import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTRunner:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Register TensorRT built-in plugins before deserializing engine
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine. "
                "This usually means the engine was built with a plugin or TensorRT version "
                "that is not available on this Jetson."
            )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = [None] * self.engine.num_bindings

        self.input_idx = None
        self.output_indices = []

        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
            else:
                self.output_indices.append(i)

        if self.input_idx is None:
            raise RuntimeError("No input binding found in TensorRT engine")

        if len(self.output_indices) != 2:
            raise RuntimeError(
                f"Expected 2 output bindings, got {len(self.output_indices)}"
            )

        self.input_name = self.engine.get_binding_name(self.input_idx)
        self.output_names = [
            self.engine.get_binding_name(i) for i in self.output_indices
        ]

        print(f"[TensorRT] input  : {self.input_name}")
        print(f"[TensorRT] outputs: {self.output_names}")

    def infer(self, x: np.ndarray):
        """
        x: [B, C, H, W], float32 numpy array

        returns:
            branch_a_out: [B, branch_a_num_classes]
            branch_b_out: [B, 1]
        """
        x = np.ascontiguousarray(x.astype(np.float32))

        self.context.set_binding_shape(self.input_idx, x.shape)

        d_input = cuda.mem_alloc(x.nbytes)
        self.bindings[self.input_idx] = int(d_input)

        cuda.memcpy_htod_async(d_input, x, self.stream)

        host_outputs = []
        device_outputs = []

        for out_idx in self.output_indices:
            out_shape = tuple(self.context.get_binding_shape(out_idx))
            out_size = int(np.prod(out_shape))

            h_out = np.empty(out_size, dtype=np.float32)
            d_out = cuda.mem_alloc(h_out.nbytes)

            self.bindings[out_idx] = int(d_out)

            host_outputs.append((h_out, out_shape))
            device_outputs.append(d_out)

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        outputs = []
        for (h_out, out_shape), d_out in zip(host_outputs, device_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
            outputs.append(h_out.reshape(out_shape))

        self.stream.synchronize()

        output_dict = {
            name: output for name, output in zip(self.output_names, outputs)
        }

        if "branch_a_out" not in output_dict:
            raise RuntimeError(
                f"branch_a_out not found. Available outputs: {self.output_names}"
            )

        if "branch_b_out" not in output_dict:
            raise RuntimeError(
                f"branch_b_out not found. Available outputs: {self.output_names}"
            )

        return output_dict["branch_a_out"], output_dict["branch_b_out"]