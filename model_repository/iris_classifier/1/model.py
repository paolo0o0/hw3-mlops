import time
from pathlib import Path

import joblib
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_path = Path(__file__).parent / "iris_classifier.joblib"
        self.model = joblib.load(model_path)

        # Две кастомные метрики
        self.processing_time_family = pb_utils.MetricFamily(
            name="iris_processing_seconds_total",
            description="Total time spent inside iris model execute()",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.processing_time_metric = self.processing_time_family.Metric(
            labels={"model": "iris_classifier", "instance": args["model_instance_name"]}
        )

        self.gauge_family = pb_utils.MetricFamily(
            name="iris_requests",
            description="Number of requests currently processed by iris model",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.gauge_metric = self.gauge_family.Metric(
            labels={"model": "iris_classifier", "instance": args["model_instance_name"]}
        )

        self.gauge = 0
        self.gauge_metric.set(0)

    def execute(self, requests):
        responses = []

        # запросы, которые тритон передал в бэкэнд
        for request in requests:
            self.gauge += 1
            self.gauge_metric.set(self.gauge)
            started = time.perf_counter()

            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
                x = input_tensor.as_numpy().astype(np.float32)

                prediction = self.model.predict(x).astype(np.int64)

                output_tensor = pb_utils.Tensor("OUTPUT", prediction)
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[output_tensor])
                )
            finally:
                elapsed = time.perf_counter() - started
                self.processing_time_metric.increment(elapsed)

                self.gauge -= 1
                self.gauge_metric.set(self.gauge)

        return responses

    def finalize(self):
        print("iris_classifier is unloading")
