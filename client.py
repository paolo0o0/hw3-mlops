"""Проверка инференса Iris через HTTP API Triton."""
import argparse
import numpy as np
import tritonclient.http as httpclient


def check(host, repeats, batch_sizes):
    examples = np.array([[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5],
                         [6.5, 3.0, 5.8, 2.2]], dtype=np.float32)
    completed = 0
    with httpclient.InferenceServerClient(url=f'{host}:8000') as client:
        if not (client.is_server_live() and client.is_server_ready()
                and client.is_model_ready('iris_classifier')):
            raise RuntimeError('Triton server or model is not ready')
        for batch in batch_sizes:
            for repeat in range(repeats):
                indices = (np.arange(batch) + repeat) % 3
                x = examples[indices]
                inp = httpclient.InferInput('INPUT', x.shape, 'FP32')
                inp.set_data_from_numpy(x)
                result = client.infer('iris_classifier', inputs=[inp]).as_numpy('OUTPUT')
                if result.dtype != np.int64 or result.shape != (batch,):
                    raise AssertionError(f'Unexpected output: {result.dtype}, {result.shape}')
                np.testing.assert_array_equal(result, indices)
                completed += 1
    print(f'HTTP: {completed}/{completed} correct requests; '
          f'batch sizes={batch_sizes}; classes 0, 1, 2 verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8])
    args = parser.parse_args()
    if args.repeats < 1 or any(b < 1 for b in args.batch_sizes):
        parser.error('repeats and batch sizes must be positive')
    check(args.host, args.repeats, args.batch_sizes)


if __name__ == '__main__':
    main()
