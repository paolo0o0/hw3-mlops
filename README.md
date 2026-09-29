# Iris classifier в Triton

Проект запускает `RandomForestClassifier` через Triton Python backend на CPU. Вход модели — массив `FP32` формы `[N, 4]`, выход — номера классов Iris в формате `INT64`.

## Структура проекта

```text
project/
├── Dockerfile
├── client.py
├── requirements-client.txt
├── model_repository/
│   └── iris_classifier/
│       ├── config.pbtxt
│       └── 1/
│           ├── iris_classifier.joblib
│           └── model.py
├── profiling/
│   ├── Dockerfile.model-analyzer
│   ├── model-analyzer.yaml
│   ├── run_model.sh
│   ├── run_perf.sh
│   └── summarize_perf.py
└── results/
    ├── client.log
    ├── metrics-summary.json
    ├── perf-summary.csv
    └── model-analyzer/
        ├── metrics-model-inference.csv
        └── result_summary.html
```

Основной `Dockerfile` собирает сервер с моделью. `model-analyzer` создаёт вспомогательный образ с Model Analyzer и зависимостями модели. Этот образ нужен только для профилирования

`run_perf.sh` запускает HTTP-нагрузку. `summarize_perf.py` объединяет CSV Performance Analyzer в одну сводку. `run_model.sh` запускает Model Analyzer по конфигу `model-analyzer.yaml`

## Запуск сервера

```bash
docker build -t triton-iris .
docker run --rm --name triton-iris-server \
  -p 8000:8000 -p 8002:8002 \
  -v "$PWD/model_repository:/models:ro" \
  triton-iris
```

Сервер готов после появления статуса `READY`

## Проверка HTTP

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-client.txt
python3 client.py
```

Клиент проверяет классы `0`, `1`, `2` для пакетов из 1, 4 и 8 объектов.

## Кастомные метрики

```bash
curl -s http://localhost:8002/metrics \
  | grep -E 'iris_processing_seconds_total|iris_requests|nv_inference_pending_request_count'
```

Во время замера с одним экземпляром счётчик времени обработки вырос с **3,73 до 101,06 секунды** и не уменьшался. Gauge активных запросов менялся от **0 до 1**. При concurrency 32 было зафиксировано **31 ожидающее обращение**. В итоговой конфигурации работают два экземпляра, поэтому суммарное число одновременно обрабатываемых запросов может достигать двух.

## Performance Analyzer

```bash
bash profiling/run_perf.sh
```

Проверены пакеты из 1, 4 и 8 объектов при concurrency 1, 8, 16 и 32. Triton batching отключён (`max_batch_size: 0`), поэтому размер пакета задаётся формой `INPUT:N,4`, без параметра `-b`.

| Объектов в запросе | Concurrency | Запросов/с | Объектов/с | p95, мс | Очередь, мс |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 159,8 | 159,8 | 7,3 | 0,04 |
| 1 | 8 | 177,6 | 177,6 | 50,8 | 38,7 |
| 1 | 16 | 173,8 | 173,8 | 110,0 | 85,4 |
| 1 | 32 | 172,8 | 172,8 | 205,1 | 177,2 |
| 4 | 1 | 144,5 | 578,1 | 8,8 | 0,04 |
| 8 | 1 | 134,5 | 1076,3 | 9,9 | 0,04 |

После concurrency 8 скорость почти не растёт, зато быстро увеличивается ожидание в очереди. Пакет из 8 объектов даёт максимальную пропускную способность: **1076 объектов/с** уже при concurrency 1.

Полная таблица из 12 измерений находится в `results/perf-summary.csv`.

## Model Analyzer

```bash
bash profiling/run_model.sh
```

Конфиг сравнивает один и два CPU-экземпляра модели при concurrency 1, 8, 16 и 32. Используется HTTP и вход `INPUT:1,4`.

| Экземпляров | Concurrency | Запросов/с | p99, мс |
|---:|---:|---:|---:|
| 1 | 1 | 184,0 | 6,2 |
| 2 | 1 | 159,5 | 8,0 |
| 1 | 8 | 189,0 | 53,6 |
| 2 | 8 | 221,6 | 55,9 |
| 1 | 32 | 188,7 | 193,4 |
| 2 | 32 | 334,3 | 149,3 |

На одном запросе второй экземпляр не помогает. При concurrency 8 пропускная способность выросла примерно на 17%. Самая заметная разница получилась при concurrency 32: **334,3 против 188,7 запроса/с**, то есть примерно на 77% больше. Одновременно p99 снизилась с **193,4 до 149,3 мс**.

Model Analyzer выбрал конфигурацию с двумя CPU-экземплярами как лучшую по throughput. Она записана в конфиг `config.pbtxt`

## Общий вывод

HTTP-сервер корректно обработал все контрольные запросы. С одним экземпляром увеличивать concurrency выше 8 нет смысла: скорость почти не меняется, а очередь и задержка растут. Передача нескольких объектов одним массивом заметно повышает общую пропускную способность. Для высокой нагрузки Model Analyzer нашёл более производительный вариант с двумя CPU-экземплярами, поэтому именно он используется в итоговой конфигурации.
