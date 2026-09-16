# План доработки Resource Metrics v2

## Статус реализации

Базовый runtime-контракт реализован:

- schema и measurement policy `resource_metrics_v2`;
- PID в process-owned measurements;
- дедуплицированный подсчёт model, buffer, optimizer и dataset tensor storage;
- Linux phase RSS sampler и отдельный process-lifetime peak;
- раздельные training/evaluation summaries;
- `resource_by_participant.csv` и `resource_transport.csv`;
- message-type transport breakdown по каждому логическому каналу;
- owned footprint для local, centralized, client, SplitServer и FedServer;
- отдельные spawn-процессы LocalTrain и LocalEval;
- строгий отказ от смешивания schema v1 и v2;
- unit, lifecycle и multiprocessing smoke tests;
- сокращённый CUDA validation на трёх реальных корпусах.

Остаются последующие экспериментальные и presentation-задачи:

- отдельный concurrent host-level sampler;
- полный повтор E1--E4;
- автоматическая генерация paper tables из schema-v2 artifacts.

## 1. Цель

Resource Metrics v2 должны обеспечивать воспроизводимое и методологически
корректное сравнение стоимости обучения в режимах `local`, `centralized`,
`federated`, `split` и `splitfed`.

Текущий единый показатель `Peak RSS` недостаточен: он отражает high-water mark
всего процесса, включает Python/PyTorch runtime, allocator cache, временные
тензоры и evaluation и не показывает, какими ресурсами действительно владеет
каждый участник. В частности, cross-corpus evaluation не должен увеличивать
стоимость local training.

Новая система должна раздельно отвечать на три вопроса:

1. Каким статическим объёмом данных и model state владеет участник?
2. Какой фактический пик CPU/GPU memory возникает во время обучения?
3. Каковы суммарные временные и коммуникационные затраты эксперимента?

## 2. Основные понятия

### 2.1. Owned footprint

Детерминированно вычисляемый объём ресурсов, принадлежащих участнику:

```text
model_bytes = parameter_bytes + model_buffer_bytes

owned_static_bytes =
    model_bytes
  + optimizer_state_bytes
  + owned_dataset_bytes
```

`owned_dataset_bytes` включает только tensor storage данных, labels,
нормализационной статистики и masks, которыми владеет конкретный участник.
Shared storage нельзя учитывать повторно.

### 2.2. Runtime peak

Фактически измеренная память процесса и CUDA:

- process-lifetime RSS high-water mark;
- sampled RSS peak конкретной фазы;
- peak CUDA allocated;
- peak CUDA reserved.

Runtime RSS нельзя описывать как чистый размер модели или датасета. Он также
включает interpreter, библиотеки, framework runtime, allocator cache и
временные batch tensors.

### 2.3. Experiment cost

- training wall time;
- CPU user/system time;
- evaluation wall/CPU time отдельно;
- число обработанных samples и batches;
- transport messages и bytes по логическим каналам.

## 3. Measurement policy

Стабильное имя политики:

```yaml
measurement_policy: resource_metrics_v2
```

Фазы измеряются независимо:

```text
dataset_load
model_setup
train
aggregation
evaluation
checkpoint
lifetime
```

В основное сравнение стоимости обучения входят только `dataset_load`,
`model_setup`, `train`, необходимая для выбранного режима `aggregation` и
training transport. Cross-corpus evaluation и checkpoint serialization
публикуются отдельно.

Все значения сохраняются в байтах. В пользовательских и paper-таблицах они
переводятся в MiB/GiB через `1024^2` и `1024^3`.

## 4. Схема интервальной записи

Необходимо повысить `RESOURCE_METRICS_SCHEMA_VERSION` до `2`.

Минимальная запись:

```yaml
schema_version: 2
measurement_policy: resource_metrics_v2
role: client
client_id: 0
process_id: 12345
round: 1
phase: train

wall_time_seconds: 1.0
cpu_user_seconds: 0.8
cpu_system_seconds: 0.1

process_peak_rss_bytes: 0
phase_start_rss_bytes: 0
phase_end_rss_bytes: 0
phase_sampled_peak_rss_bytes: 0
peak_cuda_allocated_bytes: 0
peak_cuda_reserved_bytes: 0

model_parameter_bytes: 0
model_buffer_bytes: 0
optimizer_state_bytes: 0
owned_dataset_bytes: 0
owned_static_bytes: 0

samples: 0
batches: 0
messages_sent: 0
bytes_sent: 0
```

`process_id` обязателен для process-owned ролей и позволяет подтвердить
изоляцию участников.

## 5. Измерение RSS

`resource.getrusage(...).ru_maxrss` сохранить как
`process_peak_rss_bytes`. Это накопительный максимум за всю жизнь процесса, и
он не сбрасывается между фазами.

Для phase-local peak реализовать лёгкий RSS sampler:

- Linux: `/proc/self/status` или `/proc/self/statm`;
- частота по умолчанию 10--20 Hz;
- bounded lifecycle и обязательное завершение sampler thread;
- macOS fallback с явно записанным методом измерения;
- отсутствие sampler не должно останавливать обучение, но должно быть отражено
  значением `null` и причиной в metadata.

## 6. Ownership по режимам

### 6.1. Local

Каждый corpus-owned participant запускается в отдельном свежем
`spawn`-процессе и во время обучения содержит только:

```text
complete model + optimizer + one local corpus
```

Последовательность:

1. Загрузить только собственный корпус.
2. Создать complete model и optimizer.
3. Зафиксировать owned footprint.
4. Выполнить local training и сохранить training metrics.
5. Сохранить checkpoint и завершить training process.
6. Выполнить cross-corpus evaluation в отдельном evaluation process.

Чужие корпуса никогда не должны загружаться в local training process.

### 6.2. Centralized

Один participant владеет:

```text
complete model + optimizer + all configured corpora
```

Training и evaluation phases публикуются раздельно.

### 6.3. Federated

Каждый клиент владеет:

```text
complete model + client optimizer + local corpus
```

FedServer владеет global model state и aggregation buffers. В итогах нужны:

- maximum client footprint;
- mean client footprint;
- sum of client footprints;
- FedServer footprint;
- logical system total.

### 6.4. Split и SplitFed

Клиент владеет:

```text
client model partition
+ client optimizer
+ local corpus
+ current activation/gradient buffers
```

SplitServer владеет:

```text
server model partition
+ server optimizer
+ current activation/gradient buffers
```

FedServer учитывается отдельной ролью, если выбранный режим его создаёт.
Client, SplitServer и FedServer footprint нельзя скрывать одним общим peak.

## 7. Правила агрегации

Для каждой роли сохранять:

- `max_participant_*`;
- `mean_participant_*`;
- `sum_participant_owned_bytes`;
- отдельные значения каждого participant.

Нельзя складывать RSS peaks разных процессов и называть результат host peak.
Такое сложение допустимо только для детерминированного owned footprint.

Фактический concurrent host peak требует отдельного host-level sampler и
должен иметь другое имя.

## 8. Transport accounting

Разделить коммуникации по логическим направлениям:

```yaml
transport:
  split_uplink_bytes: 0
  split_downlink_bytes: 0
  federated_uplink_bytes: 0
  federated_downlink_bytes: 0
  payload_bytes: 0
  serialized_bytes: 0
  messages: 0
```

Для Split/SplitFed отдельно учитывать:

- smashed activations;
- activation gradients;
- labels;
- model states;
- protocol overhead.

Queue transport остаётся логической оценкой, gRPC transport отражает размер
protobuf serialization. Это различие обязательно записывается в metadata.

## 9. Выходные артефакты

### `resource_metrics.csv`

Интервальные phase/round measurements.

### `resource_by_participant.csv`

Одна строка на роль и participant с owned footprint и training/evaluation
peaks.

### `resource_transport.csv`

Breakdown коммуникаций по channel, direction, message type и participant.

### `resource_summary.yaml`

```yaml
schema_version: 2
measurement_policy: resource_metrics_v2

training:
  wall_time_seconds: 0
  cpu_user_seconds: 0
  cpu_system_seconds: 0
  max_process_peak_rss_bytes: 0
  max_phase_peak_rss_bytes: 0
  max_cuda_allocated_bytes: 0
  total_transport_bytes: 0

owned_footprint:
  max_participant_bytes: 0
  sum_participant_bytes: 0
  by_role: {}

evaluation:
  wall_time_seconds: 0
  max_phase_peak_rss_bytes: 0

participants: []
```

Schema-v1 артефакты должны либо читаться как явно помеченные legacy data,
либо отклоняться с понятной ошибкой. Их нельзя молча смешивать со schema v2.

## 10. Paper-таблицы

Основная сравнительная таблица:

| Method | Train time | Max participant owned | Max participant RSS | Server RSS | Transport |
|---|---:|---:|---:|---:|---:|

Дополнительная topology-aware таблица:

| Method | Max client | Sum clients | SplitServer | FedServer | Logical total |
|---|---:|---:|---:|---:|---:|

Подпись должна пояснять:

- `Max participant` — требование к одному узлу;
- `Sum clients` и `Logical total` — сумма owned footprint, а не RSS;
- training и evaluation не смешиваются;
- process RSS и CUDA memory представлены раздельно;
- используются бинарные MiB/GiB.

Paper-таблицы должны генерироваться из schema-v2 artifacts скриптом, а не
ручным переносом чисел.

## 11. Тестирование

Обязательные тесты:

1. Точный подсчёт parameter и buffer bytes.
2. Подсчёт optimizer state до и после первого optimizer step.
3. Подсчёт dataset tensor storage без двойного учёта shared storage.
4. Local training worker получает только собственный корпус.
5. Local participants имеют разные PID.
6. Evaluation запускается отдельно и не меняет training summary.
7. Training/evaluation phase aggregation не смешивает peaks и время.
8. Max, mean, sum и role breakdown вычисляются детерминированно.
9. Transport breakdown сходится с total bytes/messages.
10. CPU `spawn` smoke test завершается без оставшихся процессов.
11. Failure injection завершает worker и публикует исходную причину.
12. Schema-v1 compatibility policy проверяется тестом.

Для CUDA validation использовать сокращённый dataset и несколько rounds. Не
запускать полный E1--E4 только ради проверки реализации.

## 12. Этапы реализации

### Этап A. Контракт и чистые функции

- утвердить schema v2 и имена полей;
- реализовать tensor-storage accounting;
- реализовать aggregation без привязки к runtime;
- покрыть функции unit-тестами.

### Этап B. Phase-aware runtime measurement

- добавить RSS sampler;
- разделить training/evaluation timers и peaks;
- записывать PID и measurement backend;
- проверить bounded shutdown sampler.

### Этап C. Local isolation

- один fresh spawn process на local training participant;
- только локальный corpus в training process;
- отдельный evaluation process;
- multiprocessing smoke и failure test.

### Этап D. Остальные topology

- centralized ownership;
- FedAvg client/FedServer ownership;
- Split/SplitFed client/SplitServer/FedServer ownership;
- topology-aware aggregation.

### Этап E. Transport и persistence

- channel/message breakdown;
- новые CSV/YAML artifacts;
- legacy schema policy;
- atomic persistence.

### Этап F. Эксперименты и paper

- CPU smoke profiles;
- сокращённый CUDA validation;
- полный повтор E1--E4 с одинаковыми seed/config;
- автоматическая генерация paper tables;
- обновление README и Repository Guide.

## 13. Критерии готовности

Resource Metrics v2 считается завершённой, когда:

- local training process загружает только собственный corpus;
- каждый local participant имеет отдельный PID;
- training и evaluation metrics не смешиваются;
- model, optimizer и dataset bytes воспроизводимо вычисляются;
- distributed modes публикуют max participant и logical total;
- RSS нигде не интерпретируется как чистый вес модели;
- byte values отображаются только в MiB/GiB с делителями `1024`;
- процессы, sampler threads и queues гарантированно завершаются;
- schema-v2 unit и multiprocessing smoke tests проходят;
- E1--E4 перезапущены с одной measurement policy;
- paper tables полностью воспроизводятся из сохранённых artifacts.

## 14. Ограничения текущих артефактов

Артефакты `artifacts/rerun_20260915` созданы по schema v1. Их RSS, CUDA,
wall/CPU и transport values пригодны как legacy measurements, но они не
содержат owned footprint и phase-local RSS peak. После реализации schema v2
их нельзя использовать в одной итоговой таблице с новыми запусками без явной
legacy маркировки.
