# Experiments

Commands below select repository configurations, not proven replicas of
published algorithms. Run from the repository root and inspect hardware/device
settings before launching full training. General setup is in the
[README](../README.md); resource reporting follows
[Resource Metrics v2](runtime/RESOURCE_METRICS.md).

## Execution status

| Experiment | Implementation | Remaining research validation |
| --- | --- | --- |
| E0 | All-record MMD and relative normalized W1 runner | Actor-level sensitivity; subsampling quantiles are not confidence intervals |
| E1 | Isolated local training and cross-corpus matrix | Matched multi-seed/fold summaries |
| E2 | Centralized/FedAvg with complete-model checkpoint evaluation | Matched budget and multi-seed comparisons |
| E3 | Shared/personalized SplitFed and MergeSFL reconstruction | Canonical protocol checks, calibrated telemetry, controlled ablations |
| E4 | Max-steps versus cycling fixed-steps profiles | Unique/repeated sample accounting and actor-held-out validation |

The numbering retains experiment identifiers used by configurations and artifacts.
Quality summaries for split methods exist,
but a common composed-checkpoint cross-corpus evaluator remains future work.
Local training/evaluation use separate spawn processes. Resource Metrics v2
already records phase timing, owned footprint, per-process peaks and transport;
a concurrent host sampler is separate future work.

## Область исследования и терминология

SplitFed Speech Emotion Classification оценивается как pre-alpha фреймворк для
бинарной классификации эмоциональной речи. `ANG` является положительным
классом; все остальные поддерживаемые эмоции отображаются в класс `0`. Проект
не является системой
распознавания речи speech-to-text и пока не предоставляет формальных гарантий
приватности.

Исследование разделяется на два независимых направления:

1. **Воспроизведение базовой методики:** каждый метод обучается и оценивается
   отдельно на каждом корпусе. Настройки исходной работы SplitFed сохраняются
   настолько точно, насколько это допускает задача классификации речи.
2. **Основное cross-corpus исследование:** в одном запуске один клиент владеет
   CREMA-D, второй — RAVDESS, третий — SAVEE. Именно здесь исследуются разные
   объёмы корпусов, domain shift, actor-disjoint оценка и оркестрация.

Результаты двух направлений нельзя объединять в одну выборку. Первое проверяет
корректность реализации baseline-методов, второе соответствует основной
исследовательской постановке.

### Определения методов

| Метод | Операционное определение |
| --- | --- |
| Local | Одна полная модель обучается на одном корпусе без взаимодействия с другими клиентами. |
| Centralized | Одна полная модель обучается на объединении доступных обучающих выборок. |
| FedAvg | Полные клиентские модели обучаются локально и агрегируются на федеративном сервере. |
| SFLv1 | Для каждого клиента существуют отдельные клиентская и серверная части; обе части агрегируются в глобальные клиентскую и серверную модели. |
| SFLv2 | Клиентские части агрегируются; одна общая серверная модель последовательно обрабатывает клиентов и не агрегируется. |
| MergeSFL | Feature merging и регулирование batch size, включая их совместную оптимизацию, реализованы в соответствии со статьёй. |
| OUR | Исследуемая синхронизированная многопроцессная реализация: общий сервер обрабатывает конкатенированный batch из activation с совпадающими `(round, step)`, а клиентские части агрегируются. Неравная нагрузка ограничивается явным бюджетом локальных шагов. |

В статье SplitFed SFLv1 использует параллельные серверные модели клиентов и их
последующую агрегацию. В SFLv2 серверная агрегация удалена: общая серверная
модель последовательно обрабатывает клиентов и обновляется после каждого из
них. Конкатенация activation не является определяющим свойством SFLv2. Поэтому
текущий runtime следует обозначать как `OUR` или `synchronized concatenated
SplitFed`, но не как канонический SFLv2.

Операции `torch.cat` над клиентскими activation также недостаточно, чтобы
называть runtime реализацией MergeSFL. MergeSFL дополнительно формирует mixed
feature sequence, регулирует batch size неоднородных workers и совместно
оптимизирует эти механизмы.

## Data and comparison rules

Training/model evaluation use actor-disjoint splits and train-only normalization.
Test actors must not participate in hyperparameter selection. E0 instead uses
all records and corpus-local analysis statistics. SAVEE's four actors limit
both split coverage and independent inference. Always report corpus/class/actor
coverage, extraction losses, feature ordering, seeds and resolved configuration.

Protocol-faithful comparisons preserve optimizer, workload, cohort and aggregation
semantics as experimental factors. A causal claim about one mechanism requires
a separate matched-budget ablation. Proposed validation-actor experiments need
an explicit validation split; the current train/test split alone does not provide it.

## E0: domain shift

The complete protocol, command, outputs and limitations live in
[E0 domain shift](experiments/E0_DOMAIN_SHIFT.md). It uses 50 repetitions:
MMD raw/local normalized at n=480 and normalized W1 between/within at n=240
with per-repeat ratio R. Table 1/2 describe corpus composition and raw features.
There is no model training, train/test split or Sinkhorn metric in current E0.

## Training experiments

### E1 — local corpus evaluation

**Вопрос:** насколько модель, обученная на одном корпусе, переносится на другие
корпуса без совместного обучения?

**Метод:** три независимые Local-модели. Каждая модель оценивается без
дообучения на actor-disjoint test views всех трёх корпусов.

**Метрики:** Anger F1, PR-AUC, recall, precision, macro и worst-corpus значения.
Accuracy используется как вспомогательная метрика.

Local-only матрица запускается без каналов и серверов:

```bash
uv run secureasr \
  --config-file configs/experiments/config.e1_local.yaml
```

Конфиг явно выбирает `training.mode: local`; основной CLI направляет такой
запуск в local cross-corpus runner без `TrainingController`, каналов и серверов.
Один `training.num_rounds` соответствует одной полной локальной эпохе. Все
три модели получают одинаковую инициализацию по `training.seed`. Для каждой
ячейки сохраняются accuracy, Anger F1, macro-F1, precision, recall, UAR,
PR-AUC, confusion counts и размеры классов. Если eval fold содержит один
класс, `pr_auc` записывается как `null`, а не как вводящее в заблуждение число.
Артефакты находятся в
`artifacts/e1_local_corpus_evaluation/local_cross_corpus/`.

### E2 — FL vs Centralized

**Вопрос:** как sample-weighted FedAvg соотносится с centralized training при
одинаковых actor folds, модели, seed и числе раундов?

Centralized и FedAvg запускаются тем же entry point:

```bash
uv run secureasr --config-file configs/experiments/config.e2_centr.yaml
uv run secureasr --config-file configs/experiments/config.e2_fl.yaml
```

Оба запуска автоматически оценивают финальный validated checkpoint на трёх
корпусах и сохраняют tidy CSV и YAML с per-corpus, macro и worst-corpus
метриками. FedAvg использует общую инициализацию, `full_epoch_v1`, веса по
размеру train dataset и агрегацию после последнего локального раунда.

### E3 — MergeSFL vs SFLv1 vs SFLour

**Вопрос:** как различаются реализованные split-learning протоколы по качеству,
ресурсам и динамике сходимости?

**Методы:** реконструкция MergeSFL Algorithm 1, personalized SplitFed как
рабочий SFLv1 reference и shared-server `concat_v1` как SFLour.

E3 является protocol-faithful end-to-end сравнением. Специфичные для каждого
метода cohort selection, workload, optimizer, aggregation cadence и server
ownership сохраняются и записываются как экспериментальные факторы, а не
нормализуются искусственно. Итоговые результаты относятся к целым протоколам;
причинный эффект отдельного механизма требует дополнительной controlled
ablation с выровненным бюджетом.

```bash
uv run secureasr --config-file configs/experiments/config.e3_mergesfl.yaml
uv run secureasr --config-file configs/experiments/config.e3_sflv1.yaml
uv run secureasr --config-file configs/experiments/config.e3_our.yaml
```

**Метрики:** convergence по effective samples и wall time, Anger F1, PR-AUC,
per/worst-corpus качество и дисперсия по порядкам клиентов и seeds.

### E4 — сравнение data workload policies SFLour

**Вопрос:** уменьшает ли выравнивание нагрузки переобучение малого корпуса и
доминирование большого корпуса?

| Policy | Определение |
| --- | --- |
| Balanced max steps | Лимиты 64/64/45 сохраняют один проход без повторов |
| Fixed steps | Все выполняют 143 шага; исчерпанные loaders запускаются циклически |

```bash
uv run secureasr --config-file configs/experiments/config.e4_our_balanced.yaml
uv run secureasr --config-file configs/experiments/config.e4_our_fixed_steps.yaml
```

Необходимо записывать unique/repeated samples, optimizer steps, aggregation
weight и effective contribution. Отчёт включает train-validation F1 gap,
per-corpus F1/PR-AUC, inter-client F1 range, macro/worst-corpus F1 и timing.
Для оценки переобучения используются validation actors, а не финальные test
actors.

## Next acceptance steps

1. Repeat current profiles across matched seeds and actor folds with saved provenance.
2. Verify numerical update semantics and canonical baseline claims independently.
3. Extend composed-checkpoint evaluation, repeated-sample and waiting-time accounting.

Implementation work is tracked in the [roadmap](development/ROADMAP.md), rather
than repeated as an implementation backlog in this experiment protocol.

## Связанные работы и сохранённые ссылки

- Thapa, C.; Mahawaga Arachchige, P. C.; Camtepe, S.; Sun, L.
  **SplitFed: When Federated Learning Meets Split Learning.** AAAI 2022,
  36(8), 8485–8493. [DOI](https://doi.org/10.1609/aaai.v36i8.20825),
  [страница издателя](https://ojs.aaai.org/index.php/AAAI/article/view/20825),
  [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/20825/20584),
  [arXiv](https://arxiv.org/abs/2004.12088) и
  [код](https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning).
  Репозиторий содержит `SFLV2_ResNet_HAM10000.py`. Конфигурацию Python 3.7.2,
  PyTorch 1.2.0, HAM10000 и ResNet18 необходимо адаптировать, не изменяя
  неявно порядок обновлений SFLv2.
- Liao, Y.; Xu, Y.; Xu, H.; Wang, L.; Yao, Z.; Qiao, C.
  **MergeSFL: Split Federated Learning with Feature Merging and Batch Size
  Regulation.** ICDE 2024, 2054–2067.
  [DOI](https://doi.org/10.1109/ICDE60146.2024.00164),
  [arXiv](https://arxiv.org/abs/2311.13348) и
  [training utilities](https://github.com/ymliao98/MergeSFL/blob/main/training_utils.py).
- Yang, J.; Liu, Y. **SCALA: Split Federated Learning with Concatenated
  Activations and Logit Adjustments.** [arXiv:2405.04875](https://arxiv.org/abs/2405.04875).
- Huang, C.; Tian, G.; Tang, M. **When MiniBatch SGD Meets SplitFed Learning:
  Convergence Analysis and Performance Evaluation.**
  [arXiv:2308.11953](https://arxiv.org/abs/2308.11953).

SCALA важна для сравнения, поскольку в ней конкатенация дополняется явной
корректировкой logits при label skew. MiniBatch-SFL относится к server-side
optimization и client drift при non-IID данных. Эти работы являются related
baselines, но не доказывают, что текущий queue runtime полностью реализует их
алгоритмы.
