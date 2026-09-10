# Экспериментальный план

## 1. Область исследования и терминология

SecureASR оценивается как pre-alpha фреймворк для бинарной классификации
эмоциональной речи. `ANG` является положительным классом; все остальные
поддерживаемые эмоции отображаются в класс `0`. Проект не является системой
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

## 2. Аудит текущего среза: цель, результат и разрыв

Аудит выполнен по текущим entry points, конфигурационной схеме, владельцам
моделей, экспериментальным harness и тестам. Наличие класса или режима само по
себе не считается готовым научным baseline: нужен воспроизводимый запуск,
однозначная семантика обновлений и сопоставимый контракт результатов.

### 2.1. Сопоставление по целям

| Цель | Требуемый результат | Что есть сейчас | Чего нет / следующий проверяемый шаг |
| --- | --- | --- | --- |
| Измерить исходную неоднородность корпусов | Воспроизводимые оценки label/acoustic shift без обучения целевой модели | E0 notebook, валидный dataset-only config, actor IDs, extraction-loss counts, duration, JS/Wasserstein/energy/KS, MMD, domain AUC и PCA | Автоматический export tidy-таблиц и manifest из notebook; повторные seeds/folds и итоговая сводка с неопределённостью |
| Получить Local cross-corpus reference | Полная матрица train-corpus × eval-corpus с train-corpus normalization | `src.experiments.local_cross_corpus`, checkpoints, CSV/YAML, accuracy, Anger F1, macro-F1, precision, recall, UAR, PR-AUC и confusion counts | Multi-seed aggregation, доверительные интервалы и единый evaluator для checkpoint других методов |
| Получить Centralized reference | Одна полная модель на объединённых train actors и отдельные результаты на каждом test corpus | Рабочий topology, checkpoint и автоматическая per-corpus оценка по общему binary-metrics contract | Multi-seed aggregation, доверительные интервалы и matched-budget сравнение |
| Получить FedAvg reference | Полные клиентские модели, валидированный FedAvg и per-corpus evaluation | Общая инициализация, `full_epoch_v1`, sample-weighted финальная агрегация, checkpoint и автоматическая per-corpus оценка | Multi-seed aggregation, доверительные интервалы и effective-sample accounting |
| Зафиксировать классический SFLv2 | Общий сервер обслуживает клиентов в фиксированном порядке до `round_end`, обновляясь после каждого client batch; клиентские части проходят FedAvg | `sequential_v1`, выбор strategy в schema/server, фиксированный порядок `client_channels`, один server `optimizer.step()` на client batch и `spawn` smoke с неравными local steps | Детерминированный численный reference порядка и параметров, запись порядка в artifacts и полноценный benchmark harness |
| Зафиксировать OUR | Согласованные `(round, step)` batches объединяются перед одним server update; клиентские части проходят FedAvg | `concat_v1`, конкатенация activation, разделение activation gradients, один server update без batch-gradient averaging, stale-batch timeout и `spawn` smoke | Научный harness с тем же data/evaluation budget, телеметрия ожидания/bytes и multi-seed сравнение с SFLv2 |
| Получить SFLv1 | Изолированные server/client pairs локально обучаются, затем обе части агрегируются по явно заданной политике | Personalized split хранит отдельные server models и optimizer states | Нет агрегации серверных частей, общего SFLv1 checkpoint contract и эталонного теста одного global round |
| Получить MergeSFL | Реализованы feature merging, batch-size regulation и соответствующая оптимизация | Только простая конкатенация в `concat_v1` | Весь алгоритмический baseline MergeSFL и его reference tests; `concat_v1` нельзя переименовывать в MergeSFL |
| Сопоставить качество всех методов | Одинаковые actor folds, normalization policy, seeds, stopping/data budget и единая tidy-схема | Local, Centralized и FedAvg используют общий metrics contract; complete-model checkpoint evaluator сохраняет per-corpus, macro и worst-corpus результаты | Распространить контракт на SFLv2/OUR; добавить validation actors, multi-seed CI и matched-budget accounting |
| Сопоставить вычислительную стоимость | Per-process параметры/память, transmitted bytes, server wait, round и total time | Ограниченные lifecycle logs, deadlines, quorum и stale counters на уровне протокола | Версионированная телеметрия, единицы измерения, warm-up policy и экспорт в общую таблицу |
| Проверить задержки и отказы | Seeded delay/drop/straggler без deadlock и без частично применённого шага | Тайм-ауты, cancellation, barrier abort, failure propagation и завершение процессов тестируются | Детерминированный fault simulator и транзакция `completed/aborted`, гарантирующая отсутствие server/client update при сорванном шаге |
| Проверить неравную нагрузку | Явные `full_epoch`, `fixed_steps`, `equal_samples` с учётом повторов | Типизированы `max_steps_v1`, `full_epoch_v1` и cycling `fixed_steps_v1`; разные длины loaders и `round_end` поддерживаются, effective resource samples учитывают повторы | Реализовать отдельный repeated-sample counter, equal-samples, fairness и matched-budget experiments |
| Проверить перенос на неизвестный корпус | Held-out corpus исключён из обучения, нормализации и model selection | Local matrix измеряет перенос A → B/C, но обучающая постановка остаётся однокорпусной | Leave-one-corpus-out topology для методов совместного обучения и отдельный held-out evaluator |

### 2.2. Итог аудита текущего среза

- Инженерный runtime уже поддерживает пять режимов (`local`, `centralized`,
  `federated`, `split`, `splitfed`) и две явные стратегии общей split-server
  модели: `concat_v1` и `sequential_v1`.
- E0 остаётся notebook-анализом и помечен `analysis_only`. Local создаёт полную
  матрицу, а Centralized/FedAvg автоматически оценивают финальный complete-model
  checkpoint по каждому корпусу. SFLv2/OUR ещё не сведены к этому контракту.
- Ближайший научно полезный срез — не новый transport или scheduler, а общий
  evaluator и matched-budget harness для уже реализованных методов.
- SFLv1, MergeSFL, fault simulator, расширенные workload policies и
  leave-one-corpus-out остаются отдельными последующими срезами; их нельзя
  считать реализованными по наличию personalized mode, конкатенации или
  тайм-аутов.

### 2.3. Ограничения данных

- Корпуса являются естественными доменами, а не IID-частями одной выборки.
  Различия дикторов, условий записи, объёмов, длительностей и баланса классов
  должны быть измерены, а не предположены.
- Actor-disjoint разбиение обязательно. Test actors не могут участвовать в
  обучении, нормализации, early stopping или подборе гиперпараметров.
- В SAVEE только четыре диктора. Одно actor-disjoint разбиение имеет высокую
  дисперсию и может дать нестабильное покрытие классов. Следует использовать
  leave-one-actor-out или все допустимые actor-held-out folds. Нельзя заменять
  их случайным разбиением отдельных записей.
- В cross-corpus постановке идентификатор клиента совпадает с идентификатором
  корпуса. Всегда нужны per-corpus и worst-corpus результаты.

## 3. План необходимых доработок

### Блок A — единый контракт оценки

Local-only часть реализована в `src.experiments.local_cross_corpus`: она
строит полную матрицу train-corpus × eval-corpus, использует статистики
нормализации только train-корпуса и сохраняет tidy CSV, YAML-матрицы и один
checkpoint на train-corpus. Centralized и FedAvg используют общий evaluator
финального complete-model checkpoint. Распространение контракта на SFLv2 и OUR
остаётся следующим шагом блока.

1. Распространить общий binary-metrics/checkpoint contract с complete-model
   методов на составные checkpoints SFLv2 и OUR.
2. Добавить согласованные multi-seed запуски и агрегированные доверительные
   интервалы.

**Готово, когда:** Local, Centralized, FedAvg, SFLv2 и OUR создают одну схему
результатов на одинаковых actor folds и matched data budget.

### Блок B — точные алгоритмические baseline

1. Добавить детерминированный численный reference для уже реализованных
   `sequential_v1` и `concat_v1`: проверить порядок client updates,
   число server optimizer steps и итоговые параметры на малой задаче.
2. Записывать выбранную server strategy и фактический порядок клиентов в
   resolved artifacts каждого запуска.
3. Реализовать `sflv1_v1`: изолированные серверные модели во время локального
   обучения и их явная агрегация на глобальной границе.
4. Проверить эталонное поведение SFLv1 на synthetic задаче, затем
   отдельно на каждом речевом корпусе.
5. Реализовывать MergeSFL только после фиксации feature ordering, batch
   regulation, loss weighting и уравнений server update в детерминированных
   тестах.

**Готово, когда:** каждая policy имеет стабильное имя/версию, тест владения
моделью и детерминированный reference test градиентов и обновлений.

### Блок C — семантика нагрузки и отказов

1. Определить транзакцию шага: `created -> completed` или `aborted`. Отменённый
   шаг не изменяет параметры ни клиента, ни сервера.
2. Добавить seeded delay, jitter, message drop и длительный straggler.
3. Расширить реализованные `full_epoch_v1` и cycling `fixed_steps_v1` политикой
   `equal_samples_v1` и отдельным учётом повторно использованных данных.
4. Записывать attempted/completed/aborted/stale steps, effective/repeated
   samples, server wait time, round time и bytes.

**Готово, когда:** сценарии отказов и нагрузки завершаются без deadlock, а их
учётные соотношения проверяются тестами.

### Блок D — экспериментальный harness

1. Разделить профили `replication` и `cross_corpus`.
2. Зафиксировать manifests, actor folds, порядок features, инициализацию,
   optimizers, stopping rule и согласованный набор seeds.
3. Сохранять одну tidy-таблицу каждого запуска и одну агрегированную таблицу
   каждого эксперимента.
4. Не смешивать smoke tests и benchmark timing.

**Готово, когда:** результат воспроизводится из resolved config и manifest без
ручного изменения notebook.

## 4. Обновлённая последовательность экспериментов

### E0 — проверка естественной cross-corpus неоднородности

**Вопрос:** являются ли CREMA-D, RAVDESS и SAVEE статистически различимыми
доменами?

**Артефакт:** [`notebooks/E0_cross_corpus_heterogeneity.ipynb`](../notebooks/E0_cross_corpus_heterogeneity.ipynb).

**Метрики:** число samples/actors/classes, потери extraction, длительность,
RMS, ZCR, MFCC, распределения до и после train-only стандартизации,
Jensen–Shannon, Wasserstein/energy distance, скорректированные KS tests,
permutation MMD и actor-grouped domain-classification ROC-AUC.

**Решение:** использовать cross-corpus выводы только при согласии нескольких
метрик размера эффекта и идентификации домена. E0 не доказывает пользу
совместного обучения.

### E1 — проверка пользы совместного обучения

**Вопрос:** улучшает ли обмен знаниями результат относительно изолированного
обучения?

**Методы:** Local, Centralized, FedAvg.

- Воспроизведение: отдельный запуск на каждом корпусе.
- Cross-corpus: три локальные модели, затем Centralized и FedAvg по трём
  клиентам.
- Каждая локальная модель оценивается на всех трёх test corpora. Centralized и
  FedAvg оцениваются по такой же трёхколоночной матрице.

**Метрики:** Anger F1, PR-AUC, recall, precision, macro и worst-corpus значения.
Accuracy используется как вспомогательная метрика.

Local-only матрица запускается без каналов и серверов:

```bash
uv run secureasr \
  --config-file configs/experiments/config.e1.1.yaml
```

Конфиг явно выбирает `training.mode: local`; основной CLI направляет такой
запуск в local cross-corpus runner без `TrainingController`, каналов и серверов.
Один `training.num_rounds` соответствует одной полной локальной эпохе. Все
три модели получают одинаковую инициализацию по `training.seed`. Для каждой
ячейки сохраняются accuracy, Anger F1, macro-F1, precision, recall, UAR,
PR-AUC, confusion counts и размеры классов. Если eval fold содержит один
класс, `pr_auc` записывается как `null`, а не как вводящее в заблуждение число.
Артефакты находятся в
`artifacts/e1_local_cross_corpus/local_cross_corpus/`.

Centralized и FedAvg запускаются тем же entry point:

```bash
uv run secureasr --config-file configs/experiments/config.e1.2_centr.yaml
uv run secureasr --config-file configs/experiments/config.e1.2_fl.yaml
```

Оба запуска автоматически оценивают финальный validated checkpoint на трёх
корпусах и сохраняют tidy CSV и YAML с per-corpus, macro и worst-corpus
метриками. FedAvg использует общую инициализацию, `full_epoch_v1`, веса по
размеру train dataset и агрегацию после последнего локального раунда.

### E2 — сравнение full-model FL и split training

**Вопрос:** оправдано ли разделение модели при сопоставимом качестве и объёме
обработанных данных?

**Методы:** FedAvg, канонический SFLv2, OUR. SFLv1 — опциональный архитектурный
reference; MergeSFL относится к E3.

**Контроль:** одинаковые actor folds, initialization seeds, effective samples,
семейство optimizer, частота evaluation и stopping budget.

**Метрики:** метрики качества E1, число клиентских параметров, transmitted
bytes, round time, total training time и peak process RSS/device memory, если
доступно.

### E3 — выделение эффекта server-side optimization

**Вопрос:** что даёт синхронизированное/объединённое обновление сервера по
сравнению с последовательным SFLv2?

**Методы:** канонический SFLv2, простая синхронизированная конкатенация
(`OUR-core`), полная реализация MergeSFL и полный OUR при включённой workload
policy.

**Ablation:** sequential/concatenated server batch; fixed/random client order;
простая конкатенация/MergeSFL mixing; batch regulation off/on.

**Метрики:** convergence по effective samples и wall time, Anger F1, PR-AUC,
per/worst-corpus качество и дисперсия по порядкам клиентов и seeds.

### E4 — устойчивость к системной неоднородности

**Вопрос:** остаётся ли синхронизированная оркестрация корректной при задержках
и отказах?

**Методы:** SFLv2, MergeSFL, OUR.

| Сценарий | Контролируемое условие |
| --- | --- |
| A: контроль | Одинаковые клиенты, искусственная задержка `0` |
| B: медленный клиент | Для одного клиента добавляется `0.25T0`, `0.5T0`, `T0` или `2T0` |
| C: jitter | Seeded задержка из `[0, 2T0]` для заданной доли шагов |
| D: временный отказ | Activation пропускается с `p` из `{0.05, 0.10, 0.25}` |
| E: длительный straggler | Один клиент пропускает несколько шагов до превышения deadline |

`T0` измеряется во время фиксированного warm-up на том же оборудовании.
Необходимо записывать F1/PR-AUC, server waiting time, round/total time,
completed/attempted steps, stale/created batches, aborted steps и effective
samples. E4 заблокирован до появления транзакционной семантики abort/skip.

### E5 — сравнение политик неравной нагрузки

**Вопрос:** уменьшает ли выравнивание нагрузки переобучение малого корпуса и
доминирование большого корпуса?

| Policy | Определение |
| --- | --- |
| Full local epoch | Каждый клиент обрабатывает полную локальную эпоху |
| Fixed maximum steps | Все выполняют число шагов крупнейшего клиента; исчерпанные loaders запускаются циклически |
| Equal samples | Все клиенты вносят одинаковый заданный sample budget за round |

Необходимо записывать unique/repeated samples, optimizer steps, aggregation
weight и effective contribution. Отчёт включает train-validation F1 gap,
per-corpus F1/PR-AUC, inter-client F1 range, macro/worst-corpus F1 и timing.
Для оценки переобучения используются validation actors, а не финальные test
actors.

### E6 — обобщение на неизвестный корпус

**Вопрос:** обучается ли распределённый метод независимым от корпуса признакам
гнева?

**Методы:** Centralized, FedAvg, SFLv2, MergeSFL, OUR.

В каждом fold один корпус полностью исключается из обучения, нормализации,
подбора параметров и выбора модели. Модель обучается на двух других корпусах и
однократно оценивается на held-out корпусе. Для согласованных seeds записываются
per-fold и macro Anger F1, PR-AUC, recall, precision и UAR.

### E7 — опциональная проверка multi-class постановки

Эксперимент выполняется только после аудита общего пересечения emotion labels и
обновления schema, loss, output head, metrics и tests. Достаточно одного
компактного сравнения с macro-F1, UAR, per-class recall и confusion matrix. Если
в названии статьи anger detection явно указан как case study, E7 остаётся
опциональным.

## 5. Порядок выполнения

1. Завершить E0: экспортировать таблицы notebook, выполнить согласованные
   actor folds/seeds и сохранить итоговую сводку.
2. Завершить блок A и E1: привести Centralized и FedAvg к Local cross-corpus
   контракту оценки.
3. Закрыть reference-тесты `sequential_v1` и `concat_v1`, добавить
   matched-budget harness и минимальную ресурсную телеметрию, затем выполнить
   E2.
4. Выполнить E3 как прямую ablation SFLv2 против `concat_v1`; добавлять
   MergeSFL только как отдельный полностью реализованный baseline.
5. Реализовать блок C, затем выполнить E4 и E5.
6. Реализовать held-out evaluator, затем выполнить E6.
7. Выполнять E7 только при необходимости для заявленной области статьи.

## 6. Связанные работы и сохранённые ссылки

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
