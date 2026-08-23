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

## 2. Аудит текущей системы

### 2.1. Что уже реализовано

| Возможность | Подтверждение в репозитории | Использование |
| --- | --- | --- |
| CREMA-D, RAVDESS, SAVEE | Dataset processors и строгий `DatasetConfig` | E0 и cross-corpus запуски |
| Бинарные метки ANG/not-ANG | Dataset processors и валидация dataset | Основная задача |
| Actor-disjoint train/test split | `ConflictEmotionalDataset` и `split_seed` | Исключение утечки голоса диктора внутри корпуса |
| Train-only нормализация с учётом padding | Dataset pipeline и тесты нормализации | Все эксперименты качества |
| Акустические характеристики | MFCC, RMS, ZCR, Mel, contrast и `DataConcatenator` | E0 |
| Режим Centralized | Явная topology контроллера | Reference для E1 |
| Режим Federated | Полные локальные модели и обычный или dataset-weighted FedAvg | Baseline для E1/E2 |
| Split/shared и split/personalized | Явные варианты владения серверной моделью | Диагностика, но не замена SFLv1/SFLv2 |
| Текущий SplitFed/OUR | Общий сервер, синхронизация `(round, step)`, конкатенация activation и FedAvg клиентских частей | Основной исследуемый метод |
| Разная длина локальных loaders | Протокол `round_end` и логика готовности batch | Запуски с естественно неравными корпусами |
| Ограниченное ожидание и stale error | Тайм-ауты queue/barrier, удаление stale batch, коррелированный ответ об ошибке | Проверка протокола; телеметрия E4 неполна |
| Артефакты воспроизводимости | Resolved config, provenance, manifests, metrics и checkpoints | Каждый запуск |
| Accuracy, F1, precision, recall | Клиентский evaluator | Предварительное сравнение качества |
| Mode matrix | `src.experiments.mode_matrix` | Только smoke/диагностическое сравнение |
| Local-only cross-corpus matrix | `src.experiments.local_cross_corpus` | E1: независимое обучение на A и оценка на A/B/C |

Текущий сервер собирает сообщения с activation для одинаковых round и step,
конкатенирует их по размерности batch, выполняет один forward/backward общей
серверной модели, разделяет градиенты activation и возвращает каждому клиенту
его часть. Это отличие реализации `OUR` от канонического SFLv2.

### 2.2. Чего не хватает для научного сравнения

| Отсутствующий компонент | Что блокирует | Требуемый результат |
| --- | --- | --- |
| Канонический SFLv1 | Полнота набора методов | Агрегация обеих частей модели с явным владением |
| Канонический последовательный SFLv2 | E2–E4 | Воспроизводимый порядок клиентов и одно обновление сервера на client batch |
| Полная реализация MergeSFL | E3–E5 | Feature merging и batch regulation вместо простой конкатенации |
| PR-AUC, macro-F1, UAR и доверительные интервалы | E1–E7 | Per-client, per-corpus, macro и worst-client отчёты |
| Cross-corpus/held-out evaluator | E1/E6 | Оценка одного checkpoint на произвольном corpus view |
| Multi-seed harness | Все научные выводы | Одинаковые seeds, агрегированные таблицы и неопределённость |
| Ресурсная телеметрия | E2/E4/E5 | Bytes, round/wait/total time, параметры и память |
| Детерминированный симулятор задержек и отказов | E4 | Delay, jitter, пропуск step и длительный straggler |
| Явная транзакция abort/skip | E4 | Step с тайм-аутом не обновляет ни одну часть модели |
| Реестр workload policies | E5 | Full epoch, fixed maximum steps и equal-sample budget |
| Validation или nested actor split | E5 и выбор модели | Отсутствие настройки по финальным test actors |
| Leave-one-corpus-out topology | E6 | Held-out корпус не участвует в обучении и нормализации |
| Multi-class отображение меток | Опциональный E7 | Проверенные метки, новая output/loss схема и метрики |

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
checkpoint на train-corpus. Распространение того же evaluator contract на
Centralized, FedAvg и OUR остаётся следующим шагом блока.

1. Добавить PR-AUC по вероятностям, macro-F1, UAR, confusion counts и явную
   обработку одного класса или отсутствия положительных предсказаний.
2. Реализовать оценку валидированного checkpoint на любом настроенном корпусе
   без вычисления нормализации по оценочному корпусу.
3. Сохранять per-corpus, macro и worst-corpus результаты, seed, actors и число
   образцов.
4. Добавить согласованные multi-seed запуски и агрегированные доверительные
   интервалы.

**Готово, когда:** Local, Centralized, FedAvg и OUR создают одну схему
результатов на одинаковых actor folds.

### Блок B — точные алгоритмические baseline

1. Зафиксировано: `sflv2_sequential_v1` использует общий сервер,
   воспроизводимый порядок клиентов, одно обновление серверной модели на
   client batch и FedAvg клиентских частей. Текущий OUR baseline называется
   `concat_v1` и объединяет согласованные client batches перед одним server
   update.
2. Реализовать `sflv1_v1`: изолированные серверные модели во время локального
   обучения и их явная агрегация на глобальной границе.
3. Проверить эталонное поведение на детерминированной synthetic задаче, затем
   отдельно на каждом речевом корпусе.
4. Реализовывать MergeSFL только после фиксации feature ordering, batch
   regulation, loss weighting и уравнений server update в детерминированных
   тестах.

**Готово, когда:** каждая policy имеет стабильное имя/версию, тест владения
моделью и детерминированный reference test градиентов и обновлений.

### Блок C — семантика нагрузки и отказов

1. Определить транзакцию шага: `created -> completed` или `aborted`. Отменённый
   шаг не изменяет параметры ни клиента, ни сервера.
2. Добавить seeded delay, jitter, message drop и длительный straggler.
3. Добавить `full_epoch_v1`, `fixed_max_steps_v1` и `equal_samples_v1` с явным
   учётом циклического и повторного использования данных.
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
uv run python -m src.experiments.local_cross_corpus \
  --config-file configs/experiments/config.e1.local.yaml
```

Один `training.num_rounds` соответствует одной полной локальной эпохе. Все
три модели получают одинаковую инициализацию по `training.seed`. Для каждой
ячейки сохраняются accuracy, Anger F1, macro-F1, precision, recall, UAR,
PR-AUC, confusion counts и размеры классов. Если eval fold содержит один
класс, `pr_auc` записывается как `null`, а не как вводящее в заблуждение число.
Артефакты находятся в
`artifacts/e1_local_cross_corpus/local_cross_corpus/`.

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

1. Выполнить E0 с текущим notebook.
2. Реализовать блок A, затем выполнить E1.
3. Реализовать SFLv2 и телеметрию, затем выполнить E2.
4. Реализовать MergeSFL и ablation, затем выполнить E3.
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
