# FAQ и Решение Проблем

Часто задаваемые вопросы, troubleshooting и практические советы.

## Выбор модели

### Какую модель выбрать для моей задачи?

**Для быстрого прототипа:**
- Используйте **EASE** - быстро, просто, хорошие результаты

**Для implicit feedback (клики, просмотры):**
- Маленький датасет (< 100K): **EASE** или **ALS**
- Средний датасет (100K - 1M): **EASE** или **NCF** (если есть GPU)
- Большой датасет (> 1M): **ALS** или **LightGCN** (если есть GPU)

**Для explicit ratings (рейтинги 1-5):**
- Базовый: **SVD**
- С implicit feedback: **SVD++**

**Для sequential данных (есть timestamp):**
- **SASRec** (требуется GPU)

**Для максимальной точности:**
- **LightGCN** (требуется GPU и время)

### Сравнение моделей

| Модель | Скорость | Точность | Требования |
|--------|----------|----------|------------|
| EASE | ⚡⚡⚡ | ⭐⭐⭐ | CPU, < 1 min |
| SLIM | ⚡⚡ | ⭐⭐⭐ | CPU, 1-10 min |
| ALS | ⚡⚡ | ⭐⭐ | CPU, 1-5 min |
| SVD | ⚡⚡⚡ | ⭐⭐ | CPU, < 1 min |
| SVD++ | ⚡ | ⭐⭐⭐ | CPU, 5-30 min |
| NCF | ⚡⚡ | ⭐⭐⭐ | GPU preferred |
| LightGCN | ⚡ | ⭐⭐⭐⭐ | GPU required |
| SASRec | ⚡ | ⭐⭐⭐ | GPU required |

## Данные

### У меня есть рейтинги 1-5. Как их использовать?

```python
# Вариант 1: Explicit ratings (SVD, SVD++)
dataset = InteractionDataset(df, implicit=False)
model = SVDRecommender()

# Вариант 2: Конвертировать в implicit (все модели)
from recommender.data import binarize_implicit_feedback
df_implicit = binarize_implicit_feedback(df, threshold=4.0)  # >= 4 → positive
dataset = InteractionDataset(df_implicit, implicit=True)
```

### У меня только бинарные данные (клики). Какую модель?

```python
# Идеально для:
# - EASE (лучший выбор)
# - ALS
# - NCF
# - LightGCN

dataset = InteractionDataset(df, implicit=True)
model = EASERecommender(l2_reg=500.0)
```

### Сколько данных нужно минимум?

- **Минимум**: 1000 пользователей, 500 items, 10K взаимодействий
- **Хорошо**: 10K+ пользователей, 1K+ items, 100K+ взаимодействий
- **Отлично**: 100K+ пользователей, 10K+ items, 1M+ взаимодействий

**Фильтрация редких:**

```python
dataset = InteractionDataset(
    df,
    min_user_interactions=5,  # Минимум 5 взаимодействий на пользователя
    min_item_interactions=5   # Минимум 5 взаимодействий на item
)
```

### Как обработать timestamp?

```python
# Убедитесь что timestamp в правильном формате
df['timestamp'] = pd.to_datetime(df['timestamp'])
# Или
df['timestamp'] = df['timestamp'].astype(int)

# Используйте temporal split
train, test = dataset.split(strategy='temporal')
```

## Обучение

### Модель не сходится / низкое качество

**Проверьте данные:**

```python
# 1. Достаточно ли данных?
print(f"Users: {df['user_id'].nunique()}")
print(f"Items: {df['item_id'].nunique()}")
print(f"Interactions: {len(df)}")
print(f"Density: {len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4f}")

# 2. Есть ли редкие пользователи/items?
print(df['user_id'].value_counts().describe())
print(df['item_id'].value_counts().describe())

# 3. Фильтруйте редких
df = filter_by_interaction_count(df, min_user_interactions=5, min_item_interactions=5)
```

**Для deep learning моделей:**

```python
# Уменьшите learning rate
model = NCFRecommender(learning_rate=0.0001)  # Было 0.001

# Увеличьте epochs
model = NCFRecommender(epochs=50)  # Было 20

# Добавьте dropout
model = NCFRecommender(dropout=0.5)  # Было 0.2
```

### Переобучение (train >> val)

```python
# 1. Увеличьте регуляризацию
model = ALSRecommender(reg=0.1)  # Было 0.01

# 2. Уменьшите capacity
model = NCFRecommender(
    embedding_dim=32,  # Было 128
    hidden_layers=[64, 32]  # Было [256, 128, 64]
)

# 3. Early stopping
# Останавливайте если val метрика не улучшается 5-10 эпох

# 4. Больше данных
# Увеличьте train set или соберите больше данных
```

### Обучение слишком медленное

```python
# 1. Используйте GPU (для deep learning)
model = NCFRecommender(device='cuda')

# 2. Увеличьте batch size
model = NCFRecommender(batch_size=2048)  # Было 256

# 3. Уменьшите epochs/iterations
model = ALSRecommender(n_iterations=10)  # Было 15

# 4. Используйте более простую модель
# EASE вместо SLIM
# NCF вместо LightGCN
```

## Inference

### Cold start для новых пользователей

```python
def recommend_with_fallback(user_id, model, train_data, k=10):
    """
    Рекомендации с fallback на популярные items.
    """
    try:
        # Попробовать персонализированные
        return model.recommend([user_id], k=k)[user_id]
    except KeyError:
        # Новый пользователь → популярные items
        popular_items = train_data['item_id'].value_counts().head(k).index.tolist()
        return [(item_id, 1.0) for item_id in popular_items]
```

### Cold start для новых items

```python
# Решение 1: Exploration
def recommend_with_exploration(user_id, model, k=10, exploration_rate=0.1):
    # 90% персонализированных, 10% новых items
    n_exploit = int(k * 0.9)
    recs = model.recommend([user_id], k=n_exploit)[user_id]
    
    # Добавить новые items
    n_explore = k - n_exploit
    new_items = get_new_items(n_explore)
    recs.extend([(item_id, 0.5) for item_id in new_items])
    
    return recs

# Решение 2: Content-based для новых items
# Используйте features items для поиска похожих
```

### Рекомендации слишком популярные (нет diversity)

```python
# 1. Re-ranking для diversity
from sklearn.metrics.pairwise import cosine_similarity

def diversify_recommendations(recommendations, item_embeddings, lambda_param=0.5):
    """
    MMR (Maximal Marginal Relevance) для diversity.
    """
    diverse_recs = []
    candidate_items = recommendations.copy()
    
    while len(diverse_recs) < len(recommendations) and candidate_items:
        if not diverse_recs:
            # Первый item - самый релевантный
            diverse_recs.append(candidate_items.pop(0))
        else:
            # Баланс релевантности и diversity
            best_score = -float('inf')
            best_idx = 0
            
            for idx, (item_id, rel_score) in enumerate(candidate_items):
                # Similarity с уже выбранными
                item_emb = item_embeddings[item_id]
                selected_embs = [item_embeddings[i] for i, _ in diverse_recs]
                max_sim = max([cosine_similarity([item_emb], [emb])[0][0] 
                              for emb in selected_embs])
                
                # MMR score
                mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            diverse_recs.append(candidate_items.pop(best_idx))
    
    return diverse_recs

# 2. Уменьшите регуляризацию (меньше bias к популярным)
model = ALSRecommender(alpha=20)  # Было 40
```

## Production

### Out of Memory (GPU)

```python
# 1. Уменьшите batch size
model = NCFRecommender(batch_size=128)  # Было 1024

# 2. Уменьшите размерность
model = NCFRecommender(
    embedding_dim=32,
    hidden_layers=[64, 32]
)

# 3. Gradient accumulation
# Эффективный batch = batch_size * accumulation_steps

# 4. Очистите GPU cache
import torch
torch.cuda.empty_cache()
```

### API медленно отвечает

```python
# 1. Batch inference
# Обрабатывайте несколько запросов вместе

# 2. Кэширование
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_cached_recommendations(user_id, k):
    return model.recommend([user_id], k=k)[user_id]

# 3. FAISS для быстрого поиска
# См. главу 09_продакшн.md

# 4. Предвычислите рекомендации
# Периодически обновляйте рекомендации для всех пользователей
```

### Как обновлять модель в production?

**Стратегия 1: Периодическое переобучение**

```python
# Каждую неделю/месяц
# 1. Собрать новые данные
# 2. Переобучить модель
# 3. Оценить на validation
# 4. Если лучше → deploy новую модель
# 5. Иначе → оставить старую
```

**Стратегия 2: Incremental update**

```python
# Только для некоторых моделей (SGD-based)
# Дообучать на новых данных без переобучения с нуля
```

**Blue-Green Deployment:**

```python
# 1. Обучить новую модель (green)
# 2. Deploy рядом со старой (blue)
# 3. Направить часть трафика на green (A/B test)
# 4. Если OK → переключить весь трафик
# 5. Удалить blue
```

## Оценка

### Какие метрики использовать?

**Для ranking задач:**
- **NDCG@K** - лучшая метрика (учитывает позицию)
- **Recall@K** - важно найти все релевантные
- **Precision@K** - точность top-K

**Для rating prediction:**
- **RMSE** - стандартная метрика
- **MAE** - более интерпретируемая

**Beyond accuracy:**
- **Coverage** - разнообразие рекомендаций
- **Novelty** - насколько неожиданные

### Почему метрики низкие?

```python
# 1. Слишком разреженные данные
# Фильтруйте редких пользователей/items

# 2. Неправильный split
# Используйте temporal split для реалистичной оценки

# 3. Слишком большой K
# Метрики падают с ростом K - это нормально

# 4. Baseline для сравнения
# Сравните с популярными items
popular = PopularityBaseline()
popular.fit(train.data)
```

### Как интерпретировать метрики?

**Типичные значения (NDCG@10, implicit feedback):**
- **< 0.1**: Плохо (хуже random)
- **0.1 - 0.2**: Baseline (popularity)
- **0.2 - 0.3**: OK
- **0.3 - 0.4**: Хорошо
- **0.4 - 0.5**: Очень хорошо
- **> 0.5**: Excellent (редко)

## Общие вопросы

### Можно ли комбинировать модели?

Да! Используйте Model Ensemble:

```python
from recommender.utils import ModelEnsemble

ensemble = ModelEnsemble(
    models=[ease, als, ncf],
    weights=[0.4, 0.3, 0.3],
    strategy='weighted_average'
)

recommendations = ensemble.recommend(user_ids, k=10)
```

### Как добавить свою модель?

```python
from recommender.core import BaseRecommender

class MyRecommender(BaseRecommender):
    def __init__(self, my_param=1.0):
        super().__init__()
        self.my_param = my_param
    
    def fit(self, interactions):
        # Ваша логика обучения
        self._create_mappings(interactions)
        # ...
        self.is_fitted = True
        return self
    
    def predict(self, user_ids, item_ids):
        # Ваша логика предсказания
        pass
    
    def recommend(self, user_ids, k=10, exclude_seen=False):
        # Ваша логика рекомендаций
        pass
```

### Где найти больше информации?

- **README**: [../../../README.md](../../../README.md)
- **Examples**: [../../../examples/](../../../examples/)
- **Научные статьи**: См. разделы в документации моделей
- **GitHub Issues**: [github.com/hichnicksemen/svd-recommender/issues](https://github.com/hichnicksemen/svd-recommender/issues)

## Troubleshooting чеклист

Если что-то не работает, пройдитесь по этому чеклисту:

### 1. Установка

```bash
# Проверьте версию Python
python --version  # Должно быть >= 3.8

# Установите все зависимости
pip install -r requirements.txt

# Проверьте импорты
python -c "import recommender; print(recommender.__version__)"
```

### 2. Данные

```python
# Проверьте формат данных
print(df.head())
print(df.dtypes)
print(df.columns.tolist())

# Должны быть: user_id, item_id, [rating], [timestamp]

# Проверьте размер
print(f"Users: {df['user_id'].nunique()}")
print(f"Items: {df['item_id'].nunique()}")
print(f"Interactions: {len(df)}")
```

### 3. Модель

```python
# Проверьте что модель обучена
print(model.is_fitted)  # Должно быть True

# Проверьте mappings
print(f"n_users: {model.n_users}")
print(f"n_items: {model.n_items}")
```

### 4. Оценка

```python
# Проверьте что используете правильный task
# task='ranking' для implicit
# task='rating_prediction' для explicit

# Проверьте exclude_train
results = evaluator.evaluate(
    model, test,
    task='ranking',
    exclude_train=True,  # Важно!
    train_data=train
)
```

## Полезные ссылки

- **Документация**: [docs/ru/README.md](README.md)
- **Quickstart**: [../../examples/quickstart.py](../../examples/quickstart.py)
- **Тесты**: [../../tests/](../../tests/)
- **Changelog**: [../../CHANGELOG.md](../../CHANGELOG.md)

---

**Это последняя глава документации!**

**Вернуться к**: [Оглавлению](README.md)

