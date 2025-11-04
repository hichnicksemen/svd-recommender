# Тесты для SOTA Recommender Systems Library

Этот каталог содержит комплексный набор автотестов для библиотеки рекомендательных систем.

## Структура тестов

### test_recommender.py
Базовые тесты для основной функциональности:
- `TestDataset` - тесты для InteractionDataset
- `TestSimpleModels` - тесты для EASE и SLIM
- `TestMatrixFactorization` - тесты для SVD и ALS
- `TestEvaluation` - тесты для Evaluator
- `TestSaveLoad` - тесты сохранения/загрузки моделей

### test_data.py
Тесты для обработки данных:
- `TestSyntheticData` - генерация синтетических данных
- `TestInteractionDataset` - работа с датасетами
- `TestPreprocessing` - препроцессинг данных
- `TestSamplers` - negative sampling
- `TestDataLoaders` - загрузка реальных датасетов

### test_models.py
Детальные тесты моделей:
- `TestEASERecommender` - EASE модель
- `TestSLIMRecommender` - SLIM модель
- `TestSVDRecommender` - SVD модель
- `TestSVDPlusPlusRecommender` - SVD++ модель
- `TestALSRecommender` - ALS модель
- `TestEdgeCases` - граничные случаи

### test_metrics.py
Тесты метрик оценки:
- `TestRankingMetrics` - метрики ранжирования (Precision, Recall, NDCG, etc.)
- `TestRatingPredictionMetrics` - метрики предсказания (RMSE, MAE, etc.)
- `TestEdgeCases` - граничные случаи для метрик

### test_integration.py
Интеграционные тесты:
- `TestCompleteWorkflow` - полные сценарии использования
- `TestModelComparison` - сравнение моделей
- `TestColdStart` - холодный старт
- `TestDataQuality` - качество данных

## Запуск тестов

### Запуск всех тестов
```bash
python tests/run_tests.py
```

### Запуск конкретного файла тестов
```bash
python tests/run_tests.py --pattern="test_data.py"
```

### Запуск конкретного тестового класса
```bash
python tests/run_tests.py TestDataset
```

### Запуск с pytest
```bash
pytest tests/
pytest tests/test_models.py
pytest tests/test_models.py::TestEASERecommender
```

### Запуск с покрытием кода
```bash
pytest tests/ --cov=recommender --cov-report=html
```

## Требования

Базовые тесты требуют:
- numpy
- pandas
- scikit-learn
- scipy

Дополнительные зависимости для полного покрытия:
- pytest (для pytest runner)
- pytest-cov (для отчетов покрытия)

## Разработка новых тестов

При добавлении новых тестов:

1. **Именование**: Все тестовые файлы должны начинаться с `test_`
2. **Структура**: Используйте классы unittest.TestCase
3. **Документация**: Добавляйте docstring к каждому тесту
4. **Изоляция**: Каждый тест должен быть независимым
5. **Данные**: Используйте `create_synthetic_dataset` для тестовых данных
6. **Очистка**: Используйте setUp/tearDown для инициализации/очистки

Пример:
```python
class TestNewFeature(unittest.TestCase):
    \"\"\"Test new feature functionality.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures.\"\"\"
        self.df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            seed=42
        )
    
    def test_basic_functionality(self):
        \"\"\"Test basic feature behavior.\"\"\"
        # Test code here
        self.assertTrue(condition)
```

## CI/CD

Тесты автоматически запускаются через GitHub Actions при:
- Push в ветки master/main/develop
- Pull requests
- На разных OS (Ubuntu, macOS, Windows)
- На разных версиях Python (3.8-3.11)

См. `.github/workflows/tests.yml` для деталей.

## Метрики покрытия

Целевое покрытие кода: **> 80%**

Текущее покрытие можно проверить:
```bash
pytest tests/ --cov=recommender --cov-report=term
```

## Полезные команды

```bash
# Запуск только быстрых тестов
pytest tests/ -m "not slow"

# Запуск только unit тестов  
pytest tests/ -m unit

# Запуск только integration тестов
pytest tests/ -m integration

# Verbose output
python tests/run_tests.py -v

# Minimal output
python tests/run_tests.py -q
```

