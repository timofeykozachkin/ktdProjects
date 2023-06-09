
<h1>ML. Task 1. Модель линейной регресси. Предсказание стоимости авто</h1>

**1. Что было сделано**

В рамках задания стояла цель научиться предсказывать по имеющимся признакам наблюдений-объвлений о продаже машин стоимость продаваемого авто. 

*Что было сделано с начальным набором данных:*
* Исследованы признаки
* Обработаны пропуски
* Построены распределения признаков и визуализирвоаны связи признаков друг с другом и целевой переменной

*Что было сделано с моделями ML:*
* Построены модели линейной регрессии для вещественных+категориальных признаков, а также для вещественных в отдельности
* К категориальным признакам был применен OneHotEncoding
* К модели добавлены L1-, L2-регуляризации
* ! (в рамках Feature Engineering) К начальным признакам добавлены признаки year^2 и max_power/mileage

*Что было сделано для доработки модели:*
* Обработаны крайние значения по признакам имеющихся наблюдений в трейн- и тест-выборках (нули по переменным mileage, max_power - заполнены средними для наблюдений по машинам со схожими характеристиками)
* Признаки были стандартизированы
* Исключены выбросы по целевой переменной, что добавило задачи условности! - научиться предсказывать стоимость машины среднего класса (менее 1,1 млн по selling_price). Т.е. стоит ожидать, что для объявлений с экслюзивными марками машин модель будет не способна давать корректные значения.
* Опробованы методы заполнения пропусков средними по колонкам, по категориям машин, были попытки составить  
создать еще новые переменные (1/km_driven, >/< Second owner), преобразовать имеющиеся переменные (логарифмирование), но существенного улучшения модели не наблюдалось, в связи с чем оставлены только указанные выше преобразования

*Что было сделано в рамках реализации модели:*
* Реализован сервис, принимающий на вход в формате JSON, CSV признаки автомобиля, выставленного на продажу, и выдающий на выход предсказываемую цену реализациипродажи

**2. С какими результатами**
* Была выбрана модель линейной регрессии Lasso (с L1-регуляризацией)
* Скор полученной модели по критерий R2: для трейна: 0.71438;  
                                         для теста:  0.72911
* По работе сервиса: пользователю необходимо ввести только базовые характеристики продваемого автомобиля, без преобразований - просто набор характеристик (правда в запрашиваемом порядке)!

**3. Что дало наибольший буст в качестве**
* Удаление выбросов из целевой переменной, конечно, изменило формулировку цели задачи и ограничила область применения модели (все автомобили кроме дорогих по умолчанию/эксклюзивных марок), однако это позволило увеличить качество модели и перешагнуть границу скора по R2 0.7
* Новые признаки также улучшили качество. Преобразовав имеющиеся признаки (год) исходя из их взаимосвязи с целевой переменной, получилось более корректно использовать связь признаков и целевой переменной
* GridSearchCV действительно позволял находить наиболее оптимальные характеристики модели с регуляризацией

**4. Что сделать не вышло и почему**
* Не получилось создать много новых признаков из имеющихся, потому что многие характеристики машины описывают уникальную черту машины, что не позволяет ими манипулировать и комбинировать друг с другом
* Не получилось сделать много преобразований над переменными, т.к. все базовые преобразования именно над признаками (логарифмирование, построение функциональных взаимосвязей и пр.) не давали никакого улучшения модели
* Большого значения качетсва по бизнес-метрики, т.к. возможно используемые 

-- остальную пошаговую аналитику постарался расписывать оранжевым текстом в ноутбуке

P.S. Прикладываю примеры работы сервиса - функционал: predict_item, predict_items, upload (импорт базового/экспорт дополненного csv-файла).

**Сервис FastApi.** Перед запуском на локале - команды в терминале: 
1. pip install -r requirements.txt
2. uvicorn main:app --reload

![logo](https://github.com/timofeykozachkin/ML-MLDS-tasks/blob/d2825316254b85c7591bd4da9c58230460798b81/screenshots/inner_interface.png)

predict_item (предсказание цены по 1 наблюдению)

![logo](https://github.com/timofeykozachkin/ML-MLDS-tasks/blob/d2825316254b85c7591bd4da9c58230460798b81/screenshots/predict_item.png)

predict_items (предсказание цены по нескольким наблюдениям, переданным в формате JSON)

![logo](https://github.com/timofeykozachkin/ML-MLDS-tasks/blob/d2825316254b85c7591bd4da9c58230460798b81/screenshots/predict_items.png)

upload (предсказание цены по нескольким наблюдениям, переданным в формате CSV)
    импортированный на сервис CSV-файл

![logo](https://github.com/timofeykozachkin/ML-MLDS-tasks/blob/d2825316254b85c7591bd4da9c58230460798b81/screenshots/csv_file_example.png)

    работа сервиса

![logo](https://github.com/timofeykozachkin/ML-MLDS-tasks/blob/d2825316254b85c7591bd4da9c58230460798b81/screenshots/upload.png)

    экспортированный с сервиса CSV-файл

![logo](https://github.com/timofeykozachkin/ML-MLDS-tasks/blob/d2825316254b85c7591bd4da9c58230460798b81/screenshots/csv_output.png)


