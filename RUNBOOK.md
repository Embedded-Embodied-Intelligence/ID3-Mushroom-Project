# RUNBOOK — ID3 × Mushroom

> **Status:** specyfikacja implementacyjna v1.1 · **Data:** 2026-10-16  
> **Cel:** pełny projekt akademicki z algorytmu **ID3 (Iterative Dichotomiser 3)** na zbiorze **Mushroom (UCI)**.  
> **Język notebooka i prezentacji:** polski. **Język kodu:** angielski (nazwy funkcji, klas, zmiennych i plików).  
> **Docelowy odbiorca:** agent programistyczny (Claude Code CLI) implementujący projekt autonomicznie od początku do końca.

## 0. Mandat agenta i priorytety

Przyjmij rolę senior Python/ML engineer, data scientist, research engineer i redaktora materiałów dydaktycznych. Na podstawie tego dokumentu zaprojektuj, zaimplementuj, przetestuj i przygotuj do oddania kompletny projekt. **Nie twórz jedynie szablonu notebooka; wszystkie rozdziały muszą zawierać działające obliczenia i ich interpretację.**

Priorytety w kolejności:
1. Bezwzględne pokrycie wszystkich **8 wymagań prowadzącego** oraz wymogów formalnych ze slajdu.
2. Poprawność matematyczna i wierność **klasycznemu ID3**.
3. Rzetelne przygotowanie danych, walidacja, brak wycieku danych i reprodukowalność.
4. Notebook jako jednocześnie projekt, sprawozdanie i czytelna prezentacja na wykładzie.
5. Łatwe uruchomienie w **JupyterLab lokalnie** oraz w **Google Colab**.
6. Pełny, samodzielny warsztat dla grupy na **20 minut**, dostępny także do późniejszej pracy domowej.

**Zakazy:** Nie zastępuj ID3 przez `sklearn.tree.DecisionTreeClassifier(criterion='entropy')` (to nadal CART z podziałami binarnymi). Nie przeprowadzaj niezamówionego benchmarku z innymi algorytmami. Nie wymyślaj wyników, metryk, wniosków ani danych źródłowych. Nie wykorzystuj testu podczas doboru parametrów. Nie wgrywaj prywatnych plików na publiczne repozytoria. Nie uznawaj projektu za ukończony bez automatycznego wykonania notebooka i testów.

### 0.1. Osiem wymagań ze slajdu — mapa zgodności

Wymóg formalny ze slajdu:
> *„Forma: Rmarkdown albo Jupyter. Dane i plik Rmd albo ipynb spakować ZIP. Przesłać na email co najmniej jeden dzień wcześniej (warunek konieczny).”*

| Nr | Wymóg prowadzącego (dosłowny zapis ze slajdu) | Obowiązkowy artefakt / rozdział |
|---|---|---|
| 1 | Wprowadzenie, Przedstawienie algorytmu (pseudokod), Zestawienie bibliotek zawierających algorytm. | Notebook 01 + bibliografia |
| 2 | Dane, Analiza danych, Zbalansowanie, Korelacje, Braki w danych, Błędne dane, Czyszczenie danych. Uzupełnianie danych. Usunięcie zbędnych atrybutów. | Notebook 02–03; raport jakości danych |
| 3 | Przygotowanie danych do budowy modelu. `???`, Dane uczące, walidacyjne i testowe. | Notebook 04; jawny protokół podziału, kodowania/dyskretyzacji i preprocessing |
| 4 | Budowa kilku modeli próbnych w celu dostrojenia parametrów algorytmu, ocena jakości modeli. Dobór parametrów algorytmu do budowy modelu finalnego. `( Nie ??? w przypadku lasów)` | Notebook 06; tabela eksperymentów, wykresy i komentarz dot. pojedynczych drzew vs lasów |
| 5 | Budowa modelu finalnego. | Notebook 07; jawna konfiguracja i trening na `train+val` |
| 6 | Graficzne przedstawienie drzewa, Zestawienie bibliotek generujących grafikę. | Notebook 08; eksport grafiki i tabela bibliotek |
| 7 | Ocena modelu. Macierz pomyłek. Predykcje. | Notebook 09; metryki, predykcje i interpretacja |
| 8 | Przygotować inne dane i zadanie dla grupy, które zostanie wykonane na zajęciach. Kod umieścić w sieci w miejscu do którego mają dostęp tylko osoby z grupy. | Notebook 10 + `group_exercise/` + instrukcje dostępu |

**Wyjaśnienie dydaktycznych znaków `???` ze slajdu prowadzącego:**
1. W punkcie 3 znak `???` jest pytaniem kontrolnym prowadzącego o kroki przygotowania danych przed podziałem na zbiory. W Rozdziale 04 jawnie omów i zrealizuj:
   - wydzielenie macierzy cech `X` i wektora docelowego `y`,
   - kwestię kodowania zmiennych kategorycznych (wyjaśnij, czemu klasyczny ID3 działa natywnie na symbolach nominalnych bez One-Hot/Ordinal Encodingu) oraz teoretyczny wymóg **dyskretyzacji zmiennych ciągłych** w klasycznym ID3,
   - losowe przetasowanie (*shuffling*) i **stratyfikację** względem klasy docelowej,
   - dopasowanie transformacji (np. imputacji) wyłącznie na zbiorze uczącym (*fit on train, transform on val/test*).
2. W punkcie 4 dopisek `( Nie ??? w przypadku lasów)` odnosi się do tematów projektowych opartych na Lasach Losowych (*Random Forests*), gdzie nie przycina się pojedynczych drzew (*pruning*) oraz można korzystać z estymacji *Out-Of-Bag (OOB)*. W Rozdziale 06 dodaj krótki akapit wyjaśniający, dlaczego dla pojedynczego drzewa ID3 stosujemy regularyzację wzrostu (*pre-pruning*: `max_depth`, `min_samples_split`, `min_information_gain`) oraz wydzielony zbiór walidacyjny, w przeciwieństwie do nieskracanych drzew w lasach losowych.

### 0.2. Docelowe kryteria oceny

Ocenę projektu oprzyj na mierzalnych przesłankach: działającym algorytmie, jakości analiz, uzasadnieniu wyborów, wynikach eksperymentów, poprawnym protokole walidacji, czytelnych wizualizacjach i jakości warsztatu. Nie obiecuj oceny 5.0 ani określonej skuteczności predykcji przed wykonaniem obliczeń.

---

## 1. Dane i pochodzenie

**Główne źródło:** https://archive.ics.uci.edu/dataset/73/mushroom  
**Wskazane archiwum:** https://archive.ics.uci.edu/static/public/73/mushroom.zip  
**DOI:** https://doi.org/10.24432/C5959T  
**Licencja wg strony UCI:** CC BY 4.0; zapewnij prawidłowe przypisanie źródła w notebooku i README.

### 1.1. Kontrakt danych

Oryginalny plik `agaricus-lepiota.data` (wewnątrz archiwum) jest plikiem CSV **bez nagłówka**. Ma zawierać:
- **8124 wiersze**, **23 kolumny**: klasa + 22 kategoryczne cechy;
- etykiety: `e` = edible, `p` = poisonous; oficjalny rozkład: **4208 `e`, 3916 `p`**;
- dokładnie **2480 znaków `?`** w atrybucie `stalk-root` (zgodnie z oryginalnym plikiem `.names`);
- kodowanie kategorii zgodne ze słownikiem `agaricus-lepiota.names`.

Weryfikuj te liczby za pomocą `assert`/kontroli jakości **po wczytaniu oryginalnego zbioru**, a nie po modyfikacji. Jeśli konkretny wariant archiwum ma inną strukturę, zakończ etap danych czytelnym błędem i sprawdź źródło zamiast zgadywać.

Nazwy cech w kolejności oryginalnego pliku:

```python
FEATURE_NAMES = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat',
]
TARGET_NAME = 'class'
```

Wczytuj dane **jako stringi** i `?` jawnie zamieniaj na brak (`pd.NA`/`np.nan`), nie na prawidłową wartość biologiczną. Nie korzystaj z domyślnego kodowania numerycznego kategorii jako wartości porządkowych. Zaimplementuj funkcję prezentacji rozwiniętych nazw (np. `odor=foul (f)` zamiast samego `odor=f`) z mapowaniem wg dokumentacji UCI.

### 1.2. Dostęp do danych — dwa tryby

- **Lokalnie/offline:** użyj dołączonego `data/mushroom.zip` i odczytu strumieniowego `zipfile`, bez żądania dostępu do internetu. Nie wymagaj wcześniejszego ręcznego wypakowania.
- **Colab:** notebook ma znaleźć lokalny ZIP albo, jeśli go nie ma, pobrać oryginalne archiwum z powyższego URL i zapisać lokalnie. Przy braku połączenia pokaż jasne wskazówki, jak wczytać ZIP ręcznie.

**WAŻNA DECYZJA ARCHITEKTONICZNA:** główny notebook ma być **samowystarczalny kodowo w Colab**, tzn. nie może wymagać importowania niedostępnego `src/` z prywatnego repozytorium. Jednocześnie projekt ma mieć testowalne moduły w `src/`. Rozwiąż to deterministycznie (np. `scripts/build_notebook.py` wstawiający kod źródłowy modułów do komórek notebooka przy jego generowaniu). **Nie utrzymuj ręcznie dwóch rozbieżnych implementacji ID3.** Dokumentuj mechanizm synchronizacji.

> **Szczegół implementacyjny dla `scripts/build_notebook.py`:**  
> Wstrzykując kod z modułów `src/*.py` do komórek notebooka, skrypt musi automatycznie usuwać/odfiltrowywać wewnętrzne importy pakietowe (np. `from src.id3 import ...`, `from .data_io import ...`, `import src...`) oraz układać moduły w ścisłej kolejności zależności (`data_io` → `preprocessing` → `id3` → `evaluation` → `experiments` → `visualization`), tak aby uruchomienie w czystym środowisku Google Colab (bez folderu `src/`) nie powodowało błędu `ModuleNotFoundError`.

---

## 2. Technologie i struktura projektu

- **Python:** 3.11+ (dobierz i zamroź zweryfikowane kompatybilne wersje zależności).
- **Notebooki:** JupyterLab/Jupyter Notebook i Google Colab (`.ipynb`).
- **Dane/obliczenia:** `pandas`, `numpy`, `scipy`.
- **Wykresy:** `matplotlib`, `seaborn`; interaktywne tylko wtedy, gdy mają realną wartość i działają także po eksporcie.
- **Drzewo:** `graphviz`/DOT → SVG lub PNG; zapewnij sprawny graficzny fallback w `matplotlib`, jeżeli systemowy renderer (`dot`) jest niedostępny. Finalnie muszą istnieć **czytelne wizualizacje drzewa**, nie wyłącznie kod do nich ani sam zrzut tekstowy ASCII.
- **Podziały i metryki:** `scikit-learn` może służyć do `train_test_split` i `metrics`, **ale nie do wytrenowania drzewa ID3**.
- **Testowanie i eksport:** `pytest`, `nbformat`, `nbconvert`/`nbclient`.

```text
ID3-Mushroom-Project/
├── RUNBOOK.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── mushroom.zip
├── notebooks/
│   └── ID3_Mushroom_Project.ipynb
├── src/
│   ├── id3.py
│   ├── data_io.py
│   ├── preprocessing.py
│   ├── experiments.py
│   ├── evaluation.py
│   └── visualization.py
├── scripts/
│   ├── build_notebook.py
│   ├── validate_notebook.py
│   └── package_submission.py
├── tests/
│   ├── test_entropy.py
│   ├── test_information_gain.py
│   ├── test_id3.py
│   ├── test_preprocessing.py
│   ├── test_experiments.py
│   └── test_group_exercise.py
├── group_exercise/
│   ├── README.md
│   ├── delivery_data.csv
│   ├── ID3_Delivery_Workshop.ipynb
│   └── instructor_private/
│       ├── ID3_Delivery_Solution.ipynb
│       └── ANSWER_KEY.md
├── outputs/
│   ├── figures/
│   ├── experiment_results.csv
│   ├── notebook.html
│   └── QA_REPORT.md
└── submission/
    └── ID3_Mushroom_Submission.zip
```

Nazwy folderów dopasuj, jeżeli wymaga tego poprawne działanie Colab, ale utrzymaj przejrzystą strukturę. **Nie umieszczaj `instructor_private/` w paczce dla grupy.** Archiwum oddawane prowadzącemu ma zawierać podstawowy notebook, dane i niezbędny kod uruchomieniowy; materiał zadania grupowego także dołącz, rozwiązanie instruktorskie tylko jeśli udostępnienie go prowadzącemu nie powoduje ujawnienia grupie.

---

## 3. Specyfikacja matematyczna i techniczna ID3

### 3.1. Teoria w notebooku

Pokaż i wyjaśnij na prostym przykładzie:

\[H(S)=-\sum_{i=1}^{k}p_i\log_2(p_i),\qquad 0\log_2(0):=0\]

\[IG(S,A)=H(S)-\sum_{v\in V(A)}\frac{\vert{}S_v\vert{}}{\vert{}S\vert{}}H(S_v)\]

Dla klasyfikacji dwuklasowej pokaż krzywą entropii względem `p` od 0 do 1, z zaznaczeniem `H(0)=0`, `H(0.5)=1`, `H(1)=0`. Opisz intuicję, logarytm o podstawie 2, pojęcie czystości liści, zachłanny wybór cechy i powstawanie gałęzi dla wszystkich jej wartości.

Dołącz **pseudokod** ID3 i przykład ręcznego wyznaczenia pierwszego rozgałęzienia na niewielkim podzbiorze (jeśli przykład pochodzi z Mushroom, wszystkie liczby muszą być rzeczywiście wyliczone). Krótko przedstaw ograniczenia ID3:
- preferencja wobec cech o wysokiej liczbie kategorii (wspomnij o *Split Information* i *Gain Ratio* wprowadzonym w C4.5 jako korekcie tego obciążenia),
- wrażliwość na szum i przeuczenie,
- klasyczna obsługa atrybutów wyłącznie kategorycznych (konieczność wcześniejszej dyskretyzacji zmiennych ciągłych),
- ograniczenia z brakami danych,
- złożoność obliczeniowa budowy drzewa względem liczby próbek $N$ i liczby cech $d$.

### 3.2. Własna implementacja

Zaimplementuj `ID3Classifier` z metodami `fit(X, y)`, `predict(X)`, `predict_one(...)`, `explain_prediction(...)` i właściwościami `node_count_`, `leaf_count_`, `depth_`, `feature_importance_` (ostatnia na podstawie ważonego sumarycznego przyrostu informacji, jawnie zdefiniowana).

Węzeł ma przechowywać co najmniej: nazwę testowanego atrybutu, dzieci według poszczególnych **kategorii**, rozkład etykiet węzła, klasę większościową (fallback), liczbę obserwacji, entropię, zysk informacji dla podziału oraz informację, czy jest liściem.

Algorytm musi:
1. zwracać liść, gdy obserwacje należą do jednej klasy;
2. zwracać liść większościowy, gdy brak użytecznych cech lub spełniony jest warunek stopu;
3. wybierać atrybut o maksymalnym Information Gain (wiążące remisy rozstrzygaj deterministycznie, np. alfabetycznie po nazwie cechy);
4. tworzyć wielogałęziowy podział na wszystkie faktycznie występujące wartości wybranego atrybutu;
5. nie używać ponownie wybranego atrybutu na niższych poziomach tej samej ścieżki;
6. obsługiwać puste gałęzie/nieznane kategorie w predykcji przez **klasę większościową zapamiętaną w ostatnim znanym węźle**, a nie przez wyjątek ani podgląd testu;
7. być deterministyczny przy ustalonej kolejności kolumn i próbek;
8. zwracać 0 dla entropii czystego zbioru; unikać `log2(0)` i błędów float;
9. nie przyjmować braków (`NaN`/`pd.NA`), o ile nie zostały wcześniej jawnie przekształcone w kategorię; walidować schemat danych;
10. umożliwiać interpretację ścieżki predykcyjnej.

**Punkt odniesienia:** oryginalny ID3 to drzewo dla atrybutów nominalnych. Parametry regularizacyjne opisane niżej są dydaktycznymi **rozszerzeniami** jego procedury wzrostu (*pre-pruning*), a nie „oryginalnymi hiperparametrami Quinlana”.

### 3.3. Warunki zatrzymania i parametry

- `max_depth`: `None` albo nieujemna/pozytywna granica zgodna z jawną konwencją (`root depth = 0`);
- `min_samples_split`: minimalna liczba rekordów w węźle konieczna do rozważania podziału;
- `min_information_gain`: minimalny próg poprawy przy podziale; przy zerowym przyroście informacji nie twórz bezwartościowych gałęzi (uwzględnij tolerancję numeryczną np. `1e-12`);
- obsługa remisów większości klas: deterministyczna i opisana (np. preferencja klasy `p` ze względów bezpieczeństwa lub porządek alfabetyczny — jawnie zdefiniowany i przetestowany).

Nie twórz automatycznie probabilistycznego ID3, przycinania CART ani regresji. Jeśli dodatkowo implementujesz post-pruning, wyraźnie oznacz jako rozszerzenie i nie czyń go wymaganym.

### 3.4. Zestawienie bibliotek (bez benchmarkingu)

Umieść krótką tabelę bibliotek z kolumnami: nazwa, środowisko/język, czy to ID3 czy inny algorytm drzewa, ograniczenia, źródło. **WEKA `weka.classifiers.trees.Id3`** jest prawdziwym przykładem implementacji ID3 dla danych nominalnych; dokumentacja: https://weka.sourceforge.io/doc.packages/simpleEducationalLearningSchemes/weka/classifiers/trees/Id3.html . Uwzględnij także pakiety Pythona/R (np. `chefboost`, `decision-tree-id3`, `RWeka`), precyzując ich status. Osobno wyjaśnij, dlaczego `scikit-learn DecisionTreeClassifier` nie jest implementacją ID3 (buduje wyłącznie drzewa binarne CART). **Nie trenuj ani nie porównuj z nimi modeli**, bo prowadzący tego nie wymaga.

---

## 4. Analiza jakości danych, czyszczenie i EDA

### 4.1. Audyt surowego zbioru

Wyświetl `head`, `shape`, `dtypes`, słownik kategorii, rozkład klas, unikalne wartości, cardinality, `value_counts`, `isna`, kontrolę zgodności z UCI. Ustal, czy występują rzeczywiste duplikaty, cechy stałe, nieznane kody, błędne typy oraz czy istnieją wartości spoza domeny znanej ze słownika. Nie twierdź, że „błędne dane” istnieją, jeżeli ich audyt nie znalazł; pokaż negatywny wynik kontroli jako rezultat.

Dla `stalk-root` zbadaj odsetek braków w całości (2480 / 8124 = 30.53%) i osobno względem klasy (`e` vs `p`). Braki muszą być policzone **po zamianie `?` na NaN**, a nie potraktowane jako zwyczajna, poprawna obserwacja. Wszystkie wnioski o braku danych mają być poparte liczbami.

### 4.2. Korelacje i zależności

Nie obliczaj bez uzasadnienia współczynnika Pearsona na arbitralnych integer-encodingach kodów cech. Dla cech kategorycznych użyj **V Craméra** lub innej dobrze wyjaśnionej miary powiązania (np. Theil's U / Mutual Information). Pokaż heatmapę ograniczoną do czytelnej liczby najbardziej informatywnych cech, zależności wybranych cech z klasą i rozkłady kategorii.

> **Uwaga numeryczna dla V Craméra:** dla cech stałych (w zbiorze Mushroom `veil-type` ma tylko 1 unikalną wartość `'p'` dla wszystkich 8124 wierszy) mianownik w klasycznym wzorze na V Craméra wynosi $\sqrt{\min(r-1, c-1)} = 0$. Funkcja obliczająca V Craméra musi jawnie obsługiwać przypadek $\min(r-1, c-1) == 0$ bez rzucania wyjątku `ZeroDivisionError` i bez psucia heatmapy wartościami `NaN` (zwracając `0.0` lub wykluczając cechę stałą przed budową macierzy z jawnym komentarzem).

### 4.3. Balansowanie klas

Pokaż rzeczywiste udziały `e/p` (51.80% `e` vs 48.20% `p`) i wniosek, czy balansowanie ma sens. Nie dokładaj SMOTE automatycznie. Uzasadnij świadome **niebalansowanie** jako poprawną decyzję metodyczną przy niemal równym rozkładzie klas.

### 4.4. Zbędne cechy i duplikaty

Sprawdź cechy stałe (`veil-type` jest cechą stałą o zerowej entropii i zerowym zysku informacji — należy ją usunąć i udokumentować w tabeli „przed/po”), potencjalnie redundantne i problem z pełnymi duplikatami. Usuwaj wyłącznie na podstawie jawnego kryriterium, najlepiej obliczonego na treningu; zachowaj surowy zbiór i tabelę „przed/po”. **Nie usuwaj cech tylko dlatego, że wizualnie mają małą korelację z etykietą mierzoną na całym zbiorze.**

---

## 5. Protokół podziału i preprocessingu

### 5.1. Podział danych

Docelowy podział: **60% train, 20% validation, 20% test** (dwa stratifikowane podziały), `random_state=42`. Udokumentuj dokładne liczebności i proporcje klas. Wszystkie trzy indeksy mają być rozłączne. Zachowaj je do audytu. Nie wolno oglądać metryk testowych w czasie doboru hiperparametrów.

> **Audyt duplikatów (wskazówka domenowa):** W surowym zbiorze Mushroom (przy pełnych 22 cechach) jest dokładnie 0 pełnych duplikatów wierszy. Jednak **po usunięciu kolumny `stalk-root` (`drop_stalk_root`) lub po imputacji braków dominantą (`most_frequent`) identyczne wektory cech mogą się pojawić!** Przeprowadź i opisz w notebooku audyt pokazujący liczbę duplikatów przed i po każdej ze strategii obsługi braków oraz oceń, czy identyczne wektory cech trafiają jednocześnie do treningu i testu (i czy mają spójne etykiety klas).

### 5.2. Trzy strategie braków danych (obowiązkowe)

- `missing_category`: `NaN → '__MISSING__'` jako jawna kategoria;
- `most_frequent`: imputacja dominantą **wyliczoną tylko na train**, potem `transform` na validation/test;
- `drop_stalk_root`: usunięcie kolumny `stalk-root` bez imputacji tego atrybutu.

Wykonaj trening każdego wariantu na tym samym podziale danych i tym samym protokole strojenia. Zapisz metryki walidacyjne oraz charakterystykę drzew. Po wybraniu zwycięskiej strategii dopasuj preprocessing na `train+validation` przed finalnym treningiem; test transformuj wyłącznie tym obiektem. Opisz obsługę niewidzianych wcześniej kategorii w trybie predykcji.

### 5.3. Kodowanie

Nie używaj one-hot ani ordinal encoding **dla samego ID3**; ID3 powinien pracować bezpośrednio na nazwanych kategoriach. Zachowaj czytelność nazw cech i wartości gałęzi.

---

## 6. Strojenie — eksperymenty kontrolowane

Przygotuj siatkę **36 konfiguracji**:

| Oś | Wartości |
|---|---|
| Strategia braków | `missing_category`, `most_frequent`, `drop_stalk_root` |
| `max_depth` | `2`, `4`, `None` |
| `min_samples_split` | `2`, `20` |
| `min_information_gain` | `0.0`, `0.01` |

Otrzymujemy `3 × 3 × 2 × 2 = 36` modeli próbnych.
> **Uwaga domenowa dot. zbioru Mushroom:** Ponieważ zbiór Mushroom jest separowalny ze 100% dokładnością już przy głębokości 4–5 (dzięki bardzo silnym cechom `odor` i `spore-print-color`), użycie w siatce wartości `max_depth=2` gwarantuje uwidocznienie niedouczenia (*underfitting*), `max_depth=4` strefy przejściowej, a `None` pełnego nasycenia. Dodatkowo wygeneruj wykres krzywej walidacyjnej dla `max_depth` w pełnym zakresie `1, 2, 3, 4, 5, 6, None` (dla domyślnych pozostałych parametrów), aby na prezentacji pokazać dokładny moment osiągnięcia 100% skuteczności.

Dla każdej z 36 konfiguracji zapisz: identyfikator, parametry, train accuracy, validation accuracy, validation macro-F1, validation recall klasy `p`, liczbę błędów `p → e`, głębokość drzewa, liczbę węzłów/liści, czas uczenia oraz strategię przetwarzania. **Wyniki muszą pochodzić z rzeczywistych obliczeń.**

**Deterministyczna reguła wyboru modelu (leksykograficzny tie-breaking):**
Przed obejrzeniem wyników zdefiniuj i zakoduj jednoznaczną regułę sortowania konfiguracji na zbiorze walidacyjnym (rozstrzygającą częste na tym zbiorze remisy 100% accuracy):
1. minimalna liczba błędów krytycznych **`p → e`** (grzyb trujący fałszywie sklasyfikowany jako jadalny),
2. maksymalny wynik **`validation macro-F1`**,
3. minimalna liczba liści (**`leaf_count_`** — prostsze drzewo zgodnie z brzytwą Ockhama),
4. minimalna liczba wszystkich węzłów (**`node_count_`**),
5. minimalna rzeczywista głębokość drzewa (**`depth_`**),
6. preferencja strategii braków zachowującej pełną informację bez ślepego odrzucania kolumny (`missing_category` > `most_frequent` > `drop_stalk_root`) oraz mniejsze `max_depth` przy identycznym drzewie.

Wyjaśnij w notebooku działanie tej reguły i wykaż, które konfiguracje osiągnęły identyczne metryki. Nie deklaruj, że `p → e = 0` w tym zbiorze oznacza realne bezpieczeństwo grzybobrania.

Obowiązkowe wykresy: tabela rankingowa (top 10 + podsumowanie wszystkich 36), heatmapa/macierze walidacyjne dla parametrów, porównanie metod imputacji, zależność złożoności drzewa od wyników (krzywa głębokości `1..None`), wykres train-vs-validation oraz krótkie wnioski po każdym wykresie.

---

## 7. Budowa modelu finalnego i ewaluacja

1. Wybierz tylko na podstawie walidacji zwycięską konfigurację i strategię braków.
2. Zapisz w notebooku **dlaczego** została wybrana (metryki, złożoność i interpretacja).
3. Połącz `train+validation`, ponownie dopasuj preprocessing na połączonych danych i od podstaw wytrenuj finalne ID3.
4. **Jednorazowo** oceń model na odłożonym `test` (bez dalszego dostrajania).
5. Wypisz: accuracy, precision/recall/F1 dla każdej klasy, macro/weighted F1, confusion matrix z kolejnością klas jasno opisaną (`e`, `p`), liczbę i udział błędów `p → e`, liczbę liści, liczbę węzłów, głębokość oraz pełną listę reguł/ważności cech.
6. Pokaż 3–5 rzeczywistych przykładów predykcji ze zbioru testowego, w tym pełną ścieżkę decyzji (`explain_prediction`) z rozwiniętymi nazwami cech i kategorii oraz, jeżeli istnieją, przykład błędnej klasyfikacji. Jeśli model nie pomyli się na teście (co przy ID3 na Mushroom jest oczekiwane dla pełnego drzewa), pokaż ciekawe przypadki z głębszych gałęzi (np. dla `odor=none`) i nie twórz sztucznych błędów.
7. Zapisz artefakt finalnego drzewa w bezpiecznym formacie (JSON oraz DOT) oraz parametry procesu.

**Interpretacja bezpieczeństwa:** dataset jest edukacyjny, nie stanowi wystarczającego narzędzia do decyzji o spożyciu dzikich grzybów. Zamieść to ostrzeżenie w podsumowaniu i przy predykcjach.

---

## 8. Grafika, naukowe opisy i standard prezentacyjny

### 8.1. Notebook = raport + prezentacja

Każda większa sekcja w notebooku ma mieć sekwencję: **cel i teoria (Markdown) → obliczenia (kod) → wykres/tabela (output) → interpretacja (Markdown)**. Kod nie może zastępować objaśnień. Stosuj `#`, `##`, `###`, tabelki Markdown, LaTeX `\( ... \)` i `\[ ... \]`; nie wklejaj wzorów jako niewyraźnych obrazków. Wykresy tworzone kodem mają być widoczne pod komórkami; Markdown opisuje kontekst i wnioski.

**Formatowanie:** jasny, jednolity styl; granat/grafit jako kolor nagłówków, ograniczona akcentowa paleta. Spójne kolory klas na wszystkich wykresach (np. zielony/morski dla `e = edible`, czerwony/koralowy dla `p = poisonous`). Typografia czytelna z projektora, sensowne marginesy, tytuły, legendy i jednostki. Wartości liczbowe formatowane konsekwentnie. Unikaj „tęczowego dashboardu”, ogromnych wydruków `DataFrame` i wykresów bez interpretacji.

Nie dopuszczaj do sytuacji, w której wykres wygląda inaczej zależnie od losowego uruchomienia. Zapisz wyniki komórek w przekazywanym `.ipynb`. Przy eksporcie HTML dopilnuj osadzania obrazów.

### 8.2. Plan rozdziałów notebooka

| Rozdział | Treść i obowiązkowe obiekty prezentacyjne |
|---|---|
| **00. Okładka** | tytuł, temat, autor (Kamil Piejko), źródło danych, spis treści, mapa 8 wymagań prowadzącego |
| **01. ID3** | problem, historia, wzory (Entropia, IG, wzmianka o Gain Ratio), pseudokod, wykres entropii, przykład ręcznego obliczenia IG, tabela bibliotek ID3 |
| **02. Mushroom** | pochodzenie, słownik cech i kategorii, przykładowe wiersze, rozkład `e/p`, profil cech |
| **03. EDA i czyszczenie** | mapa i wykres braków (`stalk-root`), rozkład kategorii, heatmapa V Craméra, tabela audytu błędnych danych/duplikatów, usunięcie stałej cechy `veil-type` |
| **04. Dane modelowe** | odpowiedź na punkt 3 ze slajdu (kodowanie, dyskretyzacja, podział `X`/`y`), pipeline 60/20/20, stratyfikacja, 3 strategie braków, audyt duplikatów po transformacji |
| **05. Własny ID3** | funkcje matematyczne, mapowanie pseudokod–kod, przykład pierwszego węzła (`odor`), wyniki testów jednostkowych |
| **06. Strojenie** | 36 modeli, krzywa głębokości, tabela parametrów, wykresy walidacji, porównanie strategii braków, komentarz dot. lasów losowych, wybór konfiguracji |
| **07. Model finalny** | ostateczna konfiguracja, trening na `train+val`, charakterystyka i ważność cech drzewa |
| **08. Wizualizacja drzewa** | wygenerowany diagram drzewa (SVG/PNG), czytelny widok na projektor, zestawienie bibliotek generujących grafikę |
| **09. Ocena i predykcja** | macierz pomyłek na zbiorze testowym, metryki klasyfikacji, analiza błędów `p → e`, 3–5 ścieżek predykcji |
| **10. Warsztat dla grupy** | wprowadzenie do zadania „Delivery Decision”, link/instrukcja dostępu, ramy 20 min, zasady pracy w domu; bez klucza odpowiedzi |
| **11. Podsumowanie i źródła** | wnioski końcowe, ograniczenia, reprodukowalność, bibliografia, licencja |

Przygotuj około **12–16 uzasadnionych wizualizacji**. Przewijany notebook ma umożliwiać wygłoszenie prezentacji bez oddzielnego PowerPointa.

### 8.3. Drzewo decyzyjne i zestawienie bibliotek graficznych

Wygeneruj **pełne drzewo do pliku** (SVG oraz PNG) oraz **czytelny widok prezentacyjny** w notebooku. Każdy węzeł powinien pokazywać atrybut, liczbę obserwacji, rozkład lub klasę dominującą i entropię; gałęzie — czytelne wartości kategorii.

> **Wymóg techniczny renderowania drzewa:**  
> Zapisz drzewo do formatu DOT (`graphviz`). Jeśli w systemie operacyjnym jest dostępny program `dot`, wyrenderuj z niego SVG/PNG. **Jeśli binarka systemowa `dot` nie jest zainstalowana w środowisku, `visualization.py` musi automatycznie wygenerować estetyczny, kolorowany diagram blokowy drzewa za pomocą `matplotlib` (prostokątne węzły z ramkami, kolorowaniem wg klasy/typu węzła i podpisanymi krawędziami kategorii) i zapisać go do PNG/SVG.** Sam tekstowy wydruk ASCII może być dodatkiem pomocniczym, ale **nie może** zastąpić graficznego rysunku drzewa w Rozdziale 08.

Dodaj tabelę porównawczą bibliotek generujących grafikę drzew decyzyjnych (np. `Graphviz / PyGraphviz / python-graphviz`, `Matplotlib` custom tree renderer, `NetworkX + Matplotlib`, `Plotly / dtreeviz`, `scikit-learn plot_tree` z uwagą o ograniczeniu do CART).

---

## 9. Punkt 8 — samodzielny warsztat dla grupy

### 9.1. Parametry i dydaktyka

**Nazwa:** „Delivery Decision — czy przesyłka dotrze na czas?”  
**Czas na wykładzie:** około **20 min**.  
**Dla kogo:** uczestnicy z laptopem wykonują na miejscu; pozostali pobierają/otwierają materiały i rozwiązują **samodzielnie w domu**, bez limitu czasu.  
**Środowisko:** Google Colab jako ścieżka główna; ten sam `.ipynb` lokalnie w JupyterLab.  
**Dane:** osobny, niewielki, **syntetyczny** CSV (`delivery_data.csv`), 24–32 wiersze, 4 nominalne cechy i binarna etykieta `on_time`/`delayed`.

Przykładowe cechy: `distance` (`short`, `medium`, `long`), `weather` (`clear`, `rain`, `snow`), `workload` (`low`, `medium`, `high`), `shipping` (`standard`, `express`). Zaprojektuj rzeczywiste zależności klas, 2–4 kontrolowane braki w dwóch kolumnach, powtarzalny rezultat Information Gain oraz przypadek testowy do predykcji. Dane nie mogą być identyczne z Mushroom ani stanowić jego przemianowanego fragmentu.

Nie zakładaj z góry, jaka cecha zostanie korzeniem: **wylicz to po wygenerowaniu CSV** i sprawdź, że zadanie ma jednoznaczne, dydaktycznie użyteczne rozwiązanie (lub jawnie omów remis). Unikaj rozwiązania, w którym wszystkie podziały są trywialne.

### 9.2. Scenariusz 20 minut

| Minuty | Uczestnik wykonuje | Weryfikowalny wynik |
|---|---|---|
| 0–4 | Otwiera notebook, czyta opis i dane | Załadowana tabelka i etykieta celu |
| 4–9 | Wykrywa i uzupełnia braki | Tabela `missing before/after` |
| 9–16 | Oblicza entropię i zyski informacji, wybiera korzeń; używa gotowego mini-ID3 | Ranking IG + mini-drzewo |
| 16–20 | Prognozuje jedną nową przesyłkę i uzasadnia ścieżkę | Predykcja + 1–2 zdania wyjaśnienia |

W notebooku uczestnika przygotuj 3–4 jasno oznaczone komórki `TODO`, automatyczne wskazówki/testy kontrolne (`assert` z przyjaznym komunikatem) oraz działające funkcje pomocnicze. **Nie każ uczestnikom pisać całego algorytmu od zera w 20 minut.**

**Bonus do domu:** porównaj `most_frequent` z osobną kategorią braków, zobacz, czy wybór korzenia się zmienił i dlaczego. Bonus musi być opcjonalny; podstawowy scenariusz nie może od niego zależeć.

### 9.3. Samowystarczalność i udostępnianie

Workshop `.ipynb` ma działać samodzielnie **bez montowania cudzego Google Drive i bez dostępu do repozytorium głównego**. Umieść CSV jako czytelny, niewielki tekst w komórce wejściowej `io.StringIO`, wygenerowany automatycznie z kanonicznego `delivery_data.csv` podczas budowy notebooka. Dostarcz również osobny CSV w paczce.

Przygotuj dwie paczki:
- **dla uczestników (`group_exercise/` poza `instructor_private/`):** notebook warsztatowy, CSV, krótki README z instrukcją udostępnienia; **bez klucza odpowiedzi**;
- **prywatna instruktorska (`group_exercise/instructor_private/`):** rozwiązany notebook, `ANSWER_KEY.md`, oczekiwane wyniki liczbowe, checklista i sugestie dla osób pracujących w domu.

Jeśli agent nie ma autoryzowanego dostępu do chmury grupy, **wygeneruj pliki lokalnie i oddaj dokładną instrukcję publikacji z ograniczonym dostępem (Google Drive / Colab tylko dla adresów e-mail grupy lub prywatne repozytorium GitHub)**, nie twierdząc, że pliki zostały już opublikowane w sieci.

Przygotuj osobny test sprawdzający, że notebook uczestnika uruchamia się i nie zawiera `ANSWER_KEY`, gotowych odpowiedzi w komentarzach ani klucza w ukrytych komórkach.

---

## 10. Testowanie i audyt jakości

### 10.1. Testy matematyczne

- `H([])` — jawnie zdefiniowane zachowanie (np. `0.0` lub `ValueError`, spójnie udokumentowane);
- `H(['e', 'e', 'e']) == 0.0`;
- `H(['e', 'p']) == 1.0`;
- entropia zawsze nieujemna w tolerancji numerycznej;
- IG podziału idealnego równe entropii rodzica;
- IG nieujemny w tolerancji numerycznej i nie większy niż entropia rodzica;
- rozstrzyganie remisów deterministyczne;
- jawny audyt logarytmu przy zerowych prawdopodobieństwach.

### 10.2. Testy modelu

- mieszanka klas → poprawny korzeń na ręcznie sprawdzonym przykładzie;
- czysty zbiór → liść;
- wyczerpanie cech → klasa większościowa;
- nieznana kategoria testowa → fallback zapisany w węźle;
- brak danych (`NaN`) bez wcześniejszego preprocessingu → czytelny wyjątek walidacji (`ValueError`);
- maksymalna głębokość, minimalna liczność, minimalny IG → prawidłowe warunki stopu;
- działanie dla pojedynczego rekordu i paczki;
- wyniki `predict` mają prawidłowy format i etykiety;
- powtarzalność predykcji i metryk;
- brak odczytu testowych etykiet w trakcie wyboru modelu.

### 10.3. Testy pipeline'u

- loader poprawnie otwiera oficjalny ZIP bez nagłówka i mapuje 23 kolumny;
- `?` zamieniane są na braki i liczone zgodnie z dokumentacją (2480 w `stalk-root`);
- cecha stała `veil-type` zostaje wykryta i usunięta bez psucia V Craméra;
- preprocessing jest dopasowany do treningu; nie zna testu przy `fit`;
- indeksy części podziału (train/val/test) są rozłączne;
- po transformacji nie ma braków ani niespójności schematu w ID3;
- wykresy powstają i mają niepuste rozmiary plików;
- 36 konfiguracji daje kompletny, odtwarzalny raport eksperymentów;
- predykcje i confusion matrix są zgodne z tym samym modelem finalnym;
- notebook zachowuje poprawne metadane i uruchamia się bez błędu od początku do końca (także w teście symulującym brak folderu `src/`);
- warsztat uczestnika nie zawiera klucza odpowiedzi.

### 10.4. Polecenia kontroli

Przygotuj dokumentację i działające skrypty:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/build_notebook.py
python scripts/validate_notebook.py
python scripts/package_submission.py
```

`validate_notebook.py` musi wykonać notebook mechanizmem `nbclient`/`nbconvert --execute`, sprawdzić brak błędów komórek, obecność głównych wykresów i zapisanych outputów oraz zarejestrować wynik w `outputs/QA_REPORT.md`.

---

## 11. Dostarczenie i prezentacja

### 11.1. Do oddania prowadzącemu (warunek konieczny ze slajdu)

Ze slajdu wynika: **Rmarkdown albo Jupyter; dane i plik `.Rmd` albo `.ipynb` spakować ZIP; przesłać na email co najmniej jeden dzień wcześniej (warunek konieczny)**.

Archiwum `submission/ID3_Mushroom_Submission.zip` ma zawierać:
- główny `ID3_Mushroom_Project.ipynb` **z wykonanymi i zapisanymi outputami oraz osadzonymi wykresami**;
- `data/mushroom.zip` z jasną informacją o źródle UCI;
- kod źródłowy `src/`, `scripts/`, `tests/`, `requirements.txt` i `README.md`;
- materiały zadania dla grupy (`group_exercise/` bez `instructor_private/`, albo z wyraźnie oddzielonym opisem — natomiast samą paczkę dla studentów wygeneruj również osobno jako `submission/ID3_Group_Workshop_Students.zip`, aby łatwo wrzucić ją na dysk dla grupy).

> **Ograniczenie rozmiaru i czystości ZIP (`scripts/package_submission.py`):**  
> Skrypt pakujący musi bezwzględnie wykluczyć `.venv/`, `__pycache__/`, `.pytest_cache/`, `.git/` oraz pliki tymczasowe. Rozmiar wynikowego archiwum `ID3_Mushroom_Submission.zip` **nie może przekraczać 15 MB** (aby bez problemu przeszedł przez filtry załączników poczty uczelnianej).

### 11.2. Pliki dodatkowe

- `outputs/notebook.html`: statyczny eksport wykonanego notebooka jako zabezpieczenie prezentacyjne;
- `outputs/figures/`: wszystkie wykresy w PNG/SVG, w tym czytelny rysunek drzewa;
- `outputs/experiment_results.csv`: pełne rzeczywiste wyniki 36 modeli;
- `outputs/QA_REPORT.md`: automatyczny raport wykonania testów, walidacji notebooka i rozmiaru paczki ZIP.

### 11.3. Scenariusz prezentacji

Narracja: **problem → czemu ID3 → wzory i intuicja → Mushroom i braki → przygotowanie danych (`???` ze slajdu) → własne ID3 → eksperymenty i strojenie → drzewo finalne → ewaluacja → zadanie dla grupy → wnioski**. Przy każdym wykresie umieść w komórce Markdown 2–4 zdania zwięzłej interpretacji opartej wyłącznie na rzeczywistych wynikach obliczeń.

---

## 12. Sekwencja realizacji (obowiązkowe etapy agenta)

| Etap | Implementacja | Warunek ukończenia |
|---|---|---|
| **M0 — Bootstrap** | Repo, zależności, loader ZIP, definicja danych | 8124×23, 4208 `e` / 3916 `p`, 2480 `?` zweryfikowane |
| **M1 — Core ID3** | Entropia, IG, rekurencja, predykcja, ścieżka | Testy matematyczne i jednostkowe przechodzą |
| **M2 — Pipeline** | EDA, V Craméra, audyt duplikatów, split 60/20/20, 3 strategie braków | Testy wycieku i preprocessing przechodzą |
| **M3 — Experiments** | 36 modeli + krzywa głębokości, wybór wg walidacji, raport | Wyniki pełne i odtwarzalne; test nieużyty |
| **M4 — Final** | Refit `train+val`, jeden test, graficzny diagram drzewa | Metryki + grafika (DOT/Matplotlib) + ścieżki predykcji |
| **M5 — Notebook** | 12 rozdziałów, samowystarczalny kod (Colab-ready), LaTeX, wykresy | Notebook wykonuje się w całości bez `src/` |
| **M6 — Workshop** | Syntetyczne CSV, notebook 20 min, klucz instruktorski | Samodzielny notebook bez rozwiązań; testy przechodzą |
| **M7 — QA/Delivery** | Eksport HTML, paczki ZIP (<15 MB), audyt 8 wymagań | `QA_REPORT.md` + gotowe archiwa bez błędów |

Każdy etap kończ testem i zapisem rzeczywistego stanu. Kolejne etapy nie mogą opierać się na niezweryfikowanych wynikach poprzedniego etapu.

---

## 13. Definition of Done — bez negocjacji

**Projekt jest ukończony wyłącznie, gdy:**

- [ ] Notebook zawiera wszystkie osiem punktów ze slajdu (w tym jawne rozwinięcie pytań `???` z pkt 3 i 4) i spójną narrację po polsku.
- [ ] Dane Mushroom pochodzą z UCI; walidacja schematu (8124×23) i braków (2480 w `stalk-root`) jest wykonana.
- [ ] ID3 jest implementacją własną, nominalną, wielogałęziową; testy matematyczne przechodzą.
- [ ] EDA obejmuje rozkład klas, braki, duplikaty (przed i po imputacji), błędne wartości, V Craméra (odporne na stałą cechę `veil-type`) i uzasadnienie usunięcia `veil-type`.
- [ ] Podział train/validation/test (60/20/20) jest odtwarzalny i rozłączny; imputacja nie widzi walidacji/testu podczas strojenia.
- [ ] Wszystkie 3 strategie braków i wszystkie 36 modeli próbnych zostały wykonane, a reguła rozstrzygania remisów jest jawna.
- [ ] Finalny model został wybrany na walidacji, przetrenowany na `train+val` i **tylko raz** oceniony na odłożonym teście.
- [ ] Drzewo zostało wyeksportowane jako prawdziwa grafika (SVG/PNG przez Graphviz lub estetyczny diagram blokowy Matplotlib), a nie tylko tekst ASCII; są tabele bibliotek ID3 i grafiki.
- [ ] Macierz pomyłek, komplet metryk i konkretne ścieżki predykcji zawierają interpretację.
- [ ] Wykresy mają tytuły, opis osi/legendę i podpisy; nigdzie nie są wpisane ręcznie fałszywe wyniki.
- [ ] Notebook `.ipynb` jest samowystarczalny (Colab-ready, bez wewnętrznych importów `from src...`), uruchamia się od początku do końca i **ma zapisane outputy**.
- [ ] Gotowe są dwa oddzielne pakiety warsztatu (uczestnik/instruktor); notebook uczestnika jest samodzielny w Colab i nie zawiera klucza.
- [ ] Archiwum `submission/ID3_Mushroom_Submission.zip` waży poniżej 15 MB, zawiera dane oraz wykonany notebook i jest gotowe do wysyłki e-mailem minimum dzień przed zajęciami.
- [ ] Wszystkie testy `pytest` oraz `validate_notebook.py` przechodzą w 100%, a podsumowanie znajduje się w `outputs/QA_REPORT.md`.

## 14. Źródła do cytowania w notebooku

- UCI Machine Learning Repository, **Mushroom**, https://archive.ics.uci.edu/dataset/73/mushroom , DOI: https://doi.org/10.24432/C5959T .
- R. Quinlan, **Induction of Decision Trees**, *Machine Learning* 1, 81–106 (1986), DOI: https://doi.org/10.1007/BF00116251 .
- R. Quinlan, **C4.5: Programs for Machine Learning**, Morgan Kaufmann (1993).
- WEKA, `weka.classifiers.trees.Id3`, https://weka.sourceforge.io/doc.packages/simpleEducationalLearningSchemes/weka/classifiers/trees/Id3.html .
- JupyterLab — format notebooków i eksport, https://jupyterlab.readthedocs.io/en/stable/user/export.html .
- Google Colab — praca z plikami zewnętrznymi, https://colab.research.google.com/notebooks/io.ipynb .

---

## 15. Instrukcja końcowa dla agenta

**Zrealizuj cały zakres tego runbooka.** Zacznij od M0, pracuj inkrementalnie, testuj po każdym etapie. Zadbaj, by `.ipynb` był **pełnym, wykonanym raportem dydaktycznym**, a nie tylko interfejsem do kodu z `src/`. W dokumentacji i w notebooku używaj poprawnego polskiego, precyzyjnych pojęć statystycznych i naukowych. Nie wypełniaj sekcji liczbami przykładowymi, gdy możliwe jest policzenie wyników.

Na końcu przedstaw krótkie podsumowanie: co zbudowano, co faktycznie przetestowano, gdzie są pliki końcowe, jakie wyniki uzyskano oraz czy istnieją jakiekolwiek otwarte ryzyka/blokery. **Nie stwierdzaj, że wynik lub test jest pomyślny, dopóki nie został uruchomiony i zweryfikowany.**
