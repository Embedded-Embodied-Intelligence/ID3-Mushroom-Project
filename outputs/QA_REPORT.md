# QA_REPORT — ID3 × Mushroom

Raport generowany automatycznie: sekcję testów zapisuje `tests/conftest.py` (pełne uruchomienie `python -m pytest -q`), sekcje notebooka i warsztatu — `scripts/validate_notebook.py`, sekcję paczek ZIP — `scripts/package_submission.py`. Każda sekcja ma własny znacznik czasu.

<!-- BEGIN environment -->
## Środowisko

_Aktualizacja: 2026-10-16 17:40:13 CEST_

- Python 3.12.13 (/Users/kamel/.venvs/id3-mushroom/bin/python) na macOS-26.6.2
- Pakiety: `numpy` 2.5.4, `pandas` 3.0.6, `scipy` 1.18.1, `scikit-learn` 1.9.1, `matplotlib` 3.11.2, `seaborn` 0.13.2, `nbformat` 5.11.1, `nbclient` 0.11.0, `nbconvert` 7.17.2, `ipykernel` 7.4.0, `pytest` 9.1.1
- Graphviz `dot`: niedostępny → drzewo renderowane fallbackiem matplotlib
<!-- END environment -->

<!-- BEGIN pytest -->
## Testy automatyczne (pytest)

_Aktualizacja: 2026-10-16 17:42:10 CEST_

- Status: **PASS** (exit code 0)
- Wynik: 114 passed, 0 failed, 0 errors, 1 skipped
- Czas: 61.2 s; Python 3.12.13; pandas 3.0.6

| plik testów | passed | failed |
|---|---:|---:|
| `tests/test_entropy.py` | 13 | 0 |
| `tests/test_experiments.py` | 18 | 0 |
| `tests/test_group_exercise.py` | 7 | 0 |
| `tests/test_id3.py` | 28 | 0 |
| `tests/test_information_gain.py` | 8 | 0 |
| `tests/test_notebook.py` | 7 | 0 |
| `tests/test_packaging.py` | 4 | 0 |
| `tests/test_preprocessing.py` | 19 | 0 |
| `tests/test_visualization.py` | 10 | 0 |
<!-- END pytest -->

<!-- BEGIN notebook -->
## Walidacja notebooka głównego

_Aktualizacja: 2026-10-16 17:41:00 CEST_

- Status: **PASS** (16/16)

| kontrola | wynik | szczegóły |
|---|---|---|
| schemat nbformat 4 | ✅ |  |
| kernelspec python3 | ✅ |  |
| 12 rozdziałów (00–11) w kolejności | ✅ |  |
| brak importów z src/ (Colab-ready) | ✅ |  |
| moduły wstrzyknięte w kolejności zależności i zgodne z src/ | ✅ | data_io → preprocessing → id3 → evaluation → experiments → visualization |
| narracja bez niewypełnionych wstawek | ✅ |  |
| kontrola spójności narracji osadzona | ✅ |  |
| wykonanie od początku do końca (nbclient) | ✅ | 25 s |
| brak błędów w komórkach | ✅ | błędy: 0 |
| osadzone wykresy (≥ 12) | ✅ | obrazy: 18 |
| zapisane outputy komórek kodu | ✅ | komórki kodu bez outputu: 0 z 50 |
| pliki wykresów (PNG/SVG/DOT) niepuste | ✅ | 22/22 |
| artefakty (CSV, JSON, DOT) | ✅ | experiment_results.csv, split_indices.csv, test_predictions.csv, run_summary.json, narrative_facts.json, final_model/final_tree.json, final_model/final_tree.dot, final_model/final_config.json |
| experiment_results.csv: 36 modeli | ✅ | wiersze: 36 |
| eksport HTML z osadzonymi obrazami | ✅ | outputs/notebook.html (3.4 MB) |
| uruchomienie izolowane bez src/ i tests/ (symulacja Colab) | ✅ | 21 s |
<!-- END notebook -->

<!-- BEGIN colab_compat -->
## Zgodność ze starszymi wersjami bibliotek (symulacja Google Colab)

_Aktualizacja: 2026-10-16 17:22:22 CEST_

- Status: **PASS**
- Środowisko testowe (osobny venv poza projektem, Python 3.12.13): numpy 2.0.2, pandas 2.2.2, scipy 1.14.1,
  scikit-learn 1.6.1, matplotlib 3.10.0, seaborn 0.13.2 — wersje zbliżone do preinstalowanych w Google Colab
  (bez instalowania czegokolwiek z requirements.txt).
- Notebook skopiowany wyłącznie z `data/mushroom.zip` (bez `src/` i `tests/`) wykonał się od początku do końca
  w 29.6 s, bez wyjątków i bez komunikatów na stderr.
- Kontrola spójności narracji w ostatniej komórce: **149 / 149** wartości identycznych jak przy pandas 3.0 / numpy 2.5
  (wyniki nie zależą od wersji bibliotek).
- Znaleziony i usunięty problem: formatowanie tabel przez `DataFrame.style` wymagało opcjonalnego `jinja2`;
  zastąpione zaokrąglaniem (`DataFrame.round`), więc notebook nie zależy od pakietów opcjonalnych.
- Uwaga: to symulacja wersji bibliotek, nie uruchomienie w samym serwisie Colab (brak dostępu do konta Google w tej sesji).
<!-- END colab_compat -->

<!-- BEGIN workshop -->
## Warsztat dla grupy

_Aktualizacja: 2026-10-16 17:41:03 CEST_

- Status: **PASS** (8/8)

| kontrola | wynik | szczegóły |
|---|---|---|
| uczestnik: brak klucza / gotowych odpowiedzi | ✅ | [] |
| uczestnik: 3–4 komórki TODO | ✅ | komórki z TODO: 4 |
| uczestnik: zapisany bez outputów | ✅ |  |
| uczestnik: dane wbudowane = delivery_data.csv | ✅ |  |
| uczestnik: uruchamia się bez błędów (niewypełnione TODO) | ✅ | 1 s |
| rozwiązanie: wykonane z zapisanymi wynikami | ✅ | 1 s |
| rozwiązanie: wszystkie 4 kontrolki zaliczone | ✅ | 4/4 |
| ANSWER_KEY.md wygenerowany | ✅ |  |
<!-- END workshop -->

<!-- BEGIN requirements_map -->
## Audyt 8 wymagań prowadzącego

_Aktualizacja: 2026-10-16 17:41:03 CEST_

| Nr | Wymóg | Dowód |
|---|---|---|
| 1 | Wprowadzenie, pseudokod, biblioteki z ID3 | rozdz. 01 (wzory, pseudokod, przykład ręczny, tabela 7 bibliotek) |
| 2 | Dane, EDA, balans, korelacje, braki, błędne dane, czyszczenie, uzupełnianie, zbędne atrybuty | rozdz. 02–03: kontrakt UCI, audyt jakości (wyniki negatywne), braki + χ², V Craméra, decyzja o niebalansowaniu, usunięcie `veil-type` |
| 3 | Przygotowanie danych, `???`, train/val/test | rozdz. 04: X/y, kodowanie i dyskretyzacja, 60/20/20 stratyfikowane, 3 strategie braków, audyt duplikatów, `split_indices.csv` |
| 4 | Modele próbne, strojenie, dobór parametrów, uwaga o lasach | rozdz. 06: 36 modeli, krzywa głębokości, reguła wyboru, akapit o lasach/OOB, `experiment_results.csv` |
| 5 | Model finalny | rozdz. 07: konfiguracja C08, refit train+val, `final_tree.json` |
| 6 | Grafika drzewa + biblioteki grafiki | rozdz. 08: diagramy PNG/SVG + DOT, tabela 7 bibliotek; wykresów w outputs/figures: 22 |
| 7 | Ocena, macierz pomyłek, predykcje | rozdz. 09: accuracy 1.0, p→e 0, 5 ścieżek predykcji |
| 8 | Zadanie dla grupy, dostęp tylko dla grupy | rozdz. 10 + `group_exercise/` (uczestnik) + `instructor_private/` (klucz); instrukcja publikacji w README — publikacja do wykonania przez autora |
<!-- END requirements_map -->

<!-- BEGIN packages -->
## Paczki ZIP

_Aktualizacja: 2026-10-16 17:42:52 CEST_

- Status: **PASS**
- Klucz odpowiedzi w paczce prowadzącego: nie (domyślnie)
- Rozmiary zmierzone przed dołączeniem tego raportu do archiwum; archiwum jest potem budowane ponownie z kompletnym raportem i jeszcze raz sprawdzane (integralność, limit 15 MB).

| archiwum | rozmiar | pliki | kontrole |
|---|---:|---:|---|
| `submission/ID3_Mushroom_Submission.zip` | 7.42 MB | 68 | ✅ integralność ZIP; ✅ brak .venv/__pycache__/.pytest_cache/.git/plików tymczasowych; ✅ rozmiar < 15 MB; ✅ zawiera dane, notebook z outputami, src/scripts/tests; ✅ bez klucza odpowiedzi |
| `submission/ID3_Group_Workshop_Students.zip` | 0.02 MB | 3 | ✅ integralność ZIP; ✅ brak .venv/__pycache__/.pytest_cache/.git/plików tymczasowych; ✅ tylko notebook, CSV i README; bez klucza; ✅ notebook uczestnika bez outputów |
| `submission/ID3_Workshop_Instructor_PRIVATE.zip` | 0.04 MB | 2 | ✅ integralność ZIP; ✅ brak .venv/__pycache__/.pytest_cache/.git/plików tymczasowych; ✅ zawiera rozwiązanie i ANSWER_KEY |
<!-- END packages -->

<!-- BEGIN dod -->
## Definition of Done (podsumowanie automatyczne)

_Aktualizacja: 2026-10-16 17:42:52 CEST_

| Kryterium (RUNBOOK §13) | Status |
|---|---|
| Osiem punktów ze slajdu + rozwinięcie `???` (pkt 3 i 4), narracja po polsku | ✅ |
| Dane UCI: schemat 8124×23 i 2480 braków w `stalk-root` zweryfikowane | ✅ |
| Własny, nominalny, wielogałęziowy ID3; testy matematyczne | ✅ |
| EDA: klasy, braki, duplikaty przed/po, błędne dane, V Craméra odporne na `veil-type` | ✅ |
| Podział 60/20/20 odtwarzalny i rozłączny; imputacja dopasowana tylko na train | ✅ |
| 3 strategie braków, 36 modeli, jawna reguła remisów | ✅ |
| Model wybrany na walidacji, refit train+val, test użyty raz | ✅ |
| Drzewo jako grafika (SVG/PNG) + tabele bibliotek ID3 i grafiki | ✅ |
| Macierz pomyłek, metryki i ścieżki predykcji z interpretacją | ✅ |
| Notebook samowystarczalny (bez `src/`), wykonany, z zapisanymi outputami | ✅ |
| Dwa pakiety warsztatu (uczestnik bez klucza / instruktor) | ✅ |
| `ID3_Mushroom_Submission.zip` < 15 MB, z danymi i wykonanym notebookiem | ✅ |
| `pytest` i `validate_notebook.py` w 100% | ✅ |

Do wykonania ręcznie przez autora: publikacja materiałów warsztatu w miejscu dostępnym tylko dla grupy (instrukcja: `group_exercise/README.md`) i wysłanie ZIP-a mailem co najmniej dzień przed zajęciami.
<!-- END dod -->
