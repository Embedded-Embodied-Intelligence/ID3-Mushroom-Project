# ID3 × Mushroom

Projekt akademicki: **własna implementacja klasycznego algorytmu ID3** (Quinlan, 1986) na zbiorze **Mushroom (UCI)**.
Główny artefakt to notebook `notebooks/ID3_Mushroom_Project.ipynb`, który jest jednocześnie projektem, sprawozdaniem
i prezentacją na zajęcia (12 rozdziałów, po polsku). Notebook pokrywa wszystkie osiem wymagań prowadzącego, a ich mapę
zawiera rozdział 00.

> ⚠️ Mushroom to zbiór edukacyjny (hipotetyczne okazy 23 gatunków z jednego przewodnika). Wyniki modelu **nie mogą**
> służyć do oceny jadalności prawdziwych grzybów.

## Najważniejsze wyniki

| | |
|---|---|
| Implementacja | własne `ID3Classifier`: podziały wielogałęziowe, entropia/IG, deterministyczne remisy, fallback dla nieznanych kategorii, `explain_prediction`; **bez** `sklearn.tree` |
| Dane | kontrakt UCI zweryfikowany (8124 × 23, 4208 `e` / 3916 `p`, 2480 braków w `stalk-root`); brak duplikatów i błędnych wartości; usunięta stała `veil-type` |
| Strojenie | 36 konfiguracji na walidacji: jakość zależy wyłącznie od `max_depth`; 3 strategie braków dają identyczne drzewa, bo `stalk-root` nigdzie nie wygrywa; wybrana konfiguracja `C08` |
| Model finalny | trening na train+val (6499 obs.): 29 węzłów, 24 liście, głębokość 4, 5 atrybutów (`odor`, `spore-print-color`, `habitat`, `gill-size`, `cap-color`) |
| Test (jednokrotnie, 1625 obs.) | accuracy 1.0000, macro-F1 1.0000, 0 błędów krytycznych `p → e` |

Wszystkie liczby cytowane w komentarzach notebooka są wstawiane z rzeczywistego uruchomienia i sprawdzane w jego
ostatniej komórce. Raport kontroli jakości: `outputs/QA_REPORT.md`.

## Struktura

```text
├── README.md, RUNBOOK.md, requirements.txt, pytest.ini, .gitignore
├── data/mushroom.zip                 # oryginalne archiwum UCI (+ data/README.md: źródło, licencja, SHA-256)
├── notebooks/ID3_Mushroom_Project.ipynb   # wykonany notebook z zapisanymi wynikami
├── src/                              # jedyna implementacja kodu (wstrzykiwana do notebooka)
│   ├── data_io.py  preprocessing.py  id3.py  evaluation.py  experiments.py  visualization.py
├── scripts/
│   ├── build_notebook.py             # generuje notebooki z src/ + treści (notebook_main.py, notebook_workshop.py)
│   ├── validate_notebook.py          # wykonuje i waliduje notebooki, eksport HTML, QA_REPORT
│   ├── package_submission.py         # paczki ZIP
│   ├── make_workshop_data.py         # generator danych warsztatu
│   └── qa_report.py
├── tests/                            # pytest: matematyka, ID3, pipeline, eksperymenty, wykresy, notebook, warsztat, paczki
├── group_exercise/                   # warsztat „Delivery Decision”: notebook uczestnika, CSV, README
│   └── instructor_private/           # rozwiązanie + ANSWER_KEY.md (NIE dla grupy)
├── outputs/                          # figures/, experiment_results.csv, notebook.html, QA_REPORT.md, final_model/ …
└── submission/                       # archiwa ZIP do wysyłki
```

## Uruchomienie lokalne (JupyterLab)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q                      # testy (zapisują wynik w outputs/QA_REPORT.md)
python scripts/build_notebook.py         # generuje notebooki z src/ (ok. 2 min: przebieg roboczy + narracja)
python scripts/validate_notebook.py      # wykonuje notebooki, sprawdza wykresy, HTML, uruchomienie bez src/
python scripts/package_submission.py     # submission/*.zip (< 15 MB)
jupyter lab notebooks/ID3_Mushroom_Project.ipynb
```

Dane są czytane z `data/mushroom.zip` bez rozpakowywania i bez dostępu do internetu.

## Google Colab

1. *Plik → Prześlij notebook* → `notebooks/ID3_Mushroom_Project.ipynb`.
2. *Uruchom wszystko.* Notebook nie potrzebuje folderu `src/`, bo kod modułów jest w komórkach. Jeśli
   `data/mushroom.zip` nie istnieje, pobiera oryginalne archiwum z UCI. Bez internetu pokazuje instrukcję ręcznego
   wgrania pliku (`files.upload()`).
3. W Colab jest Graphviz, więc pełne drzewo renderuje program `dot`. Lokalnie bez Graphviza ten sam schemat rysuje
   fallback w matplotlib.

## Synchronizacja `src/` ↔ notebook

Kod istnieje tylko w `src/`. `scripts/build_notebook.py`:

- kopiuje moduły do komórek w kolejności zależności `data_io → preprocessing → id3 → evaluation → experiments →
  visualization`;
- usuwa importy wewnątrzpakietowe (`from .id3 import …`, `from src… import …`; analiza `ast`, również
  wielowierszowa);
- sprawdza, że każda usunięta nazwa jest zdefiniowana we wcześniejszym module i że moduły nie nadpisują nawzajem
  swoich nazw.

Test `tests/test_notebook.py` porównuje komórki z bieżącym `src/` i uruchamia notebook w katalogu bez `src/` i `tests/`.
Po każdej zmianie w `src/` uruchom ponownie `build_notebook.py` i `validate_notebook.py`.

## Warsztat dla grupy

`group_exercise/` zawiera notebook uczestnika (20 minut na zajęciach, bez limitu w domu), dane i README z instrukcją
udostępnienia **tylko osobom z grupy** (Google Drive z dostępem ograniczonym lub prywatne repozytorium).
**Materiały nie zostały opublikowane automatycznie.** Publikację wykonuje autor według `group_exercise/README.md`.
Rozwiązanie i klucz odpowiedzi są wyłącznie w `group_exercise/instructor_private/` i w osobnej paczce
`submission/ID3_Workshop_Instructor_PRIVATE.zip`.

## Paczki do oddania

| Archiwum | Dla kogo | Zawartość |
|---|---|---|
| `submission/ID3_Mushroom_Submission.zip` | prowadzący (mail ≥ 1 dzień przed zajęciami) | wykonany notebook, dane, `src/`, `scripts/`, `tests/`, `outputs/`, README, materiały uczestnika warsztatu |
| `submission/ID3_Group_Workshop_Students.zip` | grupa | notebook uczestnika, CSV, README |
| `submission/ID3_Workshop_Instructor_PRIVATE.zip` | tylko autor | rozwiązanie, `ANSWER_KEY.md` |

## Rozwiązywanie problemów

- **Projekt na Biurku lub w Dokumentach synchronizowanych z iCloud (macOS):** przy zapełnionym dysku iCloud może
  przenosić pliki środowiska `.venv` do chmury (`dataless`). Importy potrafią wtedy trwać wiele minut. Utwórz
  środowisko poza iCloud i podlinkuj je:
  `python -m venv ~/.venvs/id3-mushroom && ln -s ~/.venvs/id3-mushroom .venv`. Inna możliwość to katalog
  `.venv.nosync`, którego iCloud nie synchronizuje.
- **Graphviz (opcjonalnie):** `brew install graphviz` (macOS) lub `apt-get install graphviz` (Linux) włącza
  renderowanie programem `dot`. Bez niego działa fallback w matplotlib.

## Dane i licencja

Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T, licencja
**CC BY 4.0**. Szczegóły: `data/README.md`.
