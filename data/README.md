# Dane: UCI Mushroom

| | |
|---|---|
| Plik | `mushroom.zip` — **niezmieniona** kopia archiwum z UCI (nie rozpakowujemy go; notebook czyta je strumieniowo) |
| Źródło | UCI Machine Learning Repository, *Mushroom*: https://archive.ics.uci.edu/dataset/73/mushroom |
| Archiwum | https://archive.ics.uci.edu/static/public/73/mushroom.zip (pobrane 2026-10-16) |
| DOI | https://doi.org/10.24432/C5959T |
| Licencja | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| SHA-256 | `face32f32647e0d939f6233f36dd30dd5d619ae9f3f9b8e10bea4ac7e1f60b1a` |

**Cytowanie:** Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T

**Zawartość archiwum:** `agaricus-lepiota.data` (CSV bez nagłówka: klasa + 22 atrybuty nominalne),
`agaricus-lepiota.names` (opis i słownik kategorii), `Index`, `README`, `expanded.Z`.

**Kontrakt weryfikowany przez `src/data_io.py`** na surowym pliku: 8124 wiersze × 23 kolumny, klasy 4208 `e`
i 3916 `p`, 2480 znaków `?` wyłącznie w `stalk-root`. Słownik kategorii zapisany w kodzie jest porównywany z sekcją 7
pliku `.names`.

Dane opisują hipotetyczne okazy 23 gatunków z przewodnika *The Audubon Society Field Guide to North American
Mushrooms* (1981). Zbiór jest edukacyjny i **nie może** służyć do oceny jadalności prawdziwych grzybów.
