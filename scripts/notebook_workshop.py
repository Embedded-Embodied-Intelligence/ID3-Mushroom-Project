"""Workshop notebooks "Delivery Decision" (participant + instructor) and the answer key.

The participant notebook contains no answers: every check recomputes the expected result from the
data at run time and prints only "zaliczone" or a hint. The instructor variant fills in the TODOs and
solves the optional bonus. ``answer_key`` computes the numbers for ANSWER_KEY.md from the same code.
"""
from __future__ import annotations

import io
import textwrap

import pandas as pd

FEATURES = ["distance", "weather", "workload", "shipping"]
TARGET = "on_time"
NEW_PARCEL = {"distance": "long", "weather": "rain", "workload": "high", "shipping": "express"}


def md(text: str) -> dict:
    return {"type": "markdown", "source": textwrap.dedent(text).strip()}


def code(text: str) -> dict:
    return {"type": "code", "source": textwrap.dedent(text).strip()}


HELPERS = r'''
# --- Kontrolki zadań: porównują Twój wynik z obliczeniem wykonanym na bieżąco (bez zapisanych odpowiedzi) ---
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

FEATURES = ["distance", "weather", "workload", "shipping"]
TARGET = "on_time"
_STATUS: dict[str, bool] = {}


def todo_ok(key: str) -> bool:
    return _STATUS.get(key, False)


def _pending(key: str, hint: str) -> None:
    _STATUS[key] = False
    print(f"⏳ TODO {key} jeszcze nie jest gotowe. Wskazówka: {hint}")


def _passed(key: str, message: str) -> None:
    _STATUS[key] = True
    print(f"✅ TODO {key} zaliczone — {message}")


def _mode(series: pd.Series):
    counts = series.value_counts(dropna=True)
    return min(counts.index[counts == counts.max()])


def reference_fill(df: pd.DataFrame) -> pd.DataFrame:
    print("⚠️ Używam wersji pomocniczej uzupełniania braków, żeby można było iść dalej. Wróć do TODO 1 w domu.")
    return df.fillna({c: _mode(df[c]) for c in df.columns if df[c].isna().any()})


def reference_gains(df: pd.DataFrame) -> dict:
    return {f: information_gain(df[f], df[TARGET]) for f in FEATURES}


def missing_before_after(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"braki przed": before.isna().sum(), "braki po": after.isna().sum()})


def check_todo_1(df_original: pd.DataFrame, df_filled) -> None:
    if df_filled is None:
        return _pending("1", "dominantę kolumny zwraca df[kolumna].mode()[0]; braki uzupełnia df.fillna({kolumna: wartość}).")
    assert isinstance(df_filled, pd.DataFrame), "df_filled powinno być tabelą pandas (DataFrame)."
    assert df_filled.shape == df_original.shape, (
        f"Zmienił się kształt danych {df_original.shape} → {df_filled.shape}: nie usuwaj wierszy ani kolumn.")
    left = int(df_filled.isna().sum().sum())
    assert left == 0, f"W df_filled zostało {left} braków — sprawdź df_filled.isna().sum()."
    for column in df_original.columns:
        known = df_original[column].notna()
        assert (df_filled.loc[known, column] == df_original.loc[known, column]).all(), (
            f"W kolumnie '{column}' zmieniono wartości, które nie były brakami.")
        if (~known).any():
            inserted = sorted(set(df_filled.loc[~known, column]))
            assert inserted == [_mode(df_original[column])], (
                f"W kolumnie '{column}' brak należy zastąpić dominantą (najczęstszą wartością tej kolumny); "
                f"wstawiono: {inserted}.")
    _passed("1", "braki uzupełnione dominantami kolumn.")


def check_todo_2(my_entropy, labels: pd.Series) -> None:
    try:
        probe = my_entropy(["a", "b"])
    except Exception as error:  # the participant's function raised
        raise AssertionError(f"my_entropy zgłasza błąd: {error!r}") from error
    if probe is None:
        return _pending("2", "p = pd.Series(labels).value_counts(normalize=True), potem −Σ p·log2(p) (np.log2).")
    assert abs(float(probe) - 1.0) < 1e-9, "Dla dwóch równolicznych klas entropia musi wynosić dokładnie 1 bit."
    assert abs(float(my_entropy(["a", "a", "a"]))) < 1e-9, "Dla zbioru czystego (jedna klasa) entropia musi wynosić 0."
    for case in (["x", "y", "y", "z"], list(labels)):
        assert abs(float(my_entropy(case)) - entropy(case)) < 1e-9, (
            "Wynik nie zgadza się ze wzorem — użyj udziałów klas (a nie liczności) i logarytmu o podstawie 2.")
    _passed("2", "funkcja liczy entropię zgodnie ze wzorem.")


def check_todo_3(df: pd.DataFrame, ig_shipping, gains: dict, root_feature) -> None:
    if ig_shipping is None or not gains or root_feature is None:
        return _pending("3", "IG = H(S) − Σ |S_v|/|S|·H(S_v); grupy S_v daje df.groupby('shipping'); "
                             "potem gains = {cecha: information_gain(...)} i max(gains, key=gains.get).")
    reference = reference_gains(df)
    assert abs(float(ig_shipping) - reference["shipping"]) < 1e-9, (
        "IG(shipping) różni się od wartości ze wzoru — sprawdź wagi |S_v|/|S| i entropie gałęzi.")
    assert set(gains) == set(FEATURES), f"gains powinno zawierać wszystkie cechy: {FEATURES}."
    for feature in FEATURES:
        assert abs(float(gains[feature]) - reference[feature]) < 1e-9, f"IG cechy '{feature}' jest policzony błędnie."
    best = max(reference.values())
    assert root_feature in [f for f in FEATURES if reference[f] >= best - 1e-12], (
        "To nie jest cecha o największym przyroście informacji — porównaj wartości w gains.")
    _passed("3", "IG policzone poprawnie, korzeń wybrany zgodnie z kryterium ID3.")


def check_todo_4(model, parcel: dict, my_prediction, my_explanation: str) -> None:
    if my_prediction is None:
        return _pending("4", "mini_tree.predict_one(new_parcel) zwraca klasę; ścieżkę widać w wydruku drzewa.")
    assert my_prediction == model.predict_one(parcel), (
        "Predykcja nie zgadza się z drzewem — użyj mini_tree.predict_one(new_parcel).")
    path_features = {step.feature for step in model.explain_prediction(parcel).steps}
    text = (my_explanation or "").lower()
    assert len(text.strip()) >= 15, "Dopisz 1–2 zdania uzasadnienia w my_explanation."
    assert any(f in text for f in path_features), (
        "W uzasadnieniu wymień cechy, które drzewo sprawdziło po drodze do liścia (nazwy z wydruku drzewa).")
    _passed("4", "predykcja i uzasadnienie zgodne ze ścieżką w drzewie.")


def plot_gains(gains: dict, title: str = "Przyrost informacji IG w korzeniu") -> None:
    order = sorted(gains, key=lambda f: (-gains[f], f))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = ["#2a78d6" if f == order[-1] else "#d8d6cf" for f in order]
    ax.barh(order, [gains[f] for f in order], color=colors, height=0.55)
    for i, f in enumerate(order):
        ax.text(gains[f] + 0.005, i, f"{gains[f]:.3f}", va="center", fontsize=10)
    ax.set_xlabel("IG [bit]")
    ax.set_title(title, loc="left", fontweight="bold", color="#1d3557")
    ax.spines[["top", "right"]].set_visible(False)
    plt.show()
'''


def _helper_cell(id3_code: str) -> dict:
    header = ("# @title Kod pomocniczy: mini-ID3 i kontrolki zadań (uruchom; nie musisz go czytać)\n"
              "# Mini-ID3 to ten sam kod co w projekcie (src/id3.py), wstawiony automatycznie.\n\n")
    body = id3_code.split("\n", 2)[2] if id3_code.startswith("# @title") else id3_code
    return {"type": "code", "source": header + body.strip() + "\n\n" + HELPERS.strip(), "tags": ("helpers",),
            "metadata": {"cellView": "form", "jupyter": {"source_hidden": True}}}


def _data_cell(csv_text: str) -> dict:
    cell = code(r"""
        # Dane: 28 przesyłek (syntetyczne). Ta sama treść co plik delivery_data.csv dołączony do materiałów.
        DELIVERY_CSV = '''
        __CSV__
        '''
        df = pd.read_csv(io.StringIO(DELIVERY_CSV.strip()))
        print(f"{df.shape[0]} przesyłek × {df.shape[1]} kolumn; cechy: {FEATURES}; etykieta: '{TARGET}'")
        display(df.head(10))
        print("Rozkład etykiety:", df[TARGET].value_counts().to_dict())
        """)
    cell["source"] = cell["source"].replace("__CSV__", csv_text.strip())
    return cell


def cells(variant: str, csv_text: str, id3_code: str) -> list[dict]:
    if variant not in {"student", "solution"}:
        raise ValueError(variant)
    solution = variant == "solution"
    title = "Rozwiązanie prowadzącego" if solution else "Notebook uczestnika"
    out = [
        md(rf"""
        # Warsztat ID3: „Delivery Decision — czy przesyłka dotrze na czas?”
        **{title}** · ok. 20 minut na zajęciach · bez limitu czasu w domu

        Firma kurierska chce przewidywać, czy przesyłka dotrze na czas (`on_time`), czy się spóźni (`delayed`). Mamy
        28 historycznych przesyłek opisanych czterema cechami nominalnymi:

        | cecha | wartości |
        |---|---|
        | `distance` | `short`, `medium`, `long` |
        | `weather` | `clear`, `rain`, `snow` |
        | `workload` (obciążenie kuriera) | `low`, `medium`, `high` |
        | `shipping` | `standard`, `express` |

        **Jak zacząć (Google Colab):** *Plik → Zapisz kopię na Dysku*, potem *Środowisko wykonawcze → Uruchom wszystko*.
        Notebook jest samodzielny: dane i kod są w środku, niczego nie trzeba montować ani pobierać. Lokalnie działa
        w JupyterLab tak samo.

        | Minuty | Zadanie | Wynik do sprawdzenia |
        |---|---|---|
        | 0–4 | uruchom kod pomocniczy, obejrzyj dane | tabela danych i rozkład etykiety |
        | 4–9 | **TODO 1:** wykryj i uzupełnij braki | tabela „braki przed / po” |
        | 9–16 | **TODO 2–3:** entropia, przyrost informacji, wybór korzenia | ranking IG + mini-drzewo |
        | 16–20 | **TODO 4:** predykcja nowej przesyłki z uzasadnieniem | predykcja + 1–2 zdania |

        Po każdym TODO kontrolka wypisze **✅ zaliczone** albo **⏳ wskazówkę**. Niedokończone zadanie nie blokuje
        kolejnych, bo notebook podstawi wtedy wersję pomocniczą. Nie piszesz całego algorytmu: drzewo zbuduje gotowy
        mini-ID3.

        **Wzory:** $H(S) = -\sum_i p_i \log_2 p_i$, $\qquad IG(S, A) = H(S) - \sum_{{v}} \frac{{|S_v|}}{{|S|}} H(S_v)$.
        """),
        md("## Krok 0 (min 0–4): kod pomocniczy i dane"),
        _helper_cell(id3_code),
        _data_cell(csv_text),
        md(r"""
        ## TODO 1 (min 4–9): braki danych
        W dwóch kolumnach brakuje łącznie trzech wartości. Najpierw je **wykryj**, potem **uzupełnij dominantą**,
        czyli najczęstszą wartością danej kolumny. Klasyczny ID3 nie potrafi rozgałęziać się po brakującej wartości.
        """),
        code(r"""
        missing_before = df.isna().sum()
        display(missing_before.to_frame("braki"))
        """),
    ]
    out.append(code(r"""
        # TODO 1: uzupełnij braki dominantą każdej kolumny z brakami (nie zmieniaj pozostałych wartości).
        df_filled = df.fillna({column: df[column].mode()[0] for column in df.columns if df[column].isna().any()})
        """ if solution else r"""
        # TODO 1: uzupełnij braki dominantą każdej kolumny z brakami (nie zmieniaj pozostałych wartości).
        # Wskazówka: df["weather"].mode()[0] zwraca dominantę kolumny; df.fillna({...}) uzupełnia braki.
        df_filled = None   # ← zastąp None swoim kodem
        """))
    out += [
        code(r"""
        check_todo_1(df, df_filled)
        df_ready = df_filled if todo_ok("1") else reference_fill(df)
        display(missing_before_after(df, df_ready))
        """),
        md(r"""
        ## TODO 2 (min 9–12): entropia zbioru
        Uzupełnij funkcję `my_entropy`, która dla listy etykiet zwraca $H(S) = -\sum_i p_i \log_2 p_i$ w bitach.
        """),
    ]
    out.append(code(r"""
        def my_entropy(labels) -> float:
            # TODO 2: H(S) = −Σ p_i · log2(p_i)
            p = pd.Series(list(labels)).value_counts(normalize=True).to_numpy()
            return float(-(p * np.log2(p)).sum())
        """ if solution else r"""
        def my_entropy(labels) -> float:
            # TODO 2: H(S) = −Σ p_i · log2(p_i)
            # Wskazówka: p = pd.Series(list(labels)).value_counts(normalize=True).to_numpy(); potem np.log2(p).
            return None   # ← zastąp swoim kodem
        """))
    out += [
        code(r"""
        check_todo_2(my_entropy, df_ready[TARGET])
        H_S = my_entropy(df_ready[TARGET]) if todo_ok("2") else entropy(df_ready[TARGET])
        print(f"H(S) = {H_S:.4f} bita")
        """),
        md(r"""
        ## TODO 3 (min 12–16): przyrost informacji i wybór korzenia
        **3a.** Policz **samodzielnie** IG cechy `shipping` (2 wartości), korzystając z `my_entropy` i liczności gałęzi.
        **3b.** IG wszystkich czterech cech policz gotową funkcją `information_gain(kolumna, etykiety)` i wybierz korzeń,
        czyli cechę o największym IG.
        """),
    ]
    out.append(code(r"""
        # TODO 3a: IG(shipping) = H(S) − Σ_v |S_v|/|S| · H(S_v)
        n = len(df_ready)
        ig_shipping = H_S - sum(len(group) / n * my_entropy(group[TARGET]) for _, group in df_ready.groupby("shipping"))

        # TODO 3b: IG wszystkich cech i korzeń drzewa
        gains = {feature: information_gain(df_ready[feature], df_ready[TARGET]) for feature in FEATURES}
        root_feature = max(gains, key=gains.get)
        print(f"IG(shipping) policzone ręcznie: {ig_shipping:.4f}")
        """ if solution else r"""
        # TODO 3a: IG(shipping) = H(S) − Σ_v |S_v|/|S| · H(S_v)
        # Wskazówka: for value, group in df_ready.groupby("shipping"): ... len(group), my_entropy(group[TARGET])
        ig_shipping = None   # ← Twoje obliczenie

        # TODO 3b: IG wszystkich cech i korzeń drzewa
        gains = {}           # ← np. {feature: information_gain(df_ready[feature], df_ready[TARGET]) for feature in FEATURES}
        root_feature = None  # ← nazwa cechy o największym IG
        """))
    out += [
        code(r"""
        check_todo_3(df_ready, ig_shipping, gains, root_feature)
        plot_gains(gains if todo_ok("3") else reference_gains(df_ready))
        """),
        md(r"""
        ### Mini-drzewo (gotowy mini-ID3)
        To samo kryterium IG drzewo stosuje w każdym kolejnym węźle. Obejrzyj wydruk i porównaj korzeń ze swoim wyborem.
        """),
        code(r"""
        mini_tree = ID3Classifier().fit(df_ready[FEATURES], df_ready[TARGET])
        print(export_text(mini_tree))
        print(f"\nWęzły: {mini_tree.node_count_}, liście: {mini_tree.leaf_count_}, głębokość: {mini_tree.depth_}")
        """),
        md(r"""
        ## TODO 4 (min 16–20): predykcja nowej przesyłki
        Nowa przesyłka: dystans `long`, pogoda `rain`, obciążenie `high`, wysyłka `express`. Przewidź jej klasę gotowym
        drzewem i **uzasadnij ją 1–2 zdaniami**, opisując ścieżkę od korzenia do liścia.
        """),
    ]
    out.append(code(r"""
        new_parcel = {"distance": "long", "weather": "rain", "workload": "high", "shipping": "express"}
        # TODO 4a: predykcja z drzewa
        my_prediction = mini_tree.predict_one(new_parcel)
        # TODO 4b: uzasadnienie ścieżką w drzewie
        explanation_path = mini_tree.explain_prediction(new_parcel)
        my_explanation = ("Drzewo najpierw sprawdza weather = rain, a w tej gałęzi shipping = express; "
                          f"ten liść zawiera wyłącznie przesyłki {my_prediction}, więc tak brzmi predykcja.")
        print(my_prediction, "—", my_explanation)
        """ if solution else r"""
        new_parcel = {"distance": "long", "weather": "rain", "workload": "high", "shipping": "express"}
        # TODO 4a: predykcja z drzewa (mini_tree.predict_one)
        my_prediction = None
        # TODO 4b: 1–2 zdania uzasadnienia: które cechy drzewo sprawdziło i dlaczego tak zdecydowało?
        my_explanation = ""
        """))
    out += [
        code(r"""
        check_todo_4(mini_tree, new_parcel, my_prediction, my_explanation)
        """),
        md(r"""
        **Porównaj ze ścieżką modelu** (uruchom po wykonaniu TODO 4):
        """),
        code(r"""
        print(mini_tree.explain_prediction(new_parcel).to_text())
        """),
        md(r"""
        ### Podsumowanie: co powinno być gotowe
        - [ ] tabela „braki przed / po” bez braków po uzupełnieniu (TODO 1)
        - [ ] entropia zbioru i ranking IG czterech cech, wybrany korzeń (TODO 2–3)
        - [ ] mini-drzewo i predykcja nowej przesyłki z uzasadnieniem (TODO 4)

        ## Bonus do domu (opcjonalny)
        Zamiast dominanty potraktuj brak jako **osobną kategorię** `"__MISSING__"`. Policz ranking IG i zbuduj drzewo
        jeszcze raz. **Czy wybór korzenia się zmienił? Jak zmieniły się wartości IG i dlaczego?** Wskazówka: policz,
        ile kategorii ma wtedy `weather`, i przypomnij sobie, jak ID3 traktuje atrybuty o wielu wartościach.
        """),
    ]
    out.append(code(r"""
        df_missing = df.fillna("__MISSING__")
        gains_missing = {feature: information_gain(df_missing[feature], df_missing[TARGET]) for feature in FEATURES}
        comparison = pd.DataFrame({"IG: dominanta": pd.Series(gains), "IG: osobna kategoria": pd.Series(gains_missing)})
        comparison["zmiana"] = comparison["IG: osobna kategoria"] - comparison["IG: dominanta"]
        display(comparison.sort_values("IG: dominanta", ascending=False).round(4))
        tree_missing = ID3Classifier().fit(df_missing[FEATURES], df_missing[TARGET])
        print("Korzeń (dominanta):", mini_tree.root_.feature, "| korzeń (osobna kategoria):", tree_missing.root_.feature)
        print(export_text(tree_missing))
        """ if solution else r"""
        # BONUS: osobna kategoria braków
        df_missing = None   # ← np. df.fillna("__MISSING__")
        # Policz gains_missing jak w TODO 3b, zbuduj ID3Classifier().fit(...) i porównaj z wersją z dominantą.
        """))
    if solution:
        out.append(md(r"""
        ### Komentarz prowadzącego do bonusu
        Korzeń się **nie zmienia** i nadal jest nim `weather`, ale jego IG **rośnie**. Osobna kategoria `__MISSING__`
        tworzy dodatkową, czystą gałąź z dwiema przesyłkami `on_time`. Podobnie rośnie IG `workload` (jedna przesyłka
        `delayed` z brakiem). Dwie lekcje:

        1. **ID3 preferuje atrybuty o większej liczbie wartości:** każda dodatkowa, mała i czysta gałąź „za darmo”
           obniża entropię. Korektą jest Gain Ratio z C4.5.
        2. Drzewo z kategorią braku uczy się reguły „brak pogody → na czas”. Jest prawdziwa tylko w tej próbie
           (oba braki wypadły w przesyłkach, które zdążyły), więc łatwo o **wzorzec-artefakt**. Imputacja dominantą
           przypisała tym przesyłkom pogodę `clear`. Nie zmieniło to etykiet w gałęzi, bo obie miały niskie lub średnie
           obciążenie.
        """))
    return out


def answer_key(csv_text: str) -> str:
    """ANSWER_KEY.md content - every number is computed here from the CSV with the project ID3 code."""
    from src.id3 import ID3Classifier, entropy, export_text, information_gain_table

    df = pd.read_csv(io.StringIO(csv_text))
    modes = {c: min(df[c].value_counts().index[df[c].value_counts() == df[c].value_counts().max()])
             for c in df.columns if df[c].isna().any()}
    filled = df.fillna(modes)
    ranking = information_gain_table(filled[FEATURES], filled[TARGET])
    model = ID3Classifier().fit(filled[FEATURES], filled[TARGET])
    explanation = model.explain_prediction(NEW_PARCEL)
    missing = df.fillna("__MISSING__")
    ranking_missing = information_gain_table(missing[FEATURES], missing[TARGET])
    model_missing = ID3Classifier().fit(missing[FEATURES], missing[TARGET])
    shipping_groups = filled.groupby("shipping")[TARGET]
    shipping_terms = " + ".join(f"{len(g)}/{len(filled)}·{entropy(g):.4f}" for _, g in shipping_groups)
    rows = "\n".join(f"| {r.rank} | `{r.feature}` | {r.n_values} | {r.information_gain:.4f} | {r.gain_ratio:.4f} |"
                     for r in ranking.itertuples())
    rows_missing = "\n".join(f"| {r.rank} | `{r.feature}` | {r.n_values} | {r.information_gain:.4f} |"
                             for r in ranking_missing.itertuples())
    missing_cells = ", ".join(f"`{c}`: {int(n)}" for c, n in df.isna().sum().items() if n)
    return f"""# ANSWER_KEY — warsztat „Delivery Decision” (materiał prywatny prowadzącego)

> Nie udostępniaj grupie. Wszystkie liczby wygenerował `scripts/build_notebook.py` z `delivery_data.csv`
> tym samym kodem ID3 co projekt; rozwiązany notebook: `ID3_Delivery_Solution.ipynb`.

## Dane
- {len(df)} przesyłek, cechy {", ".join(f"`{f}`" for f in FEATURES)}, etykieta `{TARGET}`:
  {", ".join(f"{k} = {v}" for k, v in df[TARGET].value_counts().items())}.
- Braki: {missing_cells} (łącznie {int(df.isna().sum().sum())}).

## TODO 1 — uzupełnianie braków dominantą
{chr(10).join(f"- `{c}` → **{v}**" for c, v in modes.items())}
- Po uzupełnieniu: 0 braków, kształt bez zmian ({filled.shape[0]} × {filled.shape[1]}).

## TODO 2 — entropia
- H(S) = **{entropy(filled[TARGET]):.4f}** bita.

## TODO 3 — przyrost informacji (dane po imputacji dominantą)
- IG(`shipping`) ręcznie: {entropy(filled[TARGET]):.4f} − ({shipping_terms}) = **{ranking.set_index("feature").loc["shipping", "information_gain"]:.4f}**

| pozycja | cecha | wartości | IG [bit] | Gain Ratio |
|---:|---|---:|---:|---:|
{rows}

- **Korzeń: `{ranking.iloc[0]["feature"]}`**, przewaga nad drugą cechą
  {ranking.iloc[0]["information_gain"] - ranking.iloc[1]["information_gain"]:.4f} bita, bez remisu.

### Mini-drzewo ({model.node_count_} węzłów, {model.leaf_count_} liści, głębokość {model.depth_})
```text
{export_text(model)}
```

## TODO 4 — predykcja nowej przesyłki
- Przesyłka: {NEW_PARCEL}
- Predykcja: **{explanation.prediction}**
- Ścieżka:
```text
{explanation.to_text()}
```
- Wzorcowe uzasadnienie: deszcz kieruje do gałęzi `weather = rain`, w której decyduje rodzaj wysyłki. Przesyłka
  ekspresowa trafia do czystego liścia z samymi przesyłkami na czas; dystans i obciążenie nie są już sprawdzane.

## Bonus — osobna kategoria braków

| pozycja | cecha | wartości | IG [bit] |
|---:|---|---:|---:|
{rows_missing}

- Korzeń: **`{model_missing.root_.feature}`** (z dominantą: `{model.root_.feature}`), czyli
  {"bez zmiany" if model_missing.root_.feature == model.root_.feature else "ZMIANA"}.
- IG(`weather`) rośnie z {ranking.set_index("feature").loc["weather", "information_gain"]:.4f} do
  {ranking_missing.set_index("feature").loc["weather", "information_gain"]:.4f}: dodatkowa, czysta gałąź `__MISSING__`
  (preferencja ID3 dla atrybutów wielowartościowych; korekta: Gain Ratio). Drzewo uczy się reguły-artefaktu
  „brak pogody → {model_missing.root_.children["__MISSING__"].majority_class}”.

## Checklista prowadzącego
- [ ] Link do folderu (dostęp ograniczony do grupy) wysłany przed zajęciami; uczestnicy robią *Zapisz kopię na Dysku*.
- [ ] Na zajęciach: 0–4 min wprowadzenie, potem TODO 1–4 zgodnie z harmonogramem; po 9. i 16. minucie krótkie
      omówienie (dominanta; ranking IG i korzeń).
- [ ] Najczęstsze trudności: użycie liczności zamiast udziałów w entropii; brak wag |S_v|/|S| w IG; `mode()` zwraca
      Series (potrzebne `[0]`).
- [ ] Dla pracujących w domu: ten sam notebook, kontrolki działają bez prowadzącego; bonus jest opcjonalny.
"""


def expected_answers(csv_text: str) -> dict[str, object]:
    """Answers of the basic scenario, computed from the CSV (used to detect leaks in the participant notebook)."""
    from src.id3 import ID3Classifier, entropy, information_gain_table

    df = pd.read_csv(io.StringIO(csv_text))
    filled = df.fillna({c: min(df[c].value_counts().index[df[c].value_counts() == df[c].value_counts().max()])
                        for c in df.columns if df[c].isna().any()})
    ranking = information_gain_table(filled[FEATURES], filled[TARGET])
    model = ID3Classifier().fit(filled[FEATURES], filled[TARGET])
    return {"root": ranking.iloc[0]["feature"], "entropy": entropy(filled[TARGET]),
            "gains": dict(zip(ranking["feature"], ranking["information_gain"])),
            "prediction": model.predict_one(NEW_PARCEL)}


def leaked_answers(notebook_text: str, csv_text: str) -> list[str]:
    """Patterns that would reveal the expected answers in a participant notebook (empty list = clean)."""
    answers = expected_answers(csv_text)
    patterns = ["ANSWER_KEY", "ROZWIĄZANIE", "instructor_private",
                f"root_feature = \"{answers['root']}\"", f"root_feature = '{answers['root']}'",
                f"my_prediction = \"{answers['prediction']}\"", f"my_prediction = '{answers['prediction']}'",
                f"{answers['entropy']:.4f}", f"{answers['entropy']:.3f}"]
    for value in answers["gains"].values():
        patterns += [f"{value:.4f}", f"{value:.3f}"]
    return sorted({p for p in patterns if p in notebook_text})
