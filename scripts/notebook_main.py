"""Content of notebooks/ID3_Mushroom_Project.ipynb (assembled by scripts/build_notebook.py).

Markdown cells may contain placeholders ``{{name}}`` / ``{{name:.3f}}`` - they are rendered from the
values that the code cells record with ``note(...)`` during pass 1 of the build.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path


def md(text: str) -> dict:
    return {"type": "markdown", "source": textwrap.dedent(text).strip()}


def code(text: str, tags: tuple[str, ...] = ()) -> dict:
    return {"type": "code", "source": textwrap.dedent(text).strip(), "tags": tags}


MODULES = {"type": "modules"}


def listings(id3_path: Path) -> dict[str, str]:
    """Source of the key functions of src/id3.py, extracted at build time (no manual copy)."""
    source = id3_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    found: dict[str, str] = {}

    def grab(node: ast.AST) -> str:
        return textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno]))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"entropy_from_counts", "_gain_from_table",
                                                                "select_best_feature"}:
            found[node.name] = grab(node)
        if isinstance(node, ast.ClassDef) and node.name == "ID3Classifier":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {"_build_node", "_trace"}:
                    found[item.name] = grab(item)
    missing = {"entropy_from_counts", "_gain_from_table", "select_best_feature", "_build_node", "_trace"} - set(found)
    if missing:
        raise RuntimeError(f"listing source not found in src/id3.py: {missing}")
    return found


def cells(listings: dict[str, str], workshop_csv: str) -> list[dict]:
    out: list[dict] = []
    out += chapter_00()
    out += chapter_01()
    out += chapter_02()
    out += chapter_03()
    out += chapter_04()
    out += chapter_05(listings)
    out += chapter_06()
    out += chapter_07()
    out += chapter_08()
    out += chapter_09()
    out += chapter_10(workshop_csv)
    out += chapter_11()
    return out


# ---------------------------------------------------------------------------
# 00. Cover, environment, code and data
# ---------------------------------------------------------------------------

def chapter_00() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-00"></a>
        # ID3 × Mushroom
        ## Klasyczne drzewo decyzyjne ID3: własna implementacja, eksperymenty i interpretacja

        | | |
        |---|---|
        | **Autor** | Kamil Piejko |
        | **Temat** | Algorytm ID3 (*Iterative Dichotomiser 3*) na zbiorze Mushroom (UCI) |
        | **Forma** | notebook Jupyter: jednocześnie projekt, sprawozdanie i prezentacja |
        | **Dane** | UCI Machine Learning Repository, *Mushroom*, DOI [10.24432/C5959T](https://doi.org/10.24432/C5959T), licencja CC BY 4.0 |
        | **Uruchomienie** | JupyterLab lokalnie lub Google Colab (notebook jest samowystarczalny) |

        > ⚠️ **Zastrzeżenie bezpieczeństwa.** Mushroom jest zbiorem **edukacyjnym**: opisuje *hipotetyczne* okazy 23 gatunków
        > z jednego przewodnika terenowego. Żaden wynik tego notebooka, także 100% trafności, nie uprawnia do decyzji
        > o spożyciu dzikich grzybów.
        """),
        md(r"""
        ### Spis treści

        | Rozdział | Zawartość |
        |---|---|
        | [00. Okładka i środowisko](#rozdzial-00) | mapa wymagań, kod projektu, wczytanie danych |
        | [01. Algorytm ID3](#rozdzial-01) | problem, historia, entropia i IG, pseudokod, przykład ręczny, ograniczenia, biblioteki |
        | [02. Zbiór Mushroom](#rozdzial-02) | pochodzenie, kontrakt danych, słownik cech, rozkład klas |
        | [03. EDA i czyszczenie](#rozdzial-03) | audyt jakości, braki, rozkłady, V Craméra, balans klas, zbędne cechy |
        | [04. Dane modelowe](#rozdzial-04) | odpowiedź na „???” ze slajdu, podział 60/20/20, 3 strategie braków, duplikaty |
        | [05. Własny ID3](#rozdzial-05) | pseudokod ↔ kod, pierwszy węzeł (`odor`), testy |
        | [06. Strojenie](#rozdzial-06) | 36 modeli próbnych, krzywa głębokości, strategie braków, reguła wyboru, lasy losowe |
        | [07. Model finalny](#rozdzial-07) | trening na train+val, charakterystyka, ważność cech, reguły |
        | [08. Wizualizacja drzewa](#rozdzial-08) | diagram drzewa, pełne drzewo SVG/PNG, biblioteki grafiki |
        | [09. Ocena i predykcja](#rozdzial-09) | jednorazowy test, macierz pomyłek, metryki, ścieżki predykcji |
        | [10. Warsztat dla grupy](#rozdzial-10) | „Delivery Decision” — zadanie na 20 minut i do domu |
        | [11. Podsumowanie i źródła](#rozdzial-11) | wnioski, ograniczenia, reprodukowalność, bibliografia |
        """),
        md(r"""
        ### Mapa zgodności z wymaganiami prowadzącego

        | Nr | Wymóg ze slajdu | Gdzie w notebooku | Artefakt |
        |---|---|---|---|
        | 1 | Wprowadzenie, przedstawienie algorytmu (pseudokod), zestawienie bibliotek zawierających algorytm | rozdz. 01 | wzory, pseudokod, tabela bibliotek |
        | 2 | Dane, analiza danych, zbalansowanie, korelacje, braki w danych, błędne dane, czyszczenie danych, uzupełnianie danych, usunięcie zbędnych atrybutów | rozdz. 02–03 (+ 04.4) | raport jakości danych, V Craméra, tabela „przed/po” |
        | 3 | Przygotowanie danych do budowy modelu, `???`, dane uczące, walidacyjne i testowe | rozdz. 04 | protokół podziału, `split_indices.csv` |
        | 4 | Modele próbne, dostrojenie parametrów, ocena jakości, dobór parametrów; `(Nie ??? w przypadku lasów)` | rozdz. 06 | `experiment_results.csv`, wykresy, reguła wyboru |
        | 5 | Budowa modelu finalnego | rozdz. 07 | `final_model/final_tree.json`, `final_config.json` |
        | 6 | Graficzne przedstawienie drzewa, zestawienie bibliotek generujących grafikę | rozdz. 08 | `figures/final_tree.svg/.png`, `final_tree.dot` |
        | 7 | Ocena modelu, macierz pomyłek, predykcje | rozdz. 09 | metryki, `test_predictions.csv` |
        | 8 | Inne dane i zadanie dla grupy, kod w miejscu dostępnym tylko dla grupy | rozdz. 10 | `group_exercise/` (paczka uczestnika) |

        **Wymóg formalny:** notebook `.ipynb` z zapisanymi wynikami oraz dane spakowane w ZIP
        (`submission/ID3_Mushroom_Submission.zip`), przesłane mailem co najmniej dzień przed zajęciami.
        """),
        md(r"""
        ### Konwencje

        - Każda sekcja ma układ **cel/teoria → obliczenia → wykres lub tabela → interpretacja**.
        - Kolory klas są stałe na wszystkich wykresach: <span style="color:#2a9d8f">■</span> **jadalny (`e`)**,
          <span style="color:#e76f51">■</span> **trujący (`p`)**. Paletę sprawdzono walidatorem kontrastu i odróżnialności
          przy daltonizmie.
        - Liczby zapisujemy z kropką dziesiętną, tak jak w kodzie i tabelach. Entropię i IG podajemy w bitach.
        - **Każda liczba w komentarzach pochodzi z obliczeń.** Generator notebooka wstawia ją z rzeczywistego uruchomienia,
          a ostatnia komórka (rozdz. 11) porównuje wszystkie cytowane wartości z wynikami bieżącego przebiegu.
        """),
        md(r"""
        ## 0.1. Środowisko i kod projektu

        Notebook jest **samowystarczalny**: działa w czystym Google Colab bez folderu `src/`. Kod modułów nie jest jednak
        utrzymywany w dwóch miejscach. Skrypt `scripts/build_notebook.py` kopiuje pliki `src/*.py` do komórek poniżej
        w kolejności zależności `data_io → preprocessing → id3 → evaluation → experiments → visualization`. Usuwa przy tym
        importy wewnątrzpakietowe (`from .id3 import …`) i sprawdza, że żaden moduł nie nadpisuje nazw innego. Test
        `tests/test_notebook.py` pilnuje, by komórki były identyczne z `src/`.

        | Moduł | Odpowiedzialność | Rozdziały |
        |---|---|---|
        | `data_io` | ZIP z UCI (lokalnie lub pobranie w Colab), kontrakt danych, słownik kategorii | 00, 02 |
        | `preprocessing` | audyt jakości, V Craméra, podział 60/20/20, „sejf” zbioru testowego, strategie braków | 03, 04 |
        | `id3` | **własna implementacja ID3**: entropia, IG, budowa drzewa, predykcja, wyjaśnienia | 01, 05–09 |
        | `evaluation` | metryki, macierz pomyłek z ustaloną kolejnością klas, błąd krytyczny `p → e` | 06, 09 |
        | `experiments` | siatka 36 konfiguracji, krzywa głębokości, deterministyczna reguła wyboru | 06, 07 |
        | `visualization` | wykresy oraz diagram drzewa (Graphviz lub matplotlib) | wszystkie |

        Komórki modułów są zwinięte (JupyterLab/Colab): kliknij, aby rozwinąć. Najważniejsze fragmenty ID3 są omówione
        w rozdziale 05.
        """),
        code(r"""
        # Biblioteki zewnętrzne (preinstalowane w Google Colab) i ich wersje
        import importlib.metadata as metadata
        import importlib.util
        import io
        import json
        import os
        import platform
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import Image, Markdown, Math, SVG, display

        IN_COLAB = "google.colab" in sys.modules
        pd.set_option("display.max_colwidth", 110)
        pd.set_option("display.width", 200)


        def package_version(name: str) -> str:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return "—"


        ENVIRONMENT = {"Python": platform.python_version(), "uruchomienie": "Google Colab" if IN_COLAB else "lokalnie",
                       **{name: package_version(name) for name in
                          ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn")}}
        display(pd.DataFrame([ENVIRONMENT]))
        """),
        MODULES,
        md(r"""
        ## 0.2. Dane: lokalny ZIP lub pobranie z UCI

        - **Lokalnie:** używamy dołączonego `data/mushroom.zip` i czytamy plik strumieniowo z archiwum (`zipfile`).
          Nie trzeba niczego rozpakowywać ani łączyć się z internetem.
        - **Colab:** jeśli pliku nie ma, notebook pobiera oryginalne archiwum
          `https://archive.ics.uci.edu/static/public/73/mushroom.zip`. Bez sieci zatrzymuje się czytelnym komunikatem
          z instrukcją ręcznego wgrania pliku (`files.upload()`).

        Kontrakt danych sprawdzamy na **surowym** pliku, zanim zmienimy w nim cokolwiek.
        """),
        code(r"""
        PROJECT_ROOT = find_project_root()                       # katalog projektu lokalnie, /content w Colab
        DATA_ZIP = ensure_mushroom_zip(PROJECT_ROOT / "data" / "mushroom.zip")   # pobiera tylko wtedy, gdy pliku brak
        OUTPUT_DIR = PROJECT_ROOT / "outputs"
        FIG_DIR, MODEL_DIR = OUTPUT_DIR / "figures", OUTPUT_DIR / "final_model"
        for folder in (FIG_DIR, MODEL_DIR):
            folder.mkdir(parents=True, exist_ok=True)

        FACTS: dict[str, object] = {}   # liczby cytowane w komentarzach; kontrola spójności w rozdziale 11


        def note(**values) -> None:
            # Zapamiętuje wartości cytowane w tekście (porównywane z narracją na końcu notebooka).
            for key, value in values.items():
                FACTS[key] = value.item() if hasattr(value, "item") else value


        def show(fig, name: str, formats=("png",)) -> None:
            # Zapisuje wykres do outputs/figures/<name>.<format> i wyświetla go pod komórką.
            save_figure(fig, FIG_DIR / name, formats)
            plt.show()


        set_presentation_style()
        raw_df = read_raw_mushroom(DATA_ZIP)              # surowy plik: same napisy, znaki '?' bez zmian
        contract_report = validate_raw_contract(raw_df)   # DataContractError, jeśli dane ≠ dokumentacja UCI
        df = load_mushroom(DATA_ZIP)                      # jedyna modyfikacja: '?' → NaN
        X_all, y_all = split_features_target(df)
        print(f"Archiwum: {DATA_ZIP.relative_to(PROJECT_ROOT)}  (SHA-256 {file_sha256(DATA_ZIP)[:16]}…)")
        print(f"Wczytano {df.shape[0]} wierszy × {df.shape[1]} kolumn; kontrakt UCI zgodny: "
              f"{bool(contract_report['zgodne'].all())}")
        """),
    ]


# ---------------------------------------------------------------------------
# 01. ID3
# ---------------------------------------------------------------------------

def chapter_01() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-01"></a>
        # 01. Algorytm ID3

        ### Problem
        Klasyfikacja nadzorowana: obiekt opisany wektorem cech **nominalnych** $x = (a_1, \dots, a_d)$ chcemy
        przypisać do klasy $y \in \{e, p\}$ (*edible* / *poisonous*). Drzewo decyzyjne robi to serią pytań. Każdy węzeł
        wewnętrzny testuje jeden atrybut, każda gałąź odpowiada jednej jego wartości, a liść zwraca klasę. Model jest
        **czytelny dla człowieka**, bo każdą predykcję uzasadnia ścieżka reguł od korzenia do liścia.

        ### Historia
        - **1966 — CLS** (*Concept Learning System*; E. B. Hunt, J. Marin, P. J. Stone): rekurencyjny podział zbioru
          przykładów, z którego wyrasta ID3.
        - **1979–1986 — ID3** (J. R. Quinlan): atrybut do podziału wybiera się kryterium z teorii informacji Shannona.
          Kanoniczny opis to *Induction of Decision Trees*, *Machine Learning* 1, 81–106 (1986). Słowo „iteracyjny” w nazwie
          pochodzi od techniki **okna**: drzewo budowano na podzbiorze przykładów i iteracyjnie dokładano te, które
          klasyfikowało błędnie.
        - **1984 — CART** (Breiman, Friedman, Olshen, Stone): drzewa binarne z indeksem Giniego. Na nich opiera się
          `scikit-learn`.
        - **1993 — C4.5** (Quinlan): Gain Ratio, atrybuty ciągłe, braki danych, przycinanie. Następca: C5.0.
        """),
        md(r"""
        ## 1.1. Entropia i przyrost informacji

        Niech $S$ będzie zbiorem przykładów, a $p_i$ udziałem klasy $i$ w $S$ (dla $k$ klas). **Entropia** Shannona mierzy
        niepewność co do klasy losowo wybranego przykładu:

        $$H(S) = -\sum_{i=1}^{k} p_i \log_2 p_i, \qquad 0 \cdot \log_2 0 := 0 .$$

        - Logarytm o podstawie 2 wyraża entropię w **bitach**: to średnia liczba pytań „tak/nie” potrzebnych do ustalenia
          klasy przy optymalnym kodowaniu.
        - $H(S) = 0$ wtedy i tylko wtedy, gdy zbiór jest **czysty** (jedna klasa). Taki węzeł staje się liściem.
        - Dla dwóch klas maksimum $H = 1$ bit przypada przy $p = 0.5$.
        - Konwencja $0\log_2 0 := 0$ wynika z granicy $\lim_{p\to 0^+} p\log_2 p = 0$. W kodzie zerowe liczności usuwamy
          **przed** logarytmowaniem, więc $\log_2 0$ nigdy nie jest liczony.

        **Przyrost informacji** (*information gain*) atrybutu $A$ to oczekiwany spadek entropii po podziale $S$
        na podzbiory $S_v$, po jednym dla każdej wartości $v$ atrybutu:

        $$IG(S, A) = H(S) - \sum_{v \in V(A)} \frac{|S_v|}{|S|}\, H(S_v).$$

        Odejmowana suma to **entropia warunkowa** $H(S \mid A)$, czyli średnia entropia gałęzi ważona ich licznością.
        $IG(S,A)$ jest równy informacji wzajemnej $I(A;Y)$, więc zawsze $0 \le IG(S,A) \le H(S)$.

        ID3 jest algorytmem **zachłannym**. W każdym węźle wybiera atrybut o największym $IG$, tworzy **gałąź dla każdej
        wartości** tego atrybutu obecnej w węźle i powtarza procedurę rekurencyjnie, nie wracając do wcześniejszych
        decyzji. Liście powstają, gdy węzeł jest czysty albo brakuje atrybutów do testowania.
        """),
        code(r"""
        p_poisonous = float((y_all == "p").mean())
        note(p_poisonous=p_poisonous, H_class=entropy(y_all))
        show(plot_entropy_curve(("Mushroom: udział grzybów trujących", p_poisonous)), "01_entropy_curve")
        """),
        md(r"""
        **Interpretacja.** Krzywa jest symetryczna i wklęsła. Zero bitów niepewności mają zbiory czyste (`p = 0` lub `p = 1`),
        maksimum, 1 bit, przypada na równowagę klas. W zbiorze Mushroom grzyby trujące stanowią {{p_poisonous:.2%}}
        obserwacji, więc entropia całego zbioru wynosi {{H_class:.4f}} bita, czyli prawie maksimum. Zadaniem drzewa jest
        sprowadzenie tej niepewności do zera w liściach.
        """),
        md(r"""
        ## 1.2. Pseudokod ID3

        ```text
        ID3(S, A, d)                     ▷ S – przykłady, A – niewykorzystane atrybuty, d – głębokość węzła
         1  utwórz węzeł N; zapisz w nim rozkład klas S, klasę większościową, |S| i H(S)
         2  jeżeli wszystkie przykłady S należą do jednej klasy c:      zwróć liść(c)
         3  jeżeli A = ∅:                                              zwróć liść(klasa większościowa S)
         4  jeżeli d = max_depth lub |S| < min_samples_split:          zwróć liść(klasa większościowa S)   ▷ rozszerzenie
         5  dla każdego a ∈ A oblicz IG(S, a)
         6  a* ← argmax IG(S, a)                                      ▷ remis → alfabetycznie po nazwie
         7  jeżeli IG(S, a*) ≤ ε lub IG(S, a*) < min_information_gain: zwróć liść(klasa większościowa S)
         8  N.atrybut ← a*
         9  dla każdej wartości v atrybutu a* występującej w S:
        10      S_v ← { x ∈ S : x[a*] = v }
        11      N.dziecko[v] ← ID3(S_v, A \ {a*}, d + 1)
        12  zwróć N

        PREDYKCJA(N, x)
         1  dopóki N nie jest liściem:
         2      v ← x[N.atrybut]
         3      jeżeli v ∉ N.dzieci:  zwróć N.klasa_większościowa    ▷ kategoria niewidziana w treningu
         4      N ← N.dziecko[v]
         5  zwróć N.klasa
        ```

        Kroki 4 i 7 z parametrami `max_depth`, `min_samples_split` i `min_information_gain` to **dydaktyczne rozszerzenia**
        (*pre-pruning*, przycinanie wstępne). Klasyczny ID3 Quinlana zatrzymuje się tylko w krokach 2–3. Próg
        $\varepsilon = 10^{-12}$ chroni przed gałęziami o zerowym przyroście informacji. Remisy klas w liściu
        rozstrzygamy na korzyść klasy `p`: przy niepewności bezpieczniej uznać grzyb za trujący.
        """),
        md(r"""
        ## 1.3. Przykład: pierwsze rozgałęzienie „ręcznie”

        Losujemy z Mushroom **12 rzeczywistych rekordów**, po 6 z każdej klasy (`random_state=42`), i trzy atrybuty:
        `odor`, `gill-size`, `bruises`. Liczymy entropię zbioru, entropię każdej gałęzi i przyrost informacji, czyli
        dokładnie to, co ID3 robi w korzeniu. Wszystkie liczby wylicza kod. Próbka służy wyłącznie ilustracji wzoru i nie
        wpływa na żaden model.
        """),
        code(r"""
        example = (df.groupby("class", group_keys=False).sample(n=6, random_state=42)
                   [["odor", "gill-size", "bruises", "class"]].reset_index(drop=True))
        display(expand_frame(example))
        """),
        code(r"""
        def h_terms(counts) -> str:
            # Rozpisanie -Σ p·log2(p) w LaTeX-u; dla zbioru czystego zwraca 0.
            n, nonzero = sum(counts), [c for c in counts if c]
            if len(nonzero) == 1:
                return r"0 \;\text{(zbiór czysty)}"
            return " ".join(rf"- \tfrac{{{c}}}{{{n}}}\log_2\tfrac{{{c}}}{{{n}}}" for c in nonzero)


        def gain_derivation(frame: pd.DataFrame, feature: str, target: str = "class") -> str:
            labels = sorted(frame[target].unique())
            h_parent = entropy(frame[target])
            terms = gain_breakdown(frame[feature], frame[target])
            rows = [rf"H(S) &= {h_terms(frame[target].value_counts().reindex(labels).tolist())} = {h_parent:.4f}"]
            for row in terms.itertuples(index=False):
                counts = [getattr(row, f"n_{label}") for label in labels]
                rows.append(rf"H(S_{{\text{{{feature}}}={row.value}}}) &= {h_terms(counts)} = {row.entropy:.4f}")
            weighted = " + ".join(rf"\tfrac{{{r.n}}}{{{len(frame)}}}\cdot{r.entropy:.4f}" for r in terms.itertuples())
            gain = information_gain(frame[feature], frame[target])
            rows.append(rf"IG(S, \text{{{feature}}}) &= {h_parent:.4f} - \left({weighted}\right) = \mathbf{{{gain:.4f}}}")
            return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


        display(Math(gain_derivation(example, "gill-size")))
        display(Math(gain_derivation(example, "odor")))
        example_gains = information_gain_table(example[["odor", "gill-size", "bruises"]], example["class"])
        display(example_gains)
        eg = example_gains.set_index("feature")
        note(ex_n=len(example), ex_best=example_gains.iloc[0]["feature"], ex_ig_odor=eg.loc["odor", "information_gain"],
             ex_ig_bruises=eg.loc["bruises", "information_gain"], ex_ig_gill=eg.loc["gill-size", "information_gain"],
             ex_odor_values=int(eg.loc["odor", "n_values"]), ex_gr_odor=eg.loc["odor", "gain_ratio"],
             ex_gr_bruises=eg.loc["bruises", "gain_ratio"])
        """),
        md(r"""
        **Interpretacja.** W 12-elementowej próbce `odor` przyjmuje {{ex_odor_values}} wartości i rozdziela klasy
        **idealnie**: każda gałąź jest czysta, więc $IG = H(S) =$ {{ex_ig_odor:.4f}} bita. Rozkłady `gill-size` w obu
        gałęziach są prawie takie same jak w całej próbce, stąd IG zaledwie {{ex_ig_gill:.4f}}. `bruises` daje
        {{ex_ig_bruises:.4f}}. ID3 wybrałby więc `{{ex_best}}`. Kolumna `gain_ratio` pokazuje korektę z C4.5 (sekcja 1.4).
        Dzieląc IG przez entropię samego podziału (*SplitInfo*), zmniejsza ona przewagę atrybutu o wielu wartościach:
        {{ex_gr_odor:.3f}} dla `odor` wobec {{ex_gr_bruises:.3f}} dla `bruises`.
        """),
        md(r"""
        ## 1.4. Ograniczenia ID3

        | Ograniczenie | Na czym polega | Typowa odpowiedź |
        |---|---|---|
        | **Preferencja atrybutów wielowartościowych** | IG rośnie z liczbą kategorii, bo wiele małych gałęzi „za darmo” obniża entropię. Identyfikator rekordu daje $IG = H(S)$. | **Gain Ratio** (C4.5): $GR = IG(S,A) / SplitInfo(S,A)$, gdzie $SplitInfo(S,A) = -\sum_v \frac{\lvert S_v\rvert}{\lvert S\rvert}\log_2\frac{\lvert S_v\rvert}{\lvert S\rvert}$ |
        | **Szum i przeuczenie** | Drzewo rośnie do czystych liści, więc dopasowuje się również do przypadkowych wyjątków. | przycinanie wstępne (głębokość, liczność, próg IG) lub następcze (C4.5, CART); zbiór walidacyjny |
        | **Tylko atrybuty nominalne** | Klasyczny ID3 nie dzieli zmiennych ciągłych. | wcześniejsza **dyskretyzacja**: przedziały równej szerokości lub częstości, progi wg entropii (Fayyad–Irani, 1993); w C4.5 progi binarne |
        | **Braki danych** | Test $x[a] = v$ nie ma sensu dla nieznanej wartości. | jawna kategoria „brak”, imputacja dopasowana na zbiorze uczącym, ułamkowe przykłady w C4.5 |
        | **Zachłanność** | Lokalnie najlepszy podział nie gwarantuje najmniejszego drzewa. Budowa optymalnego drzewa jest NP-zupełna (Hyafil, Rivest, 1976). | akceptujemy; alternatywą są komitety (lasy losowe, *boosting*) |
        | **Złożoność obliczeniowa** | Węzeł z $n$ przykładami i $m$ kandydatami kosztuje $O(n\,m)$. Przykłady jednego poziomu się nie powtarzają, więc poziom kosztuje $O(N\,d)$, a głębokość nie przekracza $d$. | budowa w najgorszym razie $O(N\,d^2)$, predykcja $O(\text{głębokość}) \le O(d)$; na Mushroom jedno drzewo powstaje w ułamku sekundy |
        """),
        md(r"""
        ## 1.5. Biblioteki zawierające algorytm ID3

        Zestawienie ma charakter informacyjny. Zgodnie z wymaganiami **nie trenujemy** tych implementacji i nie
        porównujemy ich wyników. Wszystkie modele w notebooku powstają z własnej implementacji (`src/id3.py`).

        | Biblioteka | Środowisko / język | Czy to ID3? | Ograniczenia, uwagi | Źródło |
        |---|---|---|---|---|
        | **WEKA** `weka.classifiers.trees.Id3` | Java: GUI Explorer, CLI, API; pakiet `simpleEducationalLearningSchemes` | **tak**, klasyczny ID3 | tylko atrybuty nominalne, brak obsługi braków danych, drzewo nieprzycinane; puste gałęzie mogą zostawić przykład niesklasyfikowany | [dokumentacja WEKA](https://weka.sourceforge.io/doc.packages/simpleEducationalLearningSchemes/weka/classifiers/trees/Id3.html) |
        | **RWeka** | R, interfejs do WEKA (wymaga Javy) | **tak**, pośrednio: `WPM("install-package", "simpleEducationalLearningSchemes")`, potem `make_Weka_classifier("weka/classifiers/trees/Id3")` | ograniczenia jak w WEKA; zależność od `rJava` | [CRAN: RWeka](https://cran.r-project.org/package=RWeka) |
        | **chefboost** | Python | **tak**, obok C4.5, CART, CHAID i drzew regresyjnych | cechy liczbowe zamienia na binarne progi `<=x` / `>x`, co wykracza poza klasyczny ID3 | [GitHub: serengil/chefboost](https://github.com/serengil/chefboost) |
        | **decision-tree-id3** | Python, API zgodne ze scikit-learn (`Id3Estimator`) | **tak**, z rozszerzeniami (cechy liczbowe, opcjonalne przycinanie) | ostatnie wydanie 0.1.2 z maja 2017; zgodności z bieżącymi wersjami bibliotek nie badano | [PyPI](https://pypi.org/project/decision-tree-id3/) |
        | **scikit-learn** `DecisionTreeClassifier(criterion="entropy")` | Python | **nie**: zoptymalizowany CART | wyłącznie podziały **binarne** na progach liczbowych; brak natywnych cech kategorycznych; `criterion="entropy"` zmienia tylko miarę nieczystości | [dokumentacja scikit-learn, 1.10.6](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart) |
        | R `rpart` | R | **nie**: CART | podziały binarne (Gini lub *information*) | [CRAN: rpart](https://cran.r-project.org/package=rpart) |
        | R `C50` | R | **nie**: C5.0, następca C4.5 | inne kryterium (Gain Ratio), przycinanie, *boosting* | [CRAN: C50](https://cran.r-project.org/package=C50) |

        **Dlaczego `DecisionTreeClassifier` nie jest ID3?** Buduje wyłącznie drzewa binarne: każdy węzeł zadaje
        pytanie „$x_j \le t$?”. Kategorię trzeba zakodować liczbą (porządek kodów jest sztuczny) albo rozbić na zmienne
        zero-jedynkowe (wtedy jeden test sprawdza jedną kategorię). Atrybut może się też powtarzać na ścieżce. ID3 tworzy
        jedną gałąź dla każdej kategorii i każdy atrybut testuje na ścieżce najwyżej raz. Powstają więc inne drzewa,
        mimo tego samego kryterium entropii.
        """),
    ]


# ---------------------------------------------------------------------------
# 02. Mushroom
# ---------------------------------------------------------------------------

def chapter_02() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-02"></a>
        # 02. Zbiór danych Mushroom

        **Pochodzenie.** Rekordy opisują *hipotetyczne* okazy 23 gatunków grzybów blaszkowych z rodzin *Agaricus*
        i *Lepiota*, spisane z przewodnika *The Audubon Society Field Guide to North American Mushrooms* (1981). Zbiór
        przekazał J. Schlimmer w 1987 roku, a dziś udostępnia go UCI Machine Learning Repository
        (DOI 10.24432/C5959T) na licencji **CC BY 4.0**. Gatunki „o nieznanej jadalności, niezalecane” połączono
        w źródle z trującymi, więc klasa `p` oznacza w praktyce „nie jeść”.

        **Kontrakt danych** wg pliku `agaricus-lepiota.names`:

        - 8124 rekordy, 22 atrybuty nominalne i klasa;
        - rozkład klas 4208 `e` / 3916 `p`;
        - 2480 braków oznaczonych `?`, wszystkie w atrybucie `stalk-root`.

        Sprawdzamy to na surowym pliku, przed jakąkolwiek modyfikacją.
        """),
        code(r"""
        display(zip_members(DATA_ZIP))
        display(contract_report)
        names_check = compare_with_names_file(DATA_ZIP)
        print(f"Słownik kategorii w kodzie zgodny z sekcją 7 pliku .names: "
              f"{int(names_check['zgodne'].sum())}/{len(names_check)} cech")
        note(n_rows=df.shape[0], n_columns=df.shape[1], n_features=X_all.shape[1],
             n_e=int((y_all == "e").sum()), n_p=int((y_all == "p").sum()),
             share_e=float((y_all == "e").mean()), share_p=float((y_all == "p").mean()),
             n_missing_total=int(df.isna().sum().sum()), missing_share=float(df["stalk-root"].isna().mean()),
             names_ok=int(names_check["zgodne"].sum()))
        """),
        md(r"""
        **Interpretacja.** Wszystkie pięć kontroli kontraktu jest zgodnych z dokumentacją UCI: wymiary
        {{n_rows}} × {{n_columns}}, klasy {{n_e}} `e` / {{n_p}} `p`, liczba znaków `?` równa {{n_missing_total}},
        wyłącznie w `stalk-root`. Słownik kategorii zapisany w kodzie (`CATEGORY_NAMES`) jest identyczny z plikiem `.names` dla
        {{names_ok}}/22 cech. Archiwum zawiera też pliki pomocnicze (`Index`, `README`, `expanded.Z`), z których nie
        korzystamy.

        ### Słownik cech i kategorii
        Wszystkie atrybuty są nominalne. Kody jednoliterowe rozwijamy na nazwy z dokumentacji,
        np. `odor = f` → `foul (f)`.
        """),
        code(r"""
        display(dictionary_frame())
        """),
        md(r"""
        ### Przykładowe rekordy
        Pięć pierwszych wierszy po rozwinięciu kodów (transpozycja: cechy w wierszach). Brak danych w `stalk-root`
        pokazujemy jako „brak danych (?)”.
        """),
        code(r"""
        display(expand_frame(df.head(5)).T.rename(columns=lambda i: f"rekord {i}"))
        """),
        md(r"""
        ### Rozkład klas i profil cech
        """),
        code(r"""
        profile = feature_profile(X_all)
        unobserved = unobserved_categories(X_all)
        show(plot_dataset_overview(y_all, profile), "02_dataset_overview")
        note(max_cardinality=int(profile["kategorie obserwowane"].max()),
             max_cardinality_feature=profile.loc[profile["kategorie obserwowane"].idxmax(), "cecha"],
             n_features_unobserved=len(unobserved))
        """),
        md(r"""
        **Interpretacja.** Klasy są prawie równoliczne: {{share_e:.2%}} jadalnych i {{share_p:.2%}} trujących.
        Liczba obserwowanych kategorii waha się od 1 (`veil-type`) do {{max_cardinality}} (`{{max_cardinality_feature}}`).
        W {{n_features_unobserved}} cechach część kategorii ze słownika nie występuje ani razu (np. `veil-type = universal`,
        `stalk-root = cup`). To właściwość próby, a nie błąd danych: drzewo nie utworzy dla nich gałęzi, a mechanizm
        predykcji obsłuży je awaryjnie (rozdz. 04.6).
        """),
    ]


# ---------------------------------------------------------------------------
# 03. EDA and cleaning
# ---------------------------------------------------------------------------

def chapter_03() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-03"></a>
        # 03. Analiza danych i czyszczenie

        Analiza w tym rozdziale jest **opisowa** i obejmuje cały zbiór. Żadna decyzja wpływająca na model nie jest tu
        jednak podejmowana na podstawie zbioru testowego. Transformacje uczone z danych (dominanty, kategorie, lista cech
        stałych) dopasowujemy w rozdziale 04 wyłącznie na zbiorze uczącym.

        ## 3.1. Audyt jakości danych
        Sprawdzamy typy, zgodność wartości ze słownikiem UCI, format kodów, puste napisy, etykiety klas, duplikaty
        (pełne i sprzeczne), braki, cechy stałe oraz kategorie zadeklarowane, ale nieobecne. **Wynik negatywny**
        („nie znaleziono”) też jest wynikiem i pokazujemy go jawnie.
        """),
        code(r"""
        audit = data_quality_audit(df)
        display(audit)
        duplicates_raw = duplicate_stats(X_all, y_all)
        note(n_out_of_domain=sum(len(v) for v in out_of_domain_values(X_all).values()),
             n_dup_rows=duplicates_raw["duplicate_rows"], n_dup_vectors=duplicates_raw["duplicate_feature_vectors"],
             n_conflicts=duplicates_raw["conflicting_vectors"])
        """),
        md(r"""
        **Interpretacja.** Audyt **nie znalazł błędnych danych**:

        - wartości spoza słownika UCI: {{n_out_of_domain}};
        - pełne duplikaty wierszy: {{n_dup_rows}};
        - powtórzone wektory cech: {{n_dup_vectors}};
        - wektory z różnymi klasami: {{n_conflicts}};
        - poza tym brak nieprawidłowych typów, pustych napisów i nietypowych kodów.

        Do obsłużenia zostają dwa problemy wynikające z dokumentacji: braki w `stalk-root` (3.2) i stała cecha `veil-type`
        (3.6). Kategorie niewystępujące w danych mają charakter informacyjny.

        ## 3.2. Braki danych w `stalk-root`
        Braki liczymy **po zamianie `?` na NaN**, czyli jako brakującą obserwację, a nie poprawną wartość biologiczną.
        Sprawdzamy też, czy braki są losowe (MCAR) czy powiązane z klasą. Zależność od klasy przemawia przeciw MCAR,
        a brak sam w sobie staje się wtedy informacją.
        """),
        code(r"""
        from scipy.stats import chi2, chi2_contingency

        missing_flag = df["stalk-root"].isna().rename("brak stalk-root")
        missing_by_class = (missing_flag.groupby(y_all).agg(["sum", "count", "mean"])
                            .rename(columns={"sum": "braki", "count": "obserwacje", "mean": "odsetek braków"}))
        chi2_stat = chi2_contingency(pd.crosstab(missing_flag, y_all), correction=False)[0]
        log10_p = chi2.logsf(chi2_stat, df=1) / np.log(10)
        show(plot_missing_overview(df), "03_missing_values")
        display(missing_by_class)
        print(f"Test niezależności χ²(1) = {chi2_stat:.1f}, log10(p) ≈ {log10_p:.0f};  "
              f"V Craméra(brak, klasa) = {cramers_v(missing_flag, y_all):.3f}")
        note(miss_e_n=int(missing_by_class.loc["e", "braki"]), miss_e_share=float(missing_by_class.loc["e", "odsetek braków"]),
             miss_p_n=int(missing_by_class.loc["p", "braki"]), miss_p_share=float(missing_by_class.loc["p", "odsetek braków"]),
             miss_ratio=float(missing_by_class.loc["p", "odsetek braków"] / missing_by_class.loc["e", "odsetek braków"]),
             miss_chi2=float(chi2_stat), miss_v=cramers_v(missing_flag, y_all))
        """),
        md(r"""
        **Interpretacja.** Braki dotyczą wyłącznie `stalk-root` ({{n_missing_total}} z {{n_rows}}, czyli {{missing_share:.2%}}).
        Nie rozkładają się jednak równo między klasy: brakuje {{miss_e_share:.2%}} wartości u grzybów jadalnych
        ({{miss_e_n}}) i {{miss_p_share:.2%}} u trujących ({{miss_p_n}}), {{miss_ratio:.2f}} razy częściej. Test χ²
        ({{miss_chi2:.1f}}) jednoznacznie odrzuca niezależność braku od klasy, więc braki **nie są MCAR**, a sam fakt braku
        niesie informację (V Craméra {{miss_v:.3f}}). To argument za strategią „brak jako osobna kategoria” (rozdz. 04).
        Mapa braków pokazuje dodatkowo, że w żadnej innej kolumnie braków nie ma.

        ## 3.3. Rozkłady kategorii względem klasy
        """),
        code(r"""
        show(plot_category_distributions(df, ["odor", "spore-print-color", "gill-color", "ring-type", "stalk-root",
                                              "habitat"]), "03_category_distributions")
        odor_purity = pd.crosstab(X_all["odor"], y_all)
        none_counts = odor_purity.loc["n"]
        note(odor_pure_categories=int((odor_purity.min(axis=1) == 0).sum()), odor_categories=len(odor_purity),
             odor_none_n_full=int(none_counts.sum()), odor_none_p_full=int(none_counts["p"]))
        """),
        md(r"""
        **Interpretacja.** `odor` jest niemal deterministyczny: {{odor_pure_categories}} z {{odor_categories}} zapachów
        występuje tylko w jednej klasie (migdałowy i anyżowy tylko u jadalnych, cuchnący, rybi czy ostry tylko
        u trujących). Mieszany pozostaje wyłącznie zapach `none`: {{odor_none_n_full}} grzybów, w tym {{odor_none_p_full}}
        trujących. Właśnie tam drzewo będzie musiało szukać kolejnych testów. Podobnie silnie rozdziela klasy
        `spore-print-color` (np. `green` występuje tylko u trujących). Kategoria „brak danych” w `stalk-root` jest w większości
        trująca, co potwierdza wniosek z 3.2.

        ## 3.4. Korelacje i zależności między cechami nominalnymi
        Współczynnik Pearsona dla kodów liczbowych nie ma sensu: kolejność `a < c < f …` jest arbitralna, więc wynik
        zależałby od przypadkowego kodowania. Dla par zmiennych nominalnych stosujemy **V Craméra**

        $$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1,\; c-1)}} \in [0, 1],$$

        gdzie $r, c$ to liczby kategorii, a $\chi^2$ to statystyka testu niezależności tablicy kontyngencji. Dla cechy
        stałej (`veil-type`: $r = 1$) mianownik jest zerem. Funkcja `cramers_v` zwraca wtedy jawnie $0.0$ zamiast
        dzielić przez zero lub zwracać NaN. Z klasą zestawiamy też **informację wzajemną** $I(A;Y)$ (równą IG w korzeniu)
        i **U Theila** $U(Y\mid A) = I(A;Y) / H(Y)$, czyli odsetek niepewności o klasie usuwany przez cechę.
        Informację wzajemną liczy niezależna implementacja ze scikit-learn.
        """),
        code(r"""
        association = association_with_target(X_all, y_all)
        display(association.head(10))
        top_features = association["cecha"].head(12).tolist()
        show(plot_cramers_v_heatmap(cramers_v_matrix(df, ["class", *top_features]),
                                    "V Craméra: klasa i 12 cech najsilniej z nią powiązanych"), "03_cramers_v")
        full_matrix = cramers_v_matrix(X_all)                       # wszystkie 22 cechy, w tym stała veil-type
        upper = np.triu_indices(len(full_matrix), k=1)
        k = int(np.argmax(full_matrix.to_numpy()[upper]))
        assoc = association.set_index("cecha")
        note(v_odor=assoc.loc["odor", "V Craméra"], mi_odor=assoc.loc["odor", "I(A;Y) [bit]"],
             u_odor=assoc.loc["odor", "U Theila"], v_spore=assoc.loc["spore-print-color", "V Craméra"],
             v_veil=assoc.loc["veil-type", "V Craméra"],
             top_pair_a=full_matrix.index[upper[0][k]], top_pair_b=full_matrix.columns[upper[1][k]],
             top_pair_v=float(full_matrix.to_numpy()[upper][k]), matrix_has_nan=bool(full_matrix.isna().any().any()))
        """),
        md(r"""
        **Interpretacja.**

        - Najsilniej z klasą wiąże się `odor`: V = {{v_odor:.3f}}, $I(A;Y)$ = {{mi_odor:.3f}} bita, czyli
          {{u_odor:.1%}} niepewności o klasie. Na drugim miejscu jest `spore-print-color` (V = {{v_spore:.3f}}).
        - Między samymi cechami najsilniej powiązana jest para `{{top_pair_a}}` – `{{top_pair_b}}` (V = {{top_pair_v:.2f}}).
          Takie redundancje **nie szkodzą** ID3: po wybraniu jednej cechy przyrost informacji drugiej w gałęziach spada
          i algorytm sam ją pomija.
        - Stała `veil-type` ma V = {{v_veil:.1f}}. Pełna macierz 22 × 22 nie zawiera ani jednej wartości NaN.

        ## 3.5. Balansowanie klas
        """),
        code(r"""
        class_balance = y_all.value_counts().rename("liczność").to_frame()
        class_balance["udział"] = class_balance["liczność"] / len(y_all)
        display(class_balance)
        note(imbalance_ratio=float(class_balance["liczność"].max() / class_balance["liczność"].min()))
        """),
        md(r"""
        **Decyzja: świadomie nie balansujemy.** Stosunek klas wynosi {{imbalance_ratio:.3f}} : 1 ({{share_e:.2%}} / {{share_p:.2%}}).
        To rozkład praktycznie zrównoważony, przy którym nadpróbkowanie nie usuwa żadnego realnego problemu. Wnosiłoby
        natomiast ryzyko:

        - SMOTE interpoluje cechy liczbowe, a dla cech nominalnych tworzyłby sztuczne, nieistniejące grzyby;
        - zduplikowane rekordy trafiłyby do różnych podzbiorów i zawyżyły wyniki walidacji.

        Zamiast tego **stratyfikujemy** podział (rozdz. 04), więc każdy podzbiór zachowuje proporcje klas. Oceniamy model
        miarami odpornymi na nierównowagę (macro-F1, czułość klasy `p`, liczba błędów `p → e`).

        ## 3.6. Zbędne atrybuty: cecha stała `veil-type`
        **Kryterium usuwania:** atrybut ma najwyżej jedną obserwowaną wartość, więc $H(A) = 0$ i $IG = 0$ w każdym węźle.
        Takiego atrybutu ID3 nigdy nie wybierze. Usunięcie go nie zmienia modelu, a upraszcza dane i analizy.
        Cecha stała na całym zbiorze jest stała w każdym jego podzbiorze. Preprocesor stosuje to samo kryterium
        do zbioru uczącego (rozdz. 04), co sprawdzamy asercją.
        """),
        code(r"""
        constant = find_constant_features(X_all)
        ig_veil = information_gain(X_all["veil-type"], y_all)
        before_after = pd.DataFrame([
            {"etap": "dane surowe", "liczba cech": X_all.shape[1], "cechy stałe": ", ".join(constant),
             "kolumny z brakami": ", ".join(X_all.columns[X_all.isna().any()])},
            {"etap": "po usunięciu cech stałych", "liczba cech": X_all.shape[1] - len(constant), "cechy stałe": "—",
             "kolumny z brakami": ", ".join(X_all.drop(columns=constant).columns[X_all.drop(columns=constant).isna().any()])},
        ])
        display(before_after)
        print(f"veil-type: {X_all['veil-type'].value_counts().to_dict()} · H = {entropy(X_all['veil-type']):.1f} · "
              f"IG = {ig_veil:.1f} · V Craméra = {cramers_v(X_all['veil-type'], y_all):.1f}")
        note(constant_features=", ".join(constant), ig_veil=ig_veil)
        """),
        md(r"""
        **Interpretacja.** Jedyną cechą stałą jest `{{constant_features}}`: wszystkie 8124 grzyby mają osłonę częściową
        (`partial`), a IG = {{ig_veil:.1f}}. Usuwamy ją, a z danych modelowych zostaje 21 atrybutów. Innych cech nie usuwamy.
        Słaby związek z klasą mierzony na całym zbiorze (np. `stalk-shape`) nie jest wystarczającym kryterium: w głębszych
        węzłach słaba w korzeniu cecha może rozstrzygać trudne przypadki. Decyzję zostawiamy kryterium IG w każdym węźle.
        """),
    ]


# ---------------------------------------------------------------------------
# 04. Model data
# ---------------------------------------------------------------------------

def chapter_04() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-04"></a>
        # 04. Przygotowanie danych do budowy modelu

        ## 4.1. Odpowiedź na „???” z punktu 3 slajdu
        Przed podziałem na dane uczące, walidacyjne i testowe trzeba wykonać kilka kroków. Każdy omawiamy i realizujemy
        jawnie:

        1. **Wydzielenie `X` i `y`:** macierz 22 cech nominalnych i wektor klasy `class`.
        2. **Kodowanie zmiennych kategorycznych:** dla ID3 **nie stosujemy** One-Hot ani Ordinal Encodingu. ID3 rozgałęzia
           się bezpośrednio po symbolach (`odor = foul`). Kodowanie porządkowe wprowadziłoby nieistniejący porządek
           `a < c < f`, a one-hot rozbiłby jeden atrybut o 9 wartościach na 9 atrybutów binarnych i zniszczył podział
           wielogałęziowy.
        3. **Dyskretyzacja zmiennych ciągłych:** klasyczny ID3 przyjmuje tylko atrybuty nominalne, więc każdą cechę
           liczbową trzeba wcześniej zamienić na przedziały. W Mushroom nie ma cech ciągłych i krok jest pusty, ale nasza
           implementacja go wymusza (demonstracja niżej).
        4. **Losowe przetasowanie i stratyfikacja:** dwa kolejne podziały `train_test_split(shuffle=True, stratify=y,
           random_state=42)` dają podzbiory 60/20/20 z identycznymi proporcjami klas.
        5. **Fit on train, transform on val/test:** wszystko, czego transformacja się uczy (cechy stałe, dominanta do
           imputacji, znane kategorie), wyznaczamy **wyłącznie na zbiorze uczącym**.

        ## 4.2. Kodowanie i dyskretyzacja w praktyce
        """),
        code(r"""
        demo = pd.DataFrame({"temperatura_C": [-3.5, 12.0, 21.5, 31.0]})
        try:
            ID3Classifier().fit(demo, ["delayed", "on_time", "on_time", "delayed"])
        except ValueError as error:
            print("ID3Classifier odrzuca cechę ciągłą →", error)
        demo["temperatura"] = pd.cut(demo["temperatura_C"], bins=[-np.inf, 5, 25, np.inf],
                                     labels=["zimno", "umiarkowanie", "gorąco"]).astype(str)
        display(demo)
        tiny = ID3Classifier().fit(demo[["temperatura"]], ["delayed", "on_time", "on_time", "delayed"])
        print("Po dyskretyzacji: korzeń =", tiny.root_.feature, "| gałęzie:", list(tiny.root_.children))
        """),
        md(r"""
        **Interpretacja.** Implementacja nie zgaduje znaczenia liczb. Kolumnę liczbową trzeba świadomie zamienić na
        przedziały (`pd.cut`), dopiero wtedy ID3 tworzy po jednej gałęzi dla każdego przedziału. W Mushroom wszystkie
        cechy są już symbolami.

        ## 4.3. Podział 60 / 20 / 20 (stratyfikowany, `random_state=42`)
        Zbiór testowy od razu trafia do **„sejfu”** (`HoldoutVault`). Do rozdziału 09 jest dostępny tylko dla audytów
        danych, które nie dotykają modelu (rozdz. 4.5). Każdy dostęp jest logowany, a jednorazowe „otwarcie” do oceny
        modelu finalnego jest możliwe tylko raz.
        """),
        code(r"""
        split = make_stratified_split(y_all, train_size=0.6, validation_size=0.2, test_size=0.2, random_state=42)
        split.assert_disjoint(len(y_all))
        split_summary = split.summary(y_all)
        display(split_summary)
        show(plot_split_balance(split_summary), "04_split_balance")
        split.assignment().to_csv(OUTPUT_DIR / "split_indices.csv", index=False)

        X_train, y_train = X_all.loc[split.train], y_all.loc[split.train]
        X_val, y_val = X_all.loc[split.validation], y_all.loc[split.validation]
        test_vault = HoldoutVault(X_all.loc[split.test], y_all.loc[split.test])
        print(test_vault)
        shares = split_summary.set_index("zbiór")
        note(n_train=len(split.train), n_val=len(split.validation), n_test=len(split.test),
             p_share_train=float(shares.loc["uczący", "% p"]), p_share_val=float(shares.loc["walidacyjny", "% p"]),
             p_share_test=float(shares.loc["testowy", "% p"]))
        """),
        md(r"""
        **Interpretacja.** Liczności podzbiorów: {{n_train}} / {{n_val}} / {{n_test}}. Indeksy są rozłączne
        i razem pokrywają cały zbiór; sprawdza to `assert_disjoint`. Udział grzybów trujących wynosi {{p_share_train:.2%}},
        {{p_share_val:.2%}} i {{p_share_test:.2%}}, czyli stratyfikacja zachowała proporcje co do setnej części procenta.
        Przypisanie każdego wiersza do podzbioru zapisano w `outputs/split_indices.csv`.

        ## 4.4. Trzy strategie obsługi braków (dopasowanie tylko na zbiorze uczącym)

        | Strategia | Transformacja | Uzasadnienie |
        |---|---|---|
        | `missing_category` | NaN → jawna kategoria `__MISSING__` | brak jest informatywny (3.2); ID3 natywnie obsługuje dodatkową kategorię |
        | `most_frequent` | NaN → dominanta `stalk-root` **ze zbioru uczącego** | klasyczna imputacja; zaciera informację o braku |
        | `drop_stalk_root` | usunięcie kolumny `stalk-root` | rezygnuje z atrybutu zamiast zgadywać wartości |

        We wszystkich strategiach preprocesor usuwa też cechy stałe wykryte na zbiorze uczącym.
        """),
        code(r"""
        preprocessors = {strategy: CategoricalPreprocessor(strategy).fit(X_train) for strategy in MISSING_STRATEGIES}
        strategy_table = pd.DataFrame([{
            "strategia": strategy,
            "usunięte kolumny": ", ".join(pre.dropped_features_),
            "liczba cech": len(pre.feature_names_out_),
            "braki stalk-root w train": pre.missing_counts_fit_.get("stalk-root", 0),
            "wartość wstawiana": pre.fill_values_.get("stalk-root", "— (kolumna usunięta)"),
            "kategorie stalk-root po transformacji": ", ".join(pre.categories_.get("stalk-root", [])) or "—",
            "NaN po transformacji (train / val)": f"{int(pre.transform(X_train).isna().sum().sum())} / "
                                                  f"{int(pre.transform(X_val).isna().sum().sum())}",
        } for strategy, pre in preprocessors.items()])
        display(strategy_table)
        assert all(pre.constant_features_ == ["veil-type"] for pre in preprocessors.values())
        mode_value = preprocessors["most_frequent"].fill_values_["stalk-root"]
        note(n_missing_train=preprocessors["missing_category"].missing_counts_fit_["stalk-root"],
             train_mode=mode_value, train_mode_label=expand_value("stalk-root", mode_value),
             n_features_model=len(preprocessors["missing_category"].feature_names_out_),
             n_features_drop=len(preprocessors["drop_stalk_root"].feature_names_out_))
        """),
        md(r"""
        **Interpretacja.** Na zbiorze uczącym brakuje {{n_missing_train}} wartości `stalk-root`. Dominanta wyznaczona
        **tylko na nim** to `{{train_mode_label}}`. Strategia `most_frequent` wstawia ją również w zbiorze walidacyjnym
        i testowym, nie zaglądając do ich rozkładów. Po każdej transformacji nie zostaje żaden NaN. Modele dostają
        {{n_features_model}} cech (`missing_category`, `most_frequent`) albo {{n_features_drop}} (`drop_stalk_root`).
        Asercja potwierdza, że kryterium cech stałych zastosowane do zbioru uczącego wskazuje `veil-type`, tak jak w 3.6.

        ## 4.5. Audyt duplikatów przed i po strategiach braków
        Usunięcie kolumny albo imputacja mogą w ogólności sprawić, że różne rekordy staną się identyczne. Identyczny
        wektor cech w treningu i teście oznaczałby przeciek: model „widziałby” przykład testowy, a sprzeczne etykiety
        zaniżyłyby osiągalną trafność. Audyt korzysta z logowanego dostępu do zbioru testowego. Bada wyłącznie dane,
        bez modelu i bez metryk.
        """),
        code(r"""
        X_test_audit, y_test_audit = test_vault.audit_view("audyt duplikatów: identyczne wektory cech między podzbiorami")
        dup_audit = duplicate_audit({"train": (X_train, y_train), "validation": (X_val, y_val),
                                     "test": (X_test_audit, y_test_audit)})
        display(dup_audit)
        note(dup_max=int(dup_audit.drop(columns=["wariant", "liczba cech"]).to_numpy().max()))
        del X_test_audit, y_test_audit
        """),
        md(r"""
        **Interpretacja.** Wynik jest **negatywny w każdym wariancie**: największa liczba w tabeli wynosi {{dup_max}}.
        Wbrew obawie, że `drop_stalk_root` albo imputacja dominantą „skleją” rekordy, w Mushroom żadne dwa grzyby nie
        różnią się wyłącznie wartością `stalk-root`. Żaden wektor cech ze zbioru testowego nie występuje więc w danych
        uczących ani walidacyjnych i nie ma sprzecznych etykiet. Narzędzia audytu wykrywają duplikaty poprawnie;
        sprawdza to test na danych z celowo podłożonymi duplikatami (`tests/test_preprocessing.py`).

        ## 4.6. Kategorie niewidziane w treningu
        """),
        code(r"""
        unseen = {strategy: pre.unseen_categories(X_val) for strategy, pre in preprocessors.items()}
        display(pd.DataFrame([{"strategia": s, "niewidziane kategorie w walidacji": str(u) if u else "brak"}
                              for s, u in unseen.items()]))
        note(n_unseen_val=sum(len(values) for found in unseen.values() for values in found.values()))
        """),
        md(r"""
        **Interpretacja.** W walidacji nie ma kategorii nieznanych ze zbioru uczącego (łącznie {{n_unseen_val}}).
        Gdyby się pojawiły, preprocesor ich nie ukrywa, a ID3 przy predykcji zwraca **klasę większościową ostatniego
        węzła**, do którego przykład dotarł. Ten węzeł ma zapisany rozkład klas z treningu. Model nie zgłasza wyjątku
        i nie zagląda do danych testowych. Mechanizm sprawdzają testy jednostkowe (`tests/test_id3.py`).
        """),
    ]


# ---------------------------------------------------------------------------
# 05. Own ID3
# ---------------------------------------------------------------------------

def chapter_05(listings: dict[str, str]) -> list[dict]:
    build = listings["_build_node"]
    math = "\n\n".join(listings[name] for name in ("entropy_from_counts", "_gain_from_table", "select_best_feature"))
    trace = listings["_trace"]
    return [
        md(r"""
        <a id="rozdzial-05"></a>
        # 05. Własna implementacja ID3

        `ID3Classifier` (moduł `id3`) **nie korzysta** z drzew scikit-learn. Ze scikit-learn bierzemy wyłącznie
        `train_test_split` i metryki. Węzeł (`ID3Node`) przechowuje:

        - testowany atrybut i dzieci według kategorii;
        - rozkład klas i klasę większościową (fallback);
        - liczbę obserwacji i entropię;
        - IG wybranego podziału oraz IG wszystkich kandydatów;
        - flagę liścia i powód zatrzymania.

        | Krok pseudokodu (1.2) | Kod |
        |---|---|
        | 1: węzeł z rozkładem, klasą większościową, $\lvert S\rvert$, $H(S)$ | `ID3Classifier._build_node` → `ID3Node(...)` |
        | 2–4: liść (czystość, brak atrybutów, `max_depth`, `min_samples_split`) | `_build_node` → `_as_leaf(node, reason)` |
        | 5: $IG(S,a)$ dla każdego atrybutu | `_gain` → `_gain_from_table` (z `entropy_from_counts`) |
        | 6: argmax z deterministycznym remisem | `select_best_feature` |
        | 7: próg $\varepsilon$ i `min_information_gain` | `_build_node` (`zero_gain`, `min_information_gain`) |
        | 9–11: gałąź dla każdej występującej wartości, bez powtórzeń atrybutu | `_build_node` (`remaining`, pętla po `sorted(set(values))`) |
        | PREDYKCJA, fallback dla nieznanej kategorii | `_trace` → `predict`, `predict_one`, `explain_prediction` |
        | walidacja schematu (NaN, cechy liczbowe) | `_check_frame`, `_check_sample` → `ValueError` |
        | ważność cech: $\sum_{t:\,A_t=A} \frac{n_t}{N}\, IG(t)$, normalizowana do 1 | `feature_importance_` |
        """),
        md("### Listing: rekurencyjna budowa drzewa (wygenerowany automatycznie z `src/id3.py`)\n\n```python\n"
           + build + "\n```"),
        md("### Listing: entropia, przyrost informacji i wybór atrybutu\n\n```python\n" + math + "\n```"),
        md("### Listing: predykcja z fallbackiem i ścieżką decyzji\n\n```python\n" + trace + "\n```"),
        md(r"""
        ## 5.1. Pierwszy węzeł na zbiorze uczącym
        Liczymy IG wszystkich {{n_features_model}} atrybutów w korzeniu (strategia `missing_category`, zbiór uczący)
        i rozpisujemy składniki IG zwycięzcy. Te same liczby trafiają do korzenia drzewa.
        """),
        code(r"""
        pre_mc = preprocessors["missing_category"]
        X_train_mc = pre_mc.transform(X_train)
        root_table = information_gain_table(X_train_mc, y_train)
        display(root_table.head(8))
        show(plot_information_gain_ranking(root_table, "Kandydaci na korzeń: IG na zbiorze uczącym"),
             "05_root_information_gain")
        by_gr = root_table.sort_values(["gain_ratio", "feature"], ascending=[False, True]).reset_index(drop=True)
        rt = root_table.set_index("feature")
        note(ig_root_feature=root_table.iloc[0]["feature"], ig_odor_train=rt.loc["odor", "information_gain"],
             ig_second_feature=root_table.iloc[1]["feature"], ig_second=root_table.iloc[1]["information_gain"],
             ig_third_feature=root_table.iloc[2]["feature"], ig_third=root_table.iloc[2]["information_gain"],
             gr_first=by_gr.iloc[0]["feature"], gr_second=by_gr.iloc[1]["feature"],
             gr_second_value=by_gr.iloc[1]["gain_ratio"], gr_gill_size_ig_rank=int(rt.loc["gill-size", "rank"]),
             gill_size_values=int(rt.loc["gill-size", "n_values"]))
        """),
        code(r"""
        odor_terms = gain_breakdown(X_train_mc["odor"], y_train)
        odor_terms.insert(0, "kategoria", [expand_value("odor", v) for v in odor_terms["value"]])
        display(odor_terms.drop(columns="value"))
        H_train = entropy(y_train)
        remainder = odor_terms["weighted_entropy"].sum()
        print(f"IG(S, odor) = H(S) − Σ |S_v|/|S|·H(S_v) = {H_train:.4f} − {remainder:.4f} = {H_train - remainder:.4f} bita")
        show(plot_root_split(X_train_mc, y_train, "odor", "Pierwszy węzeł: rozkład klas w gałęziach odor (zbiór uczący)"),
             "05_root_split_odor")
        none_row = odor_terms.set_index("value").loc["n"]
        note(H_train=H_train, odor_remainder=remainder, n_odor_branches=len(odor_terms),
             n_odor_pure=int((odor_terms["entropy"] == 0).sum()), odor_none_n=int(none_row["n"]),
             odor_none_p=int(none_row["n_p"]), odor_none_H=float(none_row["entropy"]),
             odor_none_weight=float(none_row["weight"]))
        """),
        md(r"""
        **Interpretacja.** Na zbiorze uczącym $H(S)$ = {{H_train:.4f}}, a ważona entropia gałęzi `odor` to tylko
        {{odor_remainder:.4f}}. Daje to $IG$ = {{ig_odor_train:.4f}} bita, wyraźnie więcej niż u kolejnych kandydatów:
        `{{ig_second_feature}}` ({{ig_second:.3f}}) i `{{ig_third_feature}}` ({{ig_third:.3f}}). Z {{n_odor_branches}}
        gałęzi {{n_odor_pure}} jest od razu czystych i staje się liśćmi. Całą pozostałą niepewność niesie gałąź `none`:
        $n$ = {{odor_none_n}}, w tym {{odor_none_p}} trujących, $H$ = {{odor_none_H:.3f}}, waga {{odor_none_weight:.1%}}.

        Ranking **Gain Ratio** (prawy panel) też stawia na czele `{{gr_first}}`, ale na drugie miejsce awansuje
        `{{gr_second}}` ({{gr_second_value:.3f}}). To cecha o {{gill_size_values}} wartościach, która według IG zajmuje
        dopiero {{gr_gill_size_ig_rank}}. pozycję. Tak w praktyce wygląda korekta preferencji IG wobec atrybutów
        wielowartościowych. Kryterium ID3 pozostaje jednak IG.

        ## 5.2. Wyniki testów
        Pełny zestaw testów (`pytest`) znajduje się w katalogu `tests/` i jest raportowany w `outputs/QA_REPORT.md`.
        Poniżej najpierw wbudowane kontrole działające także w Colab, a potem, jeśli folder `tests/` jest dostępny,
        wynik rzeczywistego uruchomienia testów matematycznych i testów modelu.
        """),
        code(r"""
        rng = np.random.default_rng(0)
        random_pairs = [(rng.integers(0, 4, 60).astype(str), rng.integers(0, 2, 60).astype(str)) for _ in range(200)]


        def raises_value_error(action) -> bool:
            try:
                action()
            except ValueError:
                return True
            return False


        probe_model = ID3Classifier().fit(X_train_mc, y_train)
        unseen_sample = X_train_mc.iloc[0].copy()
        unseen_sample["odor"] = "q"                                   # kategoria spoza słownika i treningu
        root_path_ok = all(len({f for f, _ in path}) == len(path) for path in probe_model.node_paths().values())
        checks = pd.DataFrame([
            ("H([]) = 0 (udokumentowana konwencja)", entropy([]) == 0.0),
            ("H(['e', 'e', 'e']) = 0", entropy(["e", "e", "e"]) == 0.0),
            ("H(['e', 'p']) = 1 bit", entropy(["e", "p"]) == 1.0),
            ("0·log2(0) := 0 bez wywołania log2(0)", entropy_from_counts([5, 0, 0]) == 0.0),
            ("IG podziału idealnego = H(rodzica)", abs(information_gain(["a", "a", "b"], ["e", "e", "p"]) - entropy(["e", "e", "p"])) < 1e-12),
            ("0 ≤ IG ≤ H(S) na 200 losowych próbach",
             all(-1e-12 <= information_gain(x, t) <= entropy(t) + 1e-12 for x, t in random_pairs)),
            ("IG(odor) = I(odor; klasa) ze scikit-learn",
             abs(information_gain(X_all["odor"], y_all) - association.set_index("cecha").loc["odor", "I(A;Y) [bit]"]) < 1e-10),
            ("NaN bez preprocessingu → ValueError", raises_value_error(lambda: ID3Classifier().fit(X_train[["stalk-root"]], y_train))),
            ("nieznana kategoria → klasa większościowa ostatniego węzła",
             probe_model.explain_prediction(unseen_sample).used_fallback
             and probe_model.predict_one(unseen_sample) == probe_model.root_.majority_class),
            ("determinizm: dwa treningi → identyczne drzewo",
             ID3Classifier().fit(X_train_mc, y_train).to_dict() == probe_model.to_dict()),
            ("żaden atrybut nie powtarza się na ścieżce", root_path_ok),
            ("predict_one ≡ predict dla 50 rekordów",
             [probe_model.predict_one(r) for _, r in X_train_mc.head(50).iterrows()] == probe_model.predict(X_train_mc.head(50)).tolist()),
        ], columns=["kontrola", "wynik"])
        checks["wynik"] = checks["wynik"].map({True: "✅ OK", False: "❌ BŁĄD"})
        display(checks)
        assert (checks["wynik"] == "✅ OK").all(), "wbudowane kontrole nie przeszły"
        note(n_inline_checks=len(checks))

        tests_dir = PROJECT_ROOT / "tests"
        if tests_dir.is_dir() and importlib.util.find_spec("pytest") is not None:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider", "tests/test_entropy.py",
                 "tests/test_information_gain.py", "tests/test_id3.py"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, env={**os.environ, "ID3_QA_SKIP_REPORT": "1"})
            print("pytest (testy matematyczne i modelu):", completed.stdout.strip().splitlines()[-1])
            assert completed.returncode == 0, completed.stdout[-2000:]
        else:
            print("Folder tests/ niedostępny (np. Google Colab) — pełne wyniki: outputs/QA_REPORT.md w paczce projektu.")
        """),
        md(r"""
        **Interpretacja.** Wszystkie {{n_inline_checks}} wbudowanych kontroli przechodzi:

        - własności entropii i IG są zachowane, w tym obsługa $0\log_2 0$;
        - IG zgadza się z niezależną informacją wzajemną ze scikit-learn;
        - braki danych bez preprocessingu kończą się czytelnym `ValueError`;
        - nieznana kategoria przechodzi na klasę większościową węzła;
        - drzewo jest deterministyczne i nie powtarza atrybutów na ścieżce.

        Pełny zestaw testów obejmuje także warunki stopu, schemat danych, pipeline, protokół walidacji, wykresy,
        notebook i warsztat.
        """),
    ]


# ---------------------------------------------------------------------------
# 06. Tuning
# ---------------------------------------------------------------------------

def chapter_06() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-06"></a>
        # 06. Strojenie: modele próbne i dobór parametrów

        ## 6.1. Protokół
        Każdą z **36 konfiguracji** (3 strategie braków × 3 wartości `max_depth` × 2 × 2) trenujemy na tym samym zbiorze
        uczącym i oceniamy na tym samym zbiorze walidacyjnym. Funkcje strojenia (`run_experiment_grid`, `depth_curve`,
        `rank_configurations`) nie mają argumentu, przez który mógłby trafić do nich zbiór testowy. Test
        `tests/test_experiments.py` sprawdza dodatkowo, że celowe zepsucie etykiet testowych nie zmienia ani jednego wyniku.

        | Oś | Wartości | Rola |
        |---|---|---|
        | strategia braków | `missing_category`, `most_frequent`, `drop_stalk_root` | sposób traktowania `stalk-root` |
        | `max_depth` | 2, 4, `None` | 2: niedouczenie, 4: strefa przejściowa, `None`: pełny wzrost |
        | `min_samples_split` | 2, 20 | minimalna liczność węzła do podziału |
        | `min_information_gain` | 0.0, 0.01 | minimalny przyrost informacji podziału |

        **Reguła wyboru została zakodowana przed obejrzeniem wyników** (`SELECTION_RULE` w module `experiments`). Jest
        leksykograficzna, bo na Mushroom wiele modeli remisuje przy 100% trafności walidacyjnej:
        """),
        code(r"""
        display(selection_rule_frame())
        """),
        md(r"""
        Poziomy 6c–6e rozszerzają regułę ze specyfikacji do porządku całkowitego. Przy identycznym drzewie i metrykach
        wybieramy najbardziej zachowawcze ograniczenia wzrostu. To ta sama zasada, co preferencja mniejszego `max_depth`:
        walidacja nie daje żadnego argumentu za swobodniejszym wzrostem.

        > **„(Nie ??? w przypadku lasów)”: przycinanie a lasy losowe.** W lesie losowym (Breiman, 2001) każde drzewo rośnie
        > **bez przycinania** na próbie bootstrapowej, z losowym podzbiorem cech w węzłach. Pojedyncze drzewa mają małe
        > obciążenie i dużą wariancję, a głosowanie wielu słabo skorelowanych drzew tę wariancję redukuje. Do oceny
        > wystarcza estymacja **out-of-bag (OOB)** na obserwacjach niewylosowanych do danej próby. Pojedyncze drzewo ID3 nie
        > ma takiego mechanizmu, więc jego wariancję kontrolujemy bezpośrednio: ograniczamy wzrost (`max_depth`,
        > `min_samples_split`, `min_information_gain`, czyli *pre-pruning*) i dobieramy te ograniczenia na **wydzielonym
        > zbiorze walidacyjnym**, bo naturalnego odpowiednika OOB tu nie ma.

        ## 6.2. Wyniki 36 modeli próbnych
        """),
        code(r"""
        results = run_experiment_grid(X_train, y_train, X_val, y_val)      # zbiór testowy nie jest argumentem
        results.to_csv(OUTPUT_DIR / "experiment_results.csv", index=False)
        ranked = rank_configurations(results)
        best_config = select_best_configuration(results)
        top10 = ranked.head(10)[["rank", "config_id", "strategy", "max_depth_label", "min_samples_split",
                                 "min_information_gain", "train_accuracy", "val_accuracy", "val_macro_f1",
                                 "val_recall_p", "val_p_to_e", "leaf_count", "node_count", "depth", "fit_time_ms"]]
        display(top10.round({"train_accuracy": 4, "val_accuracy": 4, "val_macro_f1": 4, "val_recall_p": 4,
                             "fit_time_ms": 1}).set_index("rank"))
        summary36 = (results.groupby(["strategy", "max_depth_label"], sort=False)
                     .agg(konfiguracje=("config_id", "size"), val_accuracy=("val_accuracy", "mean"),
                          val_p_to_e=("val_p_to_e", "max"), liście=("leaf_count", "mean"),
                          różne_drzewa=("tree_signature", "nunique")))
        display(summary36)
        perfect = results[results["val_accuracy"] == 1.0]
        shallow = results[results["max_depth_label"] == "2"]
        note(n_configs=len(results), n_perfect=len(perfect), n_shallow=len(shallow),
             shallow_acc=float(shallow["val_accuracy"].iloc[0]), shallow_p_to_e=int(shallow["val_p_to_e"].iloc[0]),
             shallow_leaves=int(shallow["leaf_count"].iloc[0]), n_signatures=int(results["tree_signature"].nunique()),
             perfect_leaves=int(perfect["leaf_count"].iloc[0]), perfect_nodes=int(perfect["node_count"].iloc[0]),
             perfect_depth=int(perfect["depth"].iloc[0]))
        """),
        code(r"""
        show(plot_experiment_heatmap(results), "06_experiment_heatmap")
        """),
        md(r"""
        **Interpretacja.**

        - O jakości decyduje **wyłącznie `max_depth`**. Wszystkie {{n_shallow}} konfiguracji z `max_depth=2` mają
          {{shallow_acc:.2%}} trafności i {{shallow_p_to_e}} błędów krytycznych `p → e`: drzewo o {{shallow_leaves}} liściach
          jest niedouczone.
        - Pozostałe konfiguracje (`max_depth` 4 lub `None`, łącznie {{n_perfect}}) osiągają 100%.
        - Progi `min_samples_split` i `min_information_gain` niczego nie zmieniają: na Mushroom podziały są albo bardzo
          informatywne, albo w ogóle niepotrzebne.
        - Strategia braków też nie zmienia wyników. Wszystkie 36 modeli to tylko {{n_signatures}} różne drzewa;
          sygnatura struktury jest liczona z atrybutów, gałęzi i klas liści.

        ## 6.3. Krzywa walidacyjna dla `max_depth` = 1…6, `None`
        Pozostałe parametry mają wartości domyślne (`min_samples_split=2`, `min_information_gain=0`). Krzywą liczymy dla
        każdej strategii braków.
        """),
        code(r"""
        curve = depth_curve(X_train, y_train, X_val, y_val)
        identical_curves = bool(curve.groupby("max_depth_label")["tree_signature"].nunique().eq(1).all())
        show(plot_depth_curve(curve, "missing_category", identical_curves), "06_depth_curve")
        mc_curve = curve[curve["strategy"] == "missing_category"].set_index("max_depth_label")
        display(mc_curve[["train_accuracy", "val_accuracy", "val_macro_f1", "val_p_to_e", "val_e_to_p", "depth",
                          "leaf_count", "node_count"]])
        first_perfect = mc_curve.index[mc_curve["val_accuracy"] >= 1.0][0]
        note(curves_identical=identical_curves, d1_acc=mc_curve.loc["1", "val_accuracy"],
             d1_p_to_e=int(mc_curve.loc["1", "val_p_to_e"]), d2_acc=mc_curve.loc["2", "val_accuracy"],
             d2_p_to_e=int(mc_curve.loc["2", "val_p_to_e"]), d3_acc=mc_curve.loc["3", "val_accuracy"],
             d3_p_to_e=int(mc_curve.loc["3", "val_p_to_e"]), d3_e_to_p=int(mc_curve.loc["3", "val_e_to_p"]),
             first_perfect_depth=first_perfect, full_depth=int(mc_curve.loc["None", "depth"]),
             d1_train_acc=mc_curve.loc["1", "train_accuracy"])
        """),
        md(r"""
        **Interpretacja.**

        | `max_depth` | Drzewo | Trafność walidacyjna | Błędy |
        |---|---|---|---|
        | 1 | sam `odor` | {{d1_acc:.2%}} | {{d1_p_to_e}} × `p → e`: trujące z zapachem `none` uznane za jadalne |
        | 2 | + `spore-print-color` | {{d2_acc:.2%}} | {{d2_p_to_e}} × `p → e` |
        | 3 | | {{d3_acc:.2%}} | {{d3_p_to_e}} × `p → e`, {{d3_e_to_p}} × `e → p` |
        | od {{first_perfect_depth}} | pełne drzewo | 100% | brak |

        Głębszy limit nic nie zmienia, bo nieograniczone drzewo kończy się naturalnie na głębokości {{full_depth}}.
        Krzywe uczenia i walidacji idą razem (przy `max_depth=1` walidacja ma nawet {{d1_acc:.2%}} wobec
        {{d1_train_acc:.2%}} na treningu). Nie ma więc **przeuczenia**, jest tylko **niedouczenie** płytkich drzew.
        Krzywe są identyczne dla trzech strategii braków.
        """),
        code(r"""
        show(plot_train_vs_validation(results), "06_train_vs_validation")
        """),
        md(r"""
        **Interpretacja.** Każdy z 36 punktów leży na przekątnej `train = walidacja` lub tuż przy niej. Model
        z mniejszą liczbą liści przegrywa jednakowo na obu zbiorach, czyli brakuje mu pojemności, a nie zdolności
        uogólniania. To typowy obraz zbioru, który jest **deterministycznie separowalny**: wystarczająco głębokie drzewo
        odtwarza regułę generującą etykiety.

        ## 6.4. Porównanie strategii braków: dlaczego wyniki są identyczne?
        """),
        code(r"""
        signatures = results.pivot_table(index=["max_depth_label", "min_samples_split", "min_information_gain"],
                                         columns="strategy", values="tree_signature", aggfunc="first")
        strategies_identical = bool((signatures.nunique(axis=1) == 1).all())
        reference_model = ID3Classifier(class_priority=SAFETY_CLASS_PRIORITY).fit(X_train_mc, y_train)
        gain_report = feature_gain_report(X_train, y_train, reference_model)
        display(gain_report)
        show(plot_strategy_gains(gain_report), "06_strategy_comparison")

        tie_rows = []
        for node in reference_model.iter_nodes():
            if not node.is_leaf:
                tied = sorted(f for f, g in node.candidate_gains.items() if g >= node.information_gain - GAIN_TOLERANCE)
                tie_rows.append({"węzeł": node.node_id, "n": node.n_samples, "wybrany atrybut": node.feature,
                                 "IG": round(node.information_gain, 4), "remisujące atrybuty": ", ".join(tied),
                                 "liczba remisujących": len(tied)})
        ties_in_tree = pd.DataFrame(tie_rows)
        display(ties_in_tree)
        biggest_tie = ties_in_tree.sort_values("liczba remisujących", ascending=False).iloc[0]
        mc_ranks = gain_report["rank_missing_category"]
        note(strategies_identical=strategies_identical, stalk_best_rank=int(mc_ranks.min()),
             tie_node_n=int(biggest_tie["n"]), tie_count=int(biggest_tie["liczba remisujących"]),
             tie_gain=float(biggest_tie["IG"]), tie_winner=biggest_tie["wybrany atrybut"],
             tie_features=biggest_tie["remisujące atrybuty"],
             tie_node_label=gain_report.loc[gain_report["n"] == biggest_tie["n"], "node"].iloc[0])
        """),
        md(r"""
        **Interpretacja.** Dla każdej z 12 kombinacji hiperparametrów trzy strategie budują **to samo drzewo**
        (`strategies_identical` = {{strategies_identical}}). Powód widać na wykresie: `stalk-root` w żadnym węźle nie
        wygrywa. W korzeniu i na dwóch kolejnych poziomach jego IG jest kilkukrotnie mniejszy niż IG atrybutu wybranego,
        niezależnie od strategii; jego najlepsza pozycja w rankingu kandydatów to {{stalk_best_rank}}. Skoro `stalk-root`
        nie jest testowany, jego braki nie mają wpływu na model.

        Najciekawszy jest węzeł **{{tie_node_label}}** ({{tie_node_n}} obserwacji). Tu aż **{{tie_count}} atrybutów**
        dzieli węzeł idealnie z identycznym IG = {{tie_gain:.4f}}: {{tie_features}}. W wariancie `missing_category` jest
        wśród nich `stalk-root` (kategoria `__MISSING__` pokrywa się tu z grzybami trującymi). O wyborze `{{tie_winner}}`
        przesądziła wyłącznie **deterministyczna reguła remisów** (alfabetycznie). Reguła remisów nie jest więc detalem
        kosmetycznym, bo inna kolejność dałaby inne, choć równie trafne drzewo. Przy równych wynikach preferujemy
        `missing_category`. Zachowuje pełną informację, także o samym fakcie braku, który według 3.2 jest skorelowany
        z klasą, i nie wymaga zgadywania wartości.

        ## 6.5. Remisy i wybór konfiguracji finalnej
        """),
        code(r"""
        ties = metric_tie_groups(ranked)
        display(ties)
        winner = ranked.iloc[0]
        display(winner[["config_id", "strategy", "max_depth_label", "min_samples_split", "min_information_gain",
                        "val_accuracy", "val_macro_f1", "val_p_to_e", "leaf_count", "node_count", "depth"]].to_frame("wybrana"))
        note(selected_config=best_config.config_id, selected_strategy=best_config.strategy,
             selected_max_depth=depth_label(best_config.max_depth), selected_min_split=best_config.min_samples_split,
             selected_min_ig=best_config.min_information_gain, top_group_size=int(ties.iloc[0]["liczba konfiguracji"]),
             top_group_trees=int(ties.iloc[0]["różne drzewa"]), selected_val_acc=float(winner["val_accuracy"]),
             selected_leaves=int(winner["leaf_count"]), selected_signature=winner["tree_signature"])
        """),
        md(r"""
        **Wybór: `{{selected_config}}`** (`strategy={{selected_strategy}}`, `max_depth={{selected_max_depth}}`,
        `min_samples_split={{selected_min_split}}`, `min_information_gain={{selected_min_ig}}`). Rozstrzygnięcie
        przebiegło tak:

        - Poziomy 1–5 reguły (0 błędów `p → e`, macro-F1 = 1, {{selected_leaves}} liście, {{perfect_nodes}} węzłów,
          głębokość {{perfect_depth}}) remisują dla **{{top_group_size}} konfiguracji**. Wszystkie budują to samo drzewo
          (różnych drzew: {{top_group_trees}}).
        - Poziom 6a wybrał `missing_category`, 6b mniejsze `max_depth = 4` zamiast `None`, a 6c–6d silniejsze progi
          `min_samples_split` i `min_information_gain`.

        Wybór nie poprawia trafności, bo wszystkie zwycięskie konfiguracje są równoważne. Daje jednoznaczny, opisany
        z góry i powtarzalny model.

        > Zero błędów `p → e` na tym zbiorze **nie oznacza** bezpieczeństwa grzybobrania. Oznacza tylko, że drzewo
        > odtwarza regułę, według której przewodnik przypisał etykiety 23 gatunkom.
        """),
    ]


# ---------------------------------------------------------------------------
# 07. Final model
# ---------------------------------------------------------------------------

def chapter_07() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-07"></a>
        # 07. Budowa modelu finalnego

        Zgodnie z protokołem łączymy zbiór uczący i walidacyjny (60% + 20%). Na połączonych danych **od nowa**
        dopasowujemy preprocessing, czyli cechy stałe i kategorie, a potem od podstaw trenujemy ID3 w wybranej
        konfiguracji. Zbiór testowy pozostaje zapieczętowany.
        """),
        code(r"""
        train_val_idx = split.train_validation()
        X_trval, y_trval = X_all.loc[train_val_idx], y_all.loc[train_val_idx]
        final_pre, final_model = fit_final_model(X_trval, y_trval, best_config)
        summary = final_model.summary()
        display(pd.DataFrame({"wartość": {
            "konfiguracja": best_config.label(), "obserwacje (train+val)": summary["n_samples_fit"],
            "cechy wejściowe": summary["n_features_in"], "węzły": summary["node_count"], "liście": summary["leaf_count"],
            "głębokość": summary["depth"], "użyte atrybuty": ", ".join(summary["used_features"]),
            "remis klas w liściu → klasa": final_model.class_priority[0], "test": repr(test_vault)}}))
        final_model.save_json(MODEL_DIR / "final_tree.json")
        final_config = {
            "selected_configuration": best_config.__dict__, "selection_rule": [r[2] for r in SELECTION_RULE],
            "preprocessing": final_pre.summary(), "split": {"random_state": split.random_state, "train": len(split.train),
                                                            "validation": len(split.validation), "test": len(split.test)},
            "tree": {k: summary[k] for k in ("node_count", "leaf_count", "depth", "used_features")},
            "data": {"source": UCI_DATASET_URL, "doi": UCI_DOI, "sha256": file_sha256(DATA_ZIP)},
            "environment": ENVIRONMENT,
        }
        (MODEL_DIR / "final_config.json").write_text(json.dumps(final_config, indent=2, ensure_ascii=False, default=str),
                                                    encoding="utf-8")
        assert ID3Classifier.load_json(MODEL_DIR / "final_tree.json").structure_signature() == final_model.structure_signature()
        importance = final_model.feature_importance_
        note(final_n=summary["n_samples_fit"], final_nodes=summary["node_count"], final_leaves=summary["leaf_count"],
             final_depth=summary["depth"], final_n_used=len(summary["used_features"]),
             final_used=", ".join(summary["used_features"]), final_root_ig=final_model.root_.information_gain,
             final_same_as_validated=final_model.structure_signature() == winner["tree_signature"],
             imp_first=importance.index[0], imp_first_value=float(importance.iloc[0]),
             imp_second=importance.index[1], imp_second_value=float(importance.iloc[1]),
             final_impure_leaves=sum(1 for n in final_model.iter_nodes() if n.is_leaf and n.entropy > 0),
             importance_bits_total=float(final_model.feature_importance_bits_.sum()),
             H_trval=entropy(y_trval))
        """),
        code(r"""
        show(plot_feature_importance(importance, "Ważność cech modelu finalnego (ważony sumaryczny IG)"),
             "07_feature_importance")
        importance_table = pd.DataFrame({"ważność [%]": (importance * 100).round(2),
                                         "IG ważony [bit]": final_model.feature_importance_bits_.round(4)})
        display(importance_table[importance_table["ważność [%]"] > 0])
        print(f"Atrybuty o zerowej ważności (nieużyte w drzewie): {int((importance == 0).sum())} z {len(importance)}")
        """),
        md(r"""
        **Charakterystyka.** Drzewo finalne wytrenowane na {{final_n}} obserwacjach ma {{final_nodes}} węzłów,
        {{final_leaves}} liście i głębokość {{final_depth}}. Testuje tylko {{final_n_used}} z 21 atrybutów:
        {{final_used}}. Wszystkie liście są czyste (liście mieszane: {{final_impure_leaves}}). Struktura jest identyczna
        z drzewem zweryfikowanym na walidacji (porównanie sygnatur struktury: {{final_same_as_validated}}), więc dołożenie danych walidacyjnych tylko
        zagęściło liczności w liściach.

        **Ważność cech** definiujemy jako $\mathrm{imp}(A) = \sum_{t:\,A_t = A} \frac{n_t}{N}\, IG(t)$, czyli ważony
        przyrost informacji ze wszystkich węzłów testujących $A$, normalizowany do sumy 1. `{{imp_first}}` odpowiada za
        {{imp_first_value:.1%}} całej redukcji entropii, `{{imp_second}}` za {{imp_second_value:.1%}}. Suma nieznormalizowanych
        ważności ({{importance_bits_total:.4f}} bita) równa się entropii zbioru $H$ = {{H_trval:.4f}}. Tak musi być, gdy
        wszystkie liście są czyste, i jest to dobra kontrola poprawności implementacji.

        ### Pełna lista reguł (jedna reguła na liść)
        """),
        code(r"""
        for i, rule in enumerate(final_model.rules_text(value_formatter=expand_value), start=1):
            print(f"{i:>2}. {rule}")
        """),
    ]


# ---------------------------------------------------------------------------
# 08. Tree visualisation
# ---------------------------------------------------------------------------

def chapter_08() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-08"></a>
        # 08. Graficzne przedstawienie drzewa

        Każdy węzeł pokazuje testowany atrybut (albo klasę liścia), liczbę obserwacji, rozkład klas (liczby i pasek
        proporcji) oraz entropię. Krawędzie opisują wartości kategorii z rozwiniętymi nazwami. Widok prezentacyjny łączy
        rodzeństwo będące **czystymi liśćmi tej samej klasy** w jedno pole (np. 6 zapachów → trujący). Model się nie
        zmienia, a schemat mieści się na ekranie projektora.
        """),
        code(r"""
        show(plot_tree(final_model, group_pure_leaves=True,
                       title="Drzewo finalne ID3 — widok prezentacyjny (czyste liście tej samej klasy zgrupowane)"),
             "08_tree_presentation", formats=("png", "svg"))
        """),
        md(r"""
        **Interpretacja.** Drzewo czyta się jak klucz do oznaczania:

        1. Zapach migdałowy lub anyżowy → jadalny; każdy z sześciu nieprzyjemnych zapachów → trujący.
        2. Zapach `none` → rozstrzyga kolor wysypu zarodników: zielony → trujący, większość kolorów → jadalny.
        3. Biały wysyp → decyduje siedlisko, a w lesie i na liściach jeszcze szerokość blaszek albo kolor kapelusza.

        Najrzadsze ścieżki (najmniejsze liście mają {{min_leaf_support}} obserwacji) obsługują najtrudniejsze przypadki,
        które zarazem są najsłabiej udokumentowane w danych.

        ### Powiększenie: poddrzewo `odor = none (n)`, każda gałąź osobno
        """),
        code(r"""
        show(plot_tree(final_model, subtree_path=[("odor", "n")], orientation="LR",
                       title="Poddrzewo odor = none (n)"),
             "08_tree_odor_none", formats=("png", "svg"))
        """),
        md(r"""
        ### Pełne drzewo do pliku: Graphviz (DOT) lub fallback matplotlib
        Drzewo zapisujemy w formacie **DOT**. Jeśli w systemie jest program `dot` (Graphviz; w Google Colab jest
        preinstalowany), SVG i PNG renderuje Graphviz. W przeciwnym razie `render_tree` automatycznie rysuje ten sam
        pełny schemat w matplotlib (układ od lewej do prawej, każda gałąź osobno).
        """),
        code(r"""
        rendered = render_tree(final_model, FIG_DIR / "final_tree", title="Pełne drzewo ID3 — model finalny")
        shutil.copy2(rendered["dot"], MODEL_DIR / "final_tree.dot")
        print(f"Renderer: {rendered['backend']}  ·  pliki: {', '.join(p.name for p in rendered['files'])}  ·  "
              f"DOT: {(MODEL_DIR / 'final_tree.dot').relative_to(PROJECT_ROOT)}")
        display(Image(filename=str(FIG_DIR / "final_tree.png"), width=820))
        """),
        md(r"""
        ### Widok tekstowy (pomocniczy)
        Wydruk ASCII służy szybkiemu podglądowi w konsoli. Nie zastępuje diagramu powyżej.
        """),
        code(r"""
        print(export_text(final_model, value_formatter=expand_value))
        """),
        md(r"""
        ## 8.1. Biblioteki generujące grafikę drzew decyzyjnych

        | Biblioteka | Język | Jak rysuje | Czy narysuje nasze drzewo ID3 (wielogałęziowe)? | Ograniczenia |
        |---|---|---|---|---|
        | **Graphviz** (`dot`) + `python-graphviz` / `PyGraphviz` | C (program), interfejsy Python/R | automatyczny układ grafu z opisu DOT → SVG/PNG/PDF | **tak**: eksportujemy DOT (`tree_to_dot`) | wymaga systemowego programu `dot` |
        | **Matplotlib**, własny renderer (`plot_tree`) | Python | prostokąty, kolory klas, opisane krawędzie; własny układ drzewa | **tak**: fallback bez zależności systemowych | układ trzeba zaprogramować samodzielnie |
        | **NetworkX + Matplotlib** | Python | graf węzłów i krawędzi | tak, po przepisaniu drzewa na graf | do hierarchii potrzebny `graphviz_layout` (pygraphviz/pydot) lub własne pozycje |
        | **Plotly** (`treemap`, `sunburst`, `icicle`) | Python/JS | interaktywne wykresy hierarchii w HTML | częściowo: hierarchia liczności, a nie klasyczny schemat węzłów | eksport statyczny wymaga dodatku `kaleido` |
        | **dtreeviz** | Python | bogate wizualizacje z rozkładami w węzłach | nie bezpośrednio: obsługuje drzewa scikit-learn, XGBoost, LightGBM, Spark | własne drzewo wymaga adaptera |
        | **scikit-learn** `plot_tree` / `export_graphviz` | Python | rysunek drzewa CART | **nie**: tylko obiekty `DecisionTreeClassifier` (drzewa binarne) | nie obsługuje drzew wielogałęziowych |
        | R: `rpart.plot`, `partykit`, `DiagrammeR` | R | wykresy drzew `rpart`/`party`; diagramy Graphviz | `DiagrammeR`: tak (DOT); pozostałe: tylko własne obiekty | związane z pakietami modelującymi R |

        W projekcie: **DOT/Graphviz** jako format wymiany (`final_tree.dot`) i renderer, gdy jest dostępny, oraz **własny
        renderer matplotlib** jako fallback i do widoków prezentacyjnych (grupowanie liści, wyróżnianie ścieżki predykcji).
        """),
    ]


# ---------------------------------------------------------------------------
# 09. Evaluation
# ---------------------------------------------------------------------------

def chapter_09() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-09"></a>
        # 09. Ocena modelu na zbiorze testowym

        Konfiguracja, preprocessing i drzewo są już ustalone. Teraz **jeden raz** otwieramy sejf ze zbiorem testowym,
        transformujemy go preprocesorem dopasowanym na train+val i oceniamy model. Po tym kroku nie wolno już niczego
        stroić. Ponowne `unlock()` zgłosiłoby wyjątek.
        """),
        code(r"""
        X_test, y_test = test_vault.unlock("jednorazowa ocena modelu finalnego " + best_config.config_id)
        display(test_vault.log_frame())
        X_test_ready = final_pre.transform(X_test)
        test_pred = final_model.predict(X_test_ready)
        test_details = final_model.predict_details(X_test_ready)
        assert (test_details["prediction"].to_numpy() == test_pred).all()   # predykcje i ścieżki z tego samego modelu
        test_metrics = classification_metrics(y_test, test_pred)
        cm = confusion_matrix_frame(y_test, test_pred, labels=CLASS_ORDER)
        display(classification_report_frame(y_test, test_pred).round(4))
        show(plot_confusion_matrix(cm, f"Macierz pomyłek — zbiór testowy (n = {len(y_test)}), kolejność klas: e, p"),
             "09_confusion_matrix")
        predictions = prediction_table(y_test, test_pred, index=X_test.index).join(test_details[["final_node_id",
                                                                                                 "path_length"]])
        predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index_label="row_index")
        note(test_n=test_metrics["n"], test_acc=test_metrics["accuracy"], test_macro_f1=test_metrics["macro_f1"],
             test_weighted_f1=test_metrics["weighted_f1"], test_errors=test_metrics["n_errors"],
             test_p_to_e=test_metrics["p_to_e"], test_e_to_p=test_metrics["e_to_p"],
             test_p_to_e_rate=test_metrics["p_to_e_rate_of_p"], test_recall_p=test_metrics["recall_p"],
             test_precision_p=test_metrics["precision_p"], test_recall_e=test_metrics["recall_e"],
             test_fallbacks=int(test_details["used_fallback"].sum()),
             cm_ee=int(cm.loc["e", "e"]), cm_ep=int(cm.loc["e", "p"]), cm_pe=int(cm.loc["p", "e"]), cm_pp=int(cm.loc["p", "p"]))
        """),
        md(r"""
        **Wyniki na odłożonym zbiorze testowym** ({{test_n}} obserwacji):

        | Miara | Wartość |
        |---|---|
        | accuracy | {{test_acc:.4f}} |
        | macro-F1 / weighted-F1 | {{test_macro_f1:.4f}} / {{test_weighted_f1:.4f}} |
        | precision / recall klasy `p` | {{test_precision_p:.4f}} / {{test_recall_p:.4f}} |
        | recall klasy `e` | {{test_recall_e:.4f}} |
        | liczba błędów (łącznie) | {{test_errors}} |
        | błędy krytyczne `p → e` (udział wśród trujących) | {{test_p_to_e}} ({{test_p_to_e_rate:.2%}}) |
        | błędy `e → p` | {{test_e_to_p}} |
        | predykcje z fallbackiem (nieznana kategoria) | {{test_fallbacks}} |

        **Macierz pomyłek** (wiersze: klasa rzeczywista, kolumny: przewidziana, kolejność `e`, `p`):
        `e → e` = {{cm_ee}}, `e → p` = {{cm_ep}}, `p → e` = {{cm_pe}}, `p → p` = {{cm_pp}}.

        **Interpretacja.** Model sklasyfikował poprawnie wszystkie {{test_n}} obserwacji testowych. Liczba błędów wynosi
        {{test_errors}}, w tym {{test_p_to_e}} krytycznych `p → e`. Wynik potwierdza ocenę walidacyjną i jest zgodny z naturą
        danych: etykiety w Mushroom wynikają z deterministycznych reguł przewodnika, a pełne drzewo ID3 te reguły odtwarza.
        Wynik 100% nie świadczy tu o przecieku. Sprawdziliśmy to wprost:

        - podzbiory są rozłączne (4.3);
        - test nie zawiera wektorów cech obecnych w danych uczących (4.5);
        - preprocessing dopasowano bez zbioru testowego;
        - sejf otwarto dokładnie raz (log powyżej).

        Żadna predykcja nie wymagała fallbacku dla nieznanej kategorii (liczba takich przypadków: {{test_fallbacks}}).

        ## 9.1. Analiza błędów `p → e` i strefy ryzyka
        Na teście nie ma błędnych klasyfikacji (liczba: {{test_errors}}), więc nie ma czego analizować. Nie tworzymy
        sztucznych błędów. Wskazujemy natomiast, gdzie model jest najmniej pewny: przykłady trafiające do liści, które
        w treningu oparto na bardzo małej liczbie obserwacji. Tam w pierwszej kolejności pojawiłyby się pomyłki
        na nowych danych.
        """),
        code(r"""
        leaf_support = {node.node_id: node.n_samples for node in final_model.iter_nodes() if node.is_leaf}
        risk = (predictions.assign(n_liścia=predictions["final_node_id"].map(leaf_support))
                .groupby("n_liścia").agg(przykłady_testowe=("predykcja", "size"),
                                         trujące=("klasa rzeczywista", lambda s: int((s == "p").sum())))
                .sort_index().head(6))
        display(risk)
        errors = predictions[predictions["wynik"] != "poprawna"]
        print("Błędne klasyfikacje na teście:", len(errors))
        note(risk_small_leaf_tests=int(risk.loc[risk.index <= 10, "przykłady_testowe"].sum()),
             min_leaf_support=int(min(leaf_support.values())))
        """),
        md(r"""
        **Interpretacja.** Najmniejsze liście mają w treningu tylko {{min_leaf_support}} obserwacji. Liczba grzybów
        testowych w liściach opartych na ≤ 10 przykładach: {{risk_small_leaf_tests}}. Te decyzje (np. `cap-color = white` przy
        siedlisku `leaves`) mają najsłabsze pokrycie w danych i to tam w pierwszej kolejności należałoby się spodziewać
        błędów na nowych danych spoza przewodnika.

        ## 9.2. Ścieżki predykcji dla rzeczywistych przykładów testowych
        Wybieramy deterministycznie 5 przykładów: najpierw ewentualne błędy, potem po jednym przykładzie z każdej
        głębokości ścieżki, a na koniec najgłębsze inne liście. Każda predykcja ma pełne uzasadnienie
        (`explain_prediction`) z rozwiniętymi nazwami cech i kategorii.
        """),
        code(r"""
        chosen = list(errors.index)
        for depth in sorted(test_details["path_length"].unique()):
            candidates = test_details.index[(test_details["path_length"] == depth) & ~test_details.index.isin(chosen)]
            chosen.append(candidates[0])
        extra = (test_details.drop(index=chosen).sort_values(["path_length", "final_node_id"], ascending=[False, True])
                 .drop_duplicates("final_node_id"))
        chosen = (chosen + [i for i in extra.index if test_details.loc[i, "final_node_id"] not in
                            set(test_details.loc[chosen, "final_node_id"])])[:5]
        for i, idx in enumerate(chosen, start=1):
            explanation = final_model.explain_prediction(X_test_ready.loc[idx])
            verdict = "poprawna" if explanation.prediction == y_test.loc[idx] else "BŁĘDNA"
            print(f"Przykład {i} — rekord #{idx}: klasa rzeczywista {describe_class(y_test.loc[idx])}, "
                  f"predykcja {describe_class(explanation.prediction)} ({verdict})")
            print(explanation.to_text(value_formatter=expand_value), end="\n\n")
        deepest = max(chosen, key=lambda i: test_details.loc[i, "path_length"])
        path_ids = [s.node_id for s in final_model.explain_prediction(X_test_ready.loc[deepest]).steps]
        path_ids.append(int(test_details.loc[deepest, "final_node_id"]))
        note(n_examples=len(chosen), deepest_example=int(deepest), deepest_length=int(test_details.loc[deepest, "path_length"]))
        """),
        code(r"""
        show(plot_tree(final_model, group_pure_leaves=True, highlight_node_ids=path_ids,
                       title=f"Ścieżka decyzji dla rekordu testowego #{deepest}"), "09_prediction_path")
        """),
        md(r"""
        **Interpretacja.** Pięć ścieżek pokazuje, jak drzewo „rozumuje”. Grzyb z charakterystycznym zapachem jest
        rozstrzygany jednym testem. Grzyb bezwonny wymaga sprawdzenia wysypu zarodników, a przy białym wysypie także
        siedliska i jeszcze jednej cechy. Najdłuższa ścieżka (rekord #{{deepest_example}}, {{deepest_length}} testy)
        przechodzi przez wyróżnione węzły na schemacie. Każdy krok podaje liczność i rozkład klas węzła, więc widać,
        na ilu przykładach z treningu opiera się decyzja.

        > ⚠️ **Ostrzeżenie.** Te predykcje to ilustracja działania algorytmu na danych edukacyjnych. Model zna tylko
        > 23 gatunki opisane w jednym przewodniku i cechy zakodowane przez człowieka. Nie nadaje się do oceny jadalności
        > prawdziwych grzybów.
        """),
    ]


# ---------------------------------------------------------------------------
# 10. Workshop
# ---------------------------------------------------------------------------

def chapter_10(workshop_csv: str) -> list[dict]:
    preview = code(r"""
        WORKSHOP_LINK = None   # ← po publikacji: link do folderu Google Drive / repozytorium dostępnego tylko dla grupy
        WORKSHOP_CSV = '''
        __CSV__
        '''  # kopia group_exercise/delivery_data.csv wstawiona przez scripts/build_notebook.py
        delivery = pd.read_csv(io.StringIO(WORKSHOP_CSV.strip()))
        print("Link dla grupy:", WORKSHOP_LINK or "jeszcze nieopublikowany — patrz group_exercise/README.md")
        print(f"Dane warsztatu: {delivery.shape[0]} przesyłek × {delivery.shape[1]} kolumn; braki: "
              f"{ {c: int(n) for c, n in delivery.isna().sum().items() if n} }")
        display(delivery.head(8))
        """)
    preview["source"] = preview["source"].replace("__CSV__", workshop_csv.strip())
    return [
        md(r"""
        <a id="rozdzial-10"></a>
        # 10. Warsztat dla grupy: „Delivery Decision — czy przesyłka dotrze na czas?”

        **Zadanie:** na małym, syntetycznym zbiorze 28 przesyłek uczestnicy samodzielnie wykonują pełny cykl ID3:

        1. wykrywają i uzupełniają braki danych;
        2. liczą entropię i przyrost informacji, a na tej podstawie wybierają korzeń;
        3. budują gotowym mini-ID3 drzewo i uzasadniają jego predykcję dla nowej przesyłki.

        Dane są **inne niż Mushroom**: 4 nominalne cechy (`distance`, `weather`, `workload`, `shipping`), etykieta
        `on_time` / `delayed`, 3 kontrolowane braki w dwóch kolumnach.

        | Minuty | Uczestnik wykonuje | Weryfikowalny wynik |
        |---|---|---|
        | 0–4 | otwiera notebook w Colab, czyta opis i dane | wczytana tabela i etykieta celu |
        | 4–9 | **TODO 1:** wykrywa i uzupełnia braki dominantą | tabela „braki przed / po” |
        | 9–16 | **TODO 2–3:** entropia, IG, wybór korzenia; mini-ID3 | ranking IG + mini-drzewo |
        | 16–20 | **TODO 4:** predykcja nowej przesyłki i uzasadnienie ścieżką | predykcja + 1–2 zdania |

        **Jak pracować:**

        - Na zajęciach notebook otwiera się w **Google Colab**: *Plik → Zapisz kopię na Dysku*, potem *Uruchom wszystko*.
          Działa też lokalnie w JupyterLab.
        - Notebook jest samowystarczalny: dane są wbudowane, nie trzeba montować cudzego Dysku ani pobierać repozytorium.
        - Każde TODO ma automatyczną kontrolę z podpowiedzią. Niedokończone zadanie nie blokuje kolejnych kroków.
        - **W domu, bez limitu czasu:** te same materiały, a do tego opcjonalny bonus (porównanie strategii braków
          i wpływ na wybór korzenia). Bonus nie jest potrzebny do zaliczenia scenariusza podstawowego.

        **Dostęp tylko dla grupy:** materiały uczestnika (`submission/ID3_Group_Workshop_Students.zip`: notebook, CSV,
        README) publikujemy w folderze Google Drive z dostępem **„Ograniczony”**, udostępnionym wyłącznie adresom e-mail
        osób z grupy. Alternatywą jest prywatne repozytorium GitHub z zaproszonymi członkami grupy. Instrukcja krok po
        kroku: `group_exercise/README.md`. **Materiały nie zostały jeszcze opublikowane.** Link należy wpisać poniżej
        po utworzeniu folderu. Rozwiązanie i klucz odpowiedzi są w osobnej, prywatnej paczce prowadzącego warsztat
        i nie trafiają do grupy.
        """),
        preview,
        md(r"""
        Powyżej jest tylko podgląd danych, bez obliczeń. Wyniki zadania (ranking IG, korzeń, predykcja) uczestnicy
        uzyskują sami w notebooku warsztatowym.
        """),
    ]


# ---------------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------------

def chapter_11() -> list[dict]:
    return [
        md(r"""
        <a id="rozdzial-11"></a>
        # 11. Podsumowanie

        ### Wnioski
        1. **Własny ID3** (entropia, IG, podziały wielogałęziowe, deterministyczne remisy, fallback dla nieznanych
           kategorii, wyjaśnienia ścieżek) spełnia testy matematyczne i zachowuje się zgodnie z pseudokodem. Na Mushroom
           wybiera w korzeniu `odor` (IG = {{ig_odor_train:.3f}} bita na zbiorze uczącym).
        2. **Dane są czyste:** nie ma duplikatów, wartości spoza słownika ani sprzecznych etykiet. Wymagają tylko obsługi
           {{n_missing_total}} braków w `stalk-root` (nielosowych: {{miss_e_share:.1%}} u jadalnych wobec
           {{miss_p_share:.1%}} u trujących) i usunięcia stałej cechy `veil-type`. Balansowanie nie było potrzebne.
        3. **Strojenie:** jakość zależy wyłącznie od głębokości. `max_depth=2` daje {{shallow_acc:.2%}} i {{shallow_p_to_e}}
           błędów `p → e` na walidacji, a od głębokości {{first_perfect_depth}} jest 100%. Trzy strategie braków budują
           identyczne drzewa, bo `stalk-root` nigdzie nie wygrywa. W jednym węźle remisuje jednak {{tie_count}} atrybutów
           i o kształcie drzewa przesądza reguła remisów.
        4. **Model finalny** `{{selected_config}}` ({{final_nodes}} węzłów, {{final_leaves}} liście, głębokość {{final_depth}},
           {{final_n_used}} atrybutów) osiągnął na jednokrotnie użytym zbiorze testowym accuracy {{test_acc:.4f}}, macro-F1
           {{test_macro_f1:.4f}} i {{test_p_to_e}} błędów krytycznych `p → e`.

        ### Ograniczenia
        - Mushroom jest zbiorem **syntetycznym i deterministycznym**: opisuje 23 gatunki z jednego przewodnika, a etykiety
          wynikają z reguł. Wysoka trafność nie przenosi się na realne okazy, wątpliwe kategorie cech ani nowe gatunki.
        - Klasa `p` łączy grzyby trujące i „o nieznanej jadalności”, więc model nie rozróżnia stopnia zagrożenia.
        - Część liści opiera się na kilku obserwacjach (minimum {{min_leaf_support}}). Te reguły są najmniej pewne.
        - Pojedynczy podział 60/20/20 nie mierzy zmienności wyników. Przy wynikach na sufit nie zmieniłoby to decyzji,
          ale na trudniejszych danych warto użyć walidacji krzyżowej na części train+val.

        ### Reprodukowalność
        - Ziarno `random_state=42` (podział), deterministyczny ID3 (sortowanie kategorii, alfabetyczne remisy), stałe
          parametry renderowania wykresów.
        - Wyniki 36 modeli mają stały **odcisk SHA-256** (kolumny deterministyczne, bez czasu uczenia), a drzewo ma stałą
          sygnaturę struktury.
        - Artefakty: `experiment_results.csv`, `split_indices.csv`, `test_predictions.csv`, `final_model/final_tree.json`
          i `.dot`, `final_config.json`, `run_summary.json`, wykresy w `outputs/figures/`.
        - Komendy kontroli: `python -m pytest -q`, `python scripts/build_notebook.py`, `python scripts/validate_notebook.py`,
          `python scripts/package_submission.py`. Raport: `outputs/QA_REPORT.md`.
        """),
        code(r"""
        run_summary = {
            "data_sha256": file_sha256(DATA_ZIP), "split_random_state": split.random_state,
            "experiment_fingerprint": results_fingerprint(results), "selected_configuration": best_config.config_id,
            "final_tree_signature": final_model.structure_signature(),
            "test_metrics": {k: test_metrics[k] for k in ("accuracy", "macro_f1", "p_to_e", "e_to_p", "n")},
            "graphviz_backend": rendered["backend"], "environment": ENVIRONMENT,
        }
        (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        display(pd.Series({k: v for k, v in run_summary.items() if k != "environment"}, name="wartość").to_frame())
        """),
        md(r"""
        ### Bibliografia i źródła
        1. Schlimmer, J. (1987). *Mushroom* [Dataset]. UCI Machine Learning Repository.
           https://archive.ics.uci.edu/dataset/73/mushroom, DOI: https://doi.org/10.24432/C5959T
        2. Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*, 1, 81–106.
           https://doi.org/10.1007/BF00116251
        3. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
        4. Hunt, E. B., Marin, J., Stone, P. J. (1966). *Experiments in Induction*. Academic Press.
        5. Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
        6. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
        7. Hyafil, L., Rivest, R. L. (1976). Constructing optimal binary decision trees is NP-complete.
           *Information Processing Letters*, 5(1), 15–17.
        8. Fayyad, U., Irani, K. (1993). Multi-interval discretization of continuous-valued attributes for classification
           learning. *IJCAI-93*, 1022–1027.
        9. WEKA, `weka.classifiers.trees.Id3`:
           https://weka.sourceforge.io/doc.packages/simpleEducationalLearningSchemes/weka/classifiers/trees/Id3.html
        10. scikit-learn: Decision Trees (1.10.6 *Tree algorithms*): https://scikit-learn.org/stable/modules/tree.html
        11. JupyterLab: eksport notebooków: https://jupyterlab.readthedocs.io/en/stable/user/export.html
        12. Google Colab: praca z plikami zewnętrznymi: https://colab.research.google.com/notebooks/io.ipynb

        ### Licencja danych
        Zbiór *Mushroom* jest udostępniany przez UCI Machine Learning Repository na licencji
        **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Wykorzystanie wymaga podania źródła. Cytowanie:
        *Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T*. Archiwum
        `data/mushroom.zip` jest niezmienioną kopią pliku pobranego z UCI.

        ### Kontrola spójności narracji
        Ostatnia komórka zestawia każdą liczbę cytowaną w komentarzach tego notebooka (wstawioną przez generator
        z rzeczywistego uruchomienia) z wartością policzoną teraz. Jakakolwiek różnica przerywa wykonanie notebooka.
        """),
        code(r"""
        NARRATIVE_FACTS = None  # filled in by scripts/build_notebook.py
        (OUTPUT_DIR / "narrative_facts.json").write_text(json.dumps(FACTS, indent=2, ensure_ascii=False, default=str),
                                                         encoding="utf-8")
        if NARRATIVE_FACTS:
            rows = []
            for key, expected in NARRATIVE_FACTS.items():
                name, _, spec = key.partition("|")
                actual = format(FACTS[name], spec) if name in FACTS else "— (brak w tym uruchomieniu)"
                rows.append({"wartość": name, "format": spec or "—", "w tekście": expected, "obliczone teraz": actual,
                             "zgodne": actual == expected})
            consistency = pd.DataFrame(rows)
            print(f"Liczby w komentarzach zgodne z obliczeniami: {int(consistency['zgodne'].sum())} / {len(consistency)}")
            display(consistency[~consistency["zgodne"]] if not consistency["zgodne"].all() else consistency.sample(
                n=min(8, len(consistency)), random_state=0).sort_values("wartość"))
            assert consistency["zgodne"].all(), "Narracja niezgodna z wynikami — przebuduj notebook (build_notebook.py)."
        else:
            print(f"Przebieg roboczy generatora: zapisano {len(FACTS)} wartości.")
        """),
    ]
