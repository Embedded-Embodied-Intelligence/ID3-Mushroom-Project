# Warsztat ID3: „Delivery Decision — czy przesyłka dotrze na czas?”

Krótkie zadanie z drzew decyzyjnych ID3: na 28 syntetycznych przesyłkach wykrywasz i uzupełniasz braki danych,
liczysz entropię i przyrost informacji, wybierasz korzeń drzewa, a gotowym mini-ID3 przewidujesz los nowej przesyłki.

## Zawartość paczki

| Plik | Opis |
|---|---|
| `ID3_Delivery_Workshop.ipynb` | notebook z zadaniem (4 komórki TODO z automatycznymi kontrolkami) |
| `delivery_data.csv` | dane: 28 wierszy, cechy `distance`, `weather`, `workload`, `shipping`, etykieta `on_time` |
| `README.md` | ten plik |

Notebook jest **samodzielny**: dane są wbudowane w komórkę, a mini-ID3 w komórkę pomocniczą. Nie trzeba niczego
instalować, montować cudzego Dysku Google ani pobierać repozytorium. Plik CSV dołączamy dla wygody, np. do
własnych eksperymentów.

## Jak pracować

**Google Colab (zalecane):** otwórz notebook z udostępnionego folderu (*Otwórz za pomocą → Google Colaboratory*),
wybierz *Plik → Zapisz kopię na Dysku*, a potem *Środowisko wykonawcze → Uruchom wszystko*. Pracujesz na własnej
kopii.

**Lokalnie:** `jupyter lab ID3_Delivery_Workshop.ipynb` (wystarczą `pandas`, `numpy`, `matplotlib`).

| Minuty | Zadanie | Wynik |
|---|---|---|
| 0–4 | uruchom kod pomocniczy, obejrzyj dane | tabela danych |
| 4–9 | TODO 1: braki danych → dominanta | tabela „braki przed / po” |
| 9–16 | TODO 2–3: entropia, IG, korzeń | ranking IG + mini-drzewo |
| 16–20 | TODO 4: predykcja nowej przesyłki + uzasadnienie | predykcja i 1–2 zdania |

Po każdym TODO kontrolka wypisze **✅ zaliczone** albo **⏳ wskazówkę**. Niedokończone zadanie nie blokuje dalszych
kroków.

**W domu:** te same materiały, bez limitu czasu. Opcjonalny **bonus** na końcu notebooka porównuje uzupełnianie
dominantą z osobną kategorią braków i ich wpływ na wybór korzenia.

---

## Dla prowadzącego warsztat: udostępnienie tylko osobom z grupy

Materiały **nie są publikowane automatycznie**. Wybierz jeden z wariantów:

**A. Google Drive / Colab (zalecane)**
1. Utwórz na swoim Dysku folder, np. `ID3 – warsztat` (nie w folderze publicznym).
2. Wgraj trzy pliki z `submission/ID3_Group_Workshop_Students.zip` (rozpakowane).
3. *Udostępnij* folder → w polu „Dostęp ogólny” pozostaw **„Ograniczony”** (nie „Każda osoba mająca link”).
4. Dodaj adresy e-mail osób z grupy z rolą **Przeglądający**. Uczestnicy robią *Zapisz kopię na Dysku*, więc
   oryginał pozostaje nienaruszony.
5. Wklej link do folderu w notebooku głównym (`WORKSHOP_LINK`, rozdział 10) i wyślij go grupie.

**B. Prywatne repozytorium GitHub**
1. Utwórz repozytorium **Private**, dodaj trzy pliki paczki (repozytorium *nie może* zawierać `instructor_private/`;
   `.gitignore` projektu już to wyklucza).
2. *Settings → Collaborators*: zaproś konta osób z grupy.

**C. Platforma uczelni** (Moodle, Teams): wgraj paczkę do kursu widocznego tylko dla grupy.

Rozwiązanie i klucz odpowiedzi (`instructor_private/`, paczka `ID3_Workshop_Instructor_PRIVATE.zip`) zostają
u prowadzącego warsztat. Nie umieszczaj ich w udostępnionym folderze.
