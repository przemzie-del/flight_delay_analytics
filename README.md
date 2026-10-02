# Flight Delay Analytics

Lekka aplikacja Streamlit prezentująca wyniki analizy opóźnień krajowych lotów pasażerskich w Stanach Zjednoczonych. Analiza obejmuje losową próbę o wielkości 10% oryginalnego zbioru danych BTS z pełnego roku 2025 i porównuje modele Logit, Random Forest oraz LightGBM.

Aplikacja korzysta wyłącznie z gotowych plików CSV znajdujących się w katalogach `outputs/statistics` i `outputs/reports`. Nie wczytuje pełnej bazy BTS i nie trenuje modeli podczas uruchomienia.

Zakładka Opis projektu przedstawia cel, dane, metodologię, chronologiczny podział na zbiór uczący i testowy oraz ograniczenia. Zakładka Modele objaśnia predyktory i progi Youdena oraz prezentuje poziome wykresy 15 najważniejszych zmiennych Random Forest i LightGBM.

## Uruchomienie lokalne

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Docelowym środowiskiem jest Python 3.12. Pakiet korzysta wyłącznie z dokładnie przypiętych wersji pandas, Matplotlib i Streamlit. Matplotlib służy do poziomych wykresów ważności zmiennych oraz map cieplnych macierzy pomyłek.
