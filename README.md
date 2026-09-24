# Flight Delay Analytics

Lekka aplikacja Streamlit prezentująca wyniki analizy opóźnień krajowych lotów pasażerskich w Stanach Zjednoczonych. Analiza obejmuje 10-procentową, powtarzalną próbę danych BTS z pełnego roku 2025 i porównuje modele Logit, Random Forest oraz LightGBM.

Aplikacja korzysta wyłącznie z gotowych plików CSV znajdujących się w katalogach `outputs/statistics` i `outputs/reports`. Nie wczytuje pełnej bazy BTS i nie trenuje modeli podczas uruchomienia.

## Uruchomienie lokalne

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Docelowym środowiskiem jest Python 3.12. Pakiet korzysta wyłącznie z dokładnie przypiętych wersji pandas, Matplotlib i Streamlit. Pełna instalacja oraz kontrola w świeżym środowisku zostaną wykonane przed publikacją.
