"""Prosta aplikacja Streamlit prezentująca gotowe wyniki analizy."""

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
STATISTICS_DIR = PROJECT_ROOT / "outputs" / "statistics"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
ANALYSIS_COLOR = "#2F6B9A"
MODEL_COLORS = {
    "Logit": "#2F6B9A",
    "Random Forest": "#E58E26",
    "LightGBM": "#16324F",
}


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


st.set_page_config(
    page_title="Flight Delay Analytics",
    layout="wide",
)

overall_path = STATISTICS_DIR / "overall_summary.csv"
metrics_path = REPORTS_DIR / "logit_metrics.csv"
random_forest_metrics_path = REPORTS_DIR / "random_forest_metrics.csv"
lightgbm_metrics_path = REPORTS_DIR / "lightgbm_metrics.csv"

if not overall_path.exists():
    st.error("Brak wyników analizy. Najpierw uruchom: python analysis.py")
    st.stop()

st.title("Flight Delay Analytics")
st.caption("Analiza opóźnień lotów BTS — pełny rok 2025, próba 10%")
st.write(
    "Aplikacja przedstawia najważniejsze wyniki analizy opóźnień krajowych "
    "lotów pasażerskich w Stanach Zjednoczonych oraz porównuje trzy modele "
    "przewidujące opóźnienie odlotu."
)

overall = load_csv(overall_path).iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1.container(border=True):
    st.metric("Liczba lotów", f"{int(overall['flights']):,}".replace(",", " "))
with col2.container(border=True):
    st.metric("Opóźnione ≥ 15 min", f"{overall['delayed_15_rate']:.1%}")
with col3.container(border=True):
    st.metric("Średnie opóźnienie", f"{overall['mean_dep_delay']:.1f} min")
with col4.container(border=True):
    st.metric(
        "Odwołane loty",
        f"{int(overall['cancelled_flights']):,}".replace(",", " "),
    )

summary_tab, analysis_tab, models_tab, conclusions_tab = st.tabs(
    ["Podsumowanie", "Analiza opisowa", "Modele", "Wnioski i metodologia"]
)

with summary_tab:
    st.header("Podsumowanie projektu")
    st.write(
        "Celem projektu jest analiza opóźnień krajowych lotów pasażerskich "
        "w Stanach Zjednoczonych oraz porównanie modeli przewidujących, czy lot "
        "będzie opóźniony o co najmniej 15 minut."
    )
    st.write(
        "Analiza wykorzystuje dane Bureau of Transportation Statistics (BTS) "
        "za pełny rok 2025. Ze względu na wielkość zbioru zastosowano "
        "powtarzalną próbę 10%."
    )

    st.subheader("Zakres przewidywania")
    st.write(
        "Zmienna docelowa przyjmuje wartość 1, gdy opóźnienie odlotu "
        "wynosi co najmniej 15 minut, oraz 0 w pozostałych przypadkach. "
        "Modele korzystają wyłącznie z informacji dostępnych przed odlotem."
    )

with analysis_tab:
    st.header("Analiza opisowa")
    st.write("Wybierz jeden przekrój danych, aby ograniczyć długość strony.")
    tables = {
        "Miesiące": "by_month.csv",
        "Dni tygodnia": "by_day_of_week.csv",
        "Pory dnia": "by_time_of_day.csv",
        "Przewoźnicy": "by_carrier.csv",
        "Lotniska startowe": "by_origin.csv",
        "Lotniska docelowe": "by_destination.csv",
        "Trasy": "by_route.csv",
        "Przyczyny opóźnień": "delay_causes.csv",
    }
    view = st.selectbox("Wybierz przekrój:", list(tables))
    table = load_csv(STATISTICS_DIR / tables[view])

    if view == "Miesiące":
        table = table.sort_values("month")
        st.line_chart(
            table,
            x="month",
            y="delayed_15_rate",
            x_label="Miesiąc",
            y_label="Udział lotów opóźnionych",
            color=ANALYSIS_COLOR,
        )
        highest = table.loc[table["delayed_15_rate"].idxmax()]
        st.caption(
            f"Najwyższy udział opóźnień wystąpił w miesiącu "
            f"{int(highest['month'])}: {highest['delayed_15_rate']:.1%}."
        )
    elif view == "Dni tygodnia":
        day_order = [
            "Poniedzialek",
            "Wtorek",
            "Sroda",
            "Czwartek",
            "Piatek",
            "Sobota",
            "Niedziela",
        ]
        table["day_of_week"] = pd.Categorical(
            table["day_of_week"], categories=day_order, ordered=True
        )
        table = table.sort_values("day_of_week")
        table["day_of_week"] = table["day_of_week"].cat.rename_categories(
            {
                "Poniedzialek": "Poniedziałek",
                "Sroda": "Środa",
                "Piatek": "Piątek",
            }
        )
        st.line_chart(
            table,
            x="day_of_week",
            y="delayed_15_rate",
            x_label="Dzień tygodnia",
            y_label="Udział lotów opóźnionych",
            color=ANALYSIS_COLOR,
        )
        highest = table.loc[table["delayed_15_rate"].idxmax()]
        st.caption(
            f"Najwyższy udział opóźnień odnotowano w dniu "
            f"{highest['day_of_week']}: {highest['delayed_15_rate']:.1%}."
        )
    elif view == "Pory dnia":
        time_order = ["Noc", "Rano", "Poludnie", "Wieczor"]
        table["time_of_day"] = pd.Categorical(
            table["time_of_day"], categories=time_order, ordered=True
        )
        table = table.sort_values("time_of_day")
        table["time_of_day"] = table["time_of_day"].cat.rename_categories(
            {"Poludnie": "Południe", "Wieczor": "Wieczór"}
        )
        st.line_chart(
            table,
            x="time_of_day",
            y="delayed_15_rate",
            x_label="Pora dnia",
            y_label="Udział lotów opóźnionych",
            color=ANALYSIS_COLOR,
        )
        highest = table.loc[table["delayed_15_rate"].idxmax()]
        st.caption(
            f"Najwyższy udział opóźnień wystąpił w kategorii "
            f"{highest['time_of_day']}: {highest['delayed_15_rate']:.1%}."
        )
    elif view == "Przyczyny opóźnień":
        table = table.sort_values("total_minutes", ascending=False)
        table["cause"] = table["cause"].replace(
            {
                "CARRIER_DELAY": "Przewoźnik",
                "WEATHER_DELAY": "Pogoda",
                "NAS_DELAY": "System lotniczy",
                "SECURITY_DELAY": "Bezpieczeństwo",
                "LATE_AIRCRAFT_DELAY": "Spóźniony samolot",
            }
        )
        st.line_chart(
            table,
            x="cause",
            y="total_minutes",
            x_label="Przyczyna",
            y_label="Łączna liczba minut opóźnienia",
            color=ANALYSIS_COLOR,
        )
        highest = table.iloc[0]
        st.caption(
            f"Najwięcej minut opóźnienia przypisano kategorii "
            f"{highest['cause']}: {int(highest['total_minutes']):,} min."
            .replace(",", " ")
        )
    elif view == "Przewoźnicy":
        table = table.head(12)
        st.line_chart(
            table,
            x="OP_UNIQUE_CARRIER",
            y="delayed_15_rate",
            x_label="Przewoźnik",
            y_label="Udział lotów opóźnionych",
            color=ANALYSIS_COLOR,
        )
        highest = table.loc[table["delayed_15_rate"].idxmax()]
        st.caption(
            f"Wśród 12 największych przewoźników najwyższy udział "
            f"opóźnień miał {highest['OP_UNIQUE_CARRIER']}: "
            f"{highest['delayed_15_rate']:.1%}."
        )
    else:
        table = table.head(15)
        st.caption(
            "Tabela pokazuje 15 pozycji o największej liczbie lotów, "
            "aby zachować czytelność widoku."
        )

    column_names = {
        "month": "Miesiąc",
        "day_of_week": "Dzień tygodnia",
        "time_of_day": "Pora dnia",
        "OP_UNIQUE_CARRIER": "Przewoźnik",
        "ORIGIN": "Lotnisko startowe",
        "DEST": "Lotnisko docelowe",
        "route": "Trasa",
        "cause": "Przyczyna",
        "flights": "Liczba lotów",
        "known_departure_delay": "Znane opóźnienie odlotu",
        "mean_dep_delay": "Średnie opóźnienie (min)",
        "median_dep_delay": "Mediana opóźnienia (min)",
        "delayed_15_rate": "Opóźnione ≥ 15 min",
        "delayed_30_rate": "Opóźnione ≥ 30 min",
        "cancelled_rate": "Odwołane",
        "diverted_rate": "Przekierowane",
        "reported_flights": "Raportowane loty",
        "flights_with_positive_minutes": "Loty z opóźnieniem",
        "total_minutes": "Łączne opóźnienie (min)",
        "mean_minutes_when_positive": "Średnie opóźnienie dodatnie (min)",
    }
    display_table = table.rename(columns=column_names)
    percent_columns = [
        name
        for name in [
            "Opóźnione ≥ 15 min",
            "Opóźnione ≥ 30 min",
            "Odwołane",
            "Przekierowane",
        ]
        if name in display_table.columns
    ]
    table_format = {name: "{:.1%}" for name in percent_columns}
    st.dataframe(
        display_table.style.format(table_format),
        width="stretch",
        hide_index=True,
    )

with models_tab:
    st.header("Porównanie modeli")
    if not (
        metrics_path.exists()
        and random_forest_metrics_path.exists()
        and lightgbm_metrics_path.exists()
    ):
        st.warning("Brak kompletu wyników modeli w katalogu outputs/reports.")
    else:
        logit_metrics = load_csv(metrics_path).iloc[0]
        random_forest_metrics = load_csv(random_forest_metrics_path).iloc[0]
        lightgbm_metrics = load_csv(lightgbm_metrics_path).iloc[0]
        comparison = pd.DataFrame(
            {
                "Model": ["Logit", "Random Forest", "LightGBM"],
                "ROC-AUC": [
                    logit_metrics["roc_auc"],
                    random_forest_metrics["roc_auc"],
                    lightgbm_metrics["roc_auc"],
                ],
                "Accuracy": [
                    logit_metrics["accuracy"],
                    random_forest_metrics["accuracy"],
                    lightgbm_metrics["accuracy"],
                ],
                "Balanced accuracy": [
                    logit_metrics["balanced_accuracy"],
                    random_forest_metrics["balanced_accuracy"],
                    lightgbm_metrics["balanced_accuracy"],
                ],
                "Precision": [
                    logit_metrics["precision"],
                    random_forest_metrics["precision"],
                    lightgbm_metrics["precision"],
                ],
                "Recall": [
                    logit_metrics["recall"],
                    random_forest_metrics["recall"],
                    lightgbm_metrics["recall"],
                ],
            }
        )
        st.dataframe(
            comparison.style.format(
                {
                    "ROC-AUC": "{:.3f}",
                    "Accuracy": "{:.1%}",
                    "Balanced accuracy": "{:.1%}",
                    "Precision": "{:.1%}",
                    "Recall": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Metryki modeli")
        comparison_chart = comparison.set_index("Model")[
            ["ROC-AUC", "Accuracy", "Balanced accuracy", "Precision", "Recall"]
        ].transpose()
        st.line_chart(
            comparison_chart,
            x_label="Metryka",
            y_label="Wartość metryki",
            color=[
                MODEL_COLORS["Logit"],
                MODEL_COLORS["Random Forest"],
                MODEL_COLORS["LightGBM"],
            ],
        )
        st.caption(
            f"LightGBM uzyskał najwyższe ROC-AUC "
            f"({lightgbm_metrics['roc_auc']:.3f}) i balanced accuracy "
            f"({lightgbm_metrics['balanced_accuracy']:.1%}), natomiast Logit "
            f"najwyższy recall ({logit_metrics['recall']:.1%})."
        )

        st.subheader("Progi Youdena")
        thresholds = pd.DataFrame(
            {
                "Model": ["Logit", "Random Forest", "LightGBM"],
                "Próg Youdena": [
                    logit_metrics["selected_threshold_youden"],
                    random_forest_metrics["selected_threshold_youden"],
                    lightgbm_metrics["selected_threshold_youden"],
                ],
            }
        )
        st.dataframe(
            thresholds.style.format({"Próg Youdena": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )

        importance_model = st.selectbox(
            "Najważniejsze zmienne:", ["Random Forest", "LightGBM"]
        )
        importance_files = {
            "Random Forest": "random_forest_feature_importance.csv",
            "LightGBM": "lightgbm_feature_importance.csv",
        }
        importance = load_csv(
            REPORTS_DIR / importance_files[importance_model]
        ).head(15)
        st.line_chart(
            importance,
            x="variable",
            y="importance",
            x_label="Zmienna",
            y_label="Ważność",
            color=MODEL_COLORS[importance_model],
        )
        most_important = importance.iloc[0]
        st.caption(
            f"Najważniejsza zmienna modelu {importance_model} to "
            f"{most_important['variable']} (ważność: "
            f"{most_important['importance']:.3f})."
        )
        st.dataframe(
            importance.rename(
                columns={"variable": "Zmienna", "importance": "Ważność"}
            ).style.format({"Ważność": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )

        confusion_model = st.selectbox(
            "Macierz pomyłek:", ["Logit", "Random Forest", "LightGBM"]
        )
        logit_confusion = pd.DataFrame(
            {
                "actual_class": ["actual_0", "actual_1"],
                "predicted_0": [
                    int(logit_metrics["true_negative"]),
                    int(logit_metrics["false_negative"]),
                ],
                "predicted_1": [
                    int(logit_metrics["false_positive"]),
                    int(logit_metrics["true_positive"]),
                ],
            }
        )
        confusion_tables = {
            "Logit": logit_confusion,
            "Random Forest": load_csv(
                REPORTS_DIR / "random_forest_confusion_matrix.csv"
            ),
            "LightGBM": load_csv(REPORTS_DIR / "lightgbm_confusion_matrix.csv"),
        }
        confusion = confusion_tables[confusion_model]
        confusion_max = max(
            frame[["predicted_0", "predicted_1"]].to_numpy().max()
            for frame in confusion_tables.values()
        )
        confusion = confusion.set_index("actual_class").rename(
            index={"actual_0": "Rzeczywista: 0", "actual_1": "Rzeczywista: 1"},
            columns={
                "predicted_0": "Przewidywana: 0",
                "predicted_1": "Przewidywana: 1",
            },
        )
        confusion.index.name = "Klasa rzeczywista"
        st.dataframe(
            confusion.style.background_gradient(
                axis=None, cmap="Blues", vmin=0, vmax=confusion_max
            ).format("{:.0f}"),
            width="stretch",
        )
        st.caption(
            "TN — prawidłowo rozpoznany lot nieopóźniony; FP — lot błędnie "
            "uznany za opóźniony; FN — nierozpoznany lot opóźniony; "
            "TP — prawidłowo rozpoznany lot opóźniony. Wszystkie macierze "
            "korzystają z tej samej skali kolorów."
        )

        st.info(
            "Wszystkie modele wykorzystują te same zmienne i ten sam "
            "chronologiczny zbiór testowy. Próg klasyfikacji wyznaczono "
            "metodą Youdena wyłącznie na zbiorze treningowym."
        )

with conclusions_tab:
    st.header("Wnioski")
    if (
        metrics_path.exists()
        and random_forest_metrics_path.exists()
        and lightgbm_metrics_path.exists()
    ):
        logit_metrics = load_csv(metrics_path).iloc[0]
        lightgbm_metrics = load_csv(lightgbm_metrics_path).iloc[0]
        st.write(
            f"LightGBM osiągnął najwyższe ROC-AUC "
            f"({lightgbm_metrics['roc_auc']:.3f}) i balanced accuracy "
            f"({lightgbm_metrics['balanced_accuracy']:.1%})."
        )
        st.write(
            f"Logit osiągnął najwyższy recall "
            f"({logit_metrics['recall']:.1%}), czyli wykrył największą część "
            "lotów rzeczywiście opóźnionych."
        )
    else:
        st.warning("Brak kompletu wyników potrzebnych do przedstawienia wniosków.")

    st.subheader("Jak interpretować wyniki")
    st.write(
        "Sama accuracy może zawyżać ocenę modelu, gdy klasy nie są "
        "równoliczne. Balanced accuracy nadaje taką samą wagę poprawnemu "
        "rozpoznawaniu lotów opóźnionych i nieopóźnionych."
    )

    st.subheader("Metodologia i ograniczenia")
    st.write(
        "Dane podzielono chronologicznie: wcześniejsze obserwacje wykorzystano "
        "do trenowania, a późniejsze do testowania. Ten sam zestaw zmiennych "
        "i wspólny zbiór testowy umożliwiają porównanie modeli bez wycieku danych."
    )
    st.write(
        "Wyniki dotyczą 10-procentowej próby lotów BTS z 2025 roku. "
        "Predykcja wykorzystuje wyłącznie informacje znane przed odlotem, "
        "dlatego nie obejmuje zdarzeń pojawiających się dopiero w trakcie lotu."
    )
