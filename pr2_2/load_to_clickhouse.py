import pandas as pd
import clickhouse_connect


CSV_PATH = "taxi_shifts.csv"


def main():
    # читаємо підготовлений датасет
    df = pd.read_csv(CSV_PATH, parse_dates=["start_time"])

    # ClickHouse client (хост = localhost, бо Python з хоста; порт проброшений)
    client = clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="taxi_user",
        password="test_pass",
        database="taxi",
        interface="http",
    )

    # ClickHouse очікує список списків / кортежів
    columns = [
        "driver_id",
        "driver_name",
        "deduction_rate",
        "car_number",
        "car_brand",
        "shift_date",
        "start_time",
        "duration_hours",
        "shift",
        "revenue",
        "avg_hour_revenue",
        "deduction",
        "net_income",
    ]

    # переконаємось, що є стовпець shift_date правильного типу
    if "shift_date" not in df.columns:
        # якщо раптом немає – беремо тільки дату зі start_time
        df["shift_date"] = df["start_time"].dt.date

    # приведення типів під ClickHouse
    df["shift_date"] = pd.to_datetime(df["shift_date"]).dt.date
    df["start_time"] = pd.to_datetime(df["start_time"])

    # підготовка рядків
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                int(row["driver_id"]),
                str(row["driver_name"]),
                float(row["deduction_rate"]),
                str(row["car_number"]),
                str(row["car_brand"]),
                row["shift_date"],
                row["start_time"],
                int(row["duration_hours"]),
                int(row["shift"]),
                float(row["revenue"]),
                float(row["avg_hour_revenue"]),
                float(row["deduction"]),
                float(row["net_income"]),
            ]
        )

    client.insert("taxi.shifts", rows, column_names=columns)
    print(f"Завантажено {len(rows)} рядків у taxi.shifts")


if __name__ == "__main__":
    main()