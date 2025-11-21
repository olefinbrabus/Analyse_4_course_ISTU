import random
from datetime import datetime, date, time, timedelta

import numpy as np
import pandas as pd


def build_drivers():
    # 20 водіїв з реальними на вигляд прізвищами та ставками відрахувань
    surnames = [
        "Іваненко", "Петренко", "Сидоренко", "Коваль", "Шевченко",
        "Бондар", "Мельник", "Ткаченко", "Кравченко", "Лисенко",
        "Гриценко", "Романюк", "Поліщук", "Демченко", "Онищенко",
        "Кирилюк", "Тарасенко", "Савченко", "Гнатюк", "Клименко",
    ]
    drivers = []
    base_id = 1001
    for i, surname in enumerate(surnames):
        driver_id = base_id + i
        # 15–30% відрахувань
        deduction_rate = random.choice([0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30])
        drivers.append(
            {
                "driver_id": driver_id,
                "driver_name": surname,
                "deduction_rate": deduction_rate,
            }
        )
    return drivers


def build_cars():
    # парк машин з реалістичними брендами і номерами
    brands = ["Toyota Corolla", "Skoda Octavia", "Hyundai Elantra",
              "Renault Logan", "Volkswagen Passat", "Kia Rio"]
    # 10 машин із номерними знаками UA-формату на вигляд
    cars = []
    letters = ["A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X"]
    for i in range(10):
        prefix = random.choice(letters) + random.choice(letters)
        number = random.randint(1000, 9999)
        suffix = random.choice(letters) + random.choice(letters)
        car_number = f"{prefix}{number}{suffix}"
        car_brand = random.choice(brands)
        cars.append({"car_number": car_number, "car_brand": car_brand})
    return cars


def get_shift_by_hour(hour: int) -> int:
    if 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    elif 18 <= hour < 24:
        return 3
    else:
        return 0  # технічне значення, у реальних змін не буде


def build_shifts(drivers, cars,
                 start_date=date(2025, 11, 1),
                 end_date=date(2025, 11, 30)):
    """
    Генерує зміни для 20 водіїв на місяць.
    Кожен водій: 0–2 зміни на день, випадкові машини, реалістичний виторг.
    """
    all_rows = []

    # доступні початки змін
    shift_starts = {
        1: time(6, 0),
        2: time(12, 0),
        3: time(18, 0),
    }

    current = start_date
    while current <= end_date:
        for d in drivers:
            # випадково: скільки змін у цього водія у цей день
            num_shifts = np.random.choice([0, 1, 1, 1, 2], p=[0.2, 0.4, 0.2, 0.1, 0.1])
            if num_shifts == 0:
                continue

            # обираємо, які зміни (1,2,3) він відпрацює
            possible_shifts = [1, 2, 3]
            day_shifts = random.sample(possible_shifts, num_shifts)

            for shift in day_shifts:
                start_t = shift_starts[shift]
                # тривалість: 4–8 годин, але не виходимо за межі доби
                max_duration = 8
                duration_hours = random.randint(4, max_duration)

                # дата+час початку зміни
                start_dt = datetime.combine(current, start_t)

                # машина
                car = random.choice(cars)

                # реалістичний виторг:
                # денні зміни трохи вигідніші, вечірні – ще більше
                base_rate = {
                    1: (350, 500),   # 1 зміна: грн/год
                    2: (300, 450),   # 2 зміна
                    3: (250, 550),   # 3 зміна, більше розкид через нічні
                }[shift]
                hourly_income = random.uniform(*base_rate)
                revenue = round(hourly_income * duration_hours, 2)

                avg_hour_revenue = round(revenue / duration_hours, 2)
                deduction = round(revenue * d["deduction_rate"], 2)
                net_income = round(revenue - deduction, 2)

                all_rows.append(
                    {
                        "driver_id": d["driver_id"],
                        "driver_name": d["driver_name"],
                        "deduction_rate": d["deduction_rate"],
                        "car_number": car["car_number"],
                        "car_brand": car["car_brand"],
                        "shift_date": current,
                        "start_time": start_dt,
                        "duration_hours": duration_hours,
                        "shift": shift,
                        "revenue": revenue,
                        "avg_hour_revenue": avg_hour_revenue,
                        "deduction": deduction,
                        "net_income": net_income,
                    }
                )

        current += timedelta(days=1)

    df = pd.DataFrame(all_rows)
    return df


def main():
    drivers = build_drivers()
    cars = build_cars()
    df = build_shifts(drivers, cars)

    # базова перевірка якості
    assert (df["shift"].isin([1, 2, 3])).all()
    assert (df["duration_hours"] >= 4).all()
    assert (df["duration_hours"] <= 8).all()
    assert (df["revenue"] > 0).all()
    assert (df["net_income"] > 0).all()
    assert (df["net_income"] < df["revenue"]).all()

    # зберігаємо як CSV для завантаження
    df.to_csv("taxi_shifts.csv", index=False)
    print(f"Згенеровано {len(df)} записів і збережено в data/taxi_shifts.csv")


if __name__ == "__main__":
    main()