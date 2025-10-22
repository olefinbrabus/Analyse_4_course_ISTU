import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, fft

np.random.seed(42)


def main():
    dates = pd.date_range("2016-01-01", periods=120, freq="MS")
    t = np.arange(len(dates))

    marketing = (
        35
        + 6 * np.sin(2 * np.pi * (t - 1) / 12)
        + 2 * np.sin(2 * np.pi * (t - 1) / 6)
        + np.random.normal(0, 1.2, size=len(t))
    )
    baseline = (
        120 + 0.9 * t + 18 * np.sin(2 * np.pi * t / 12) + 7 * np.sin(2 * np.pi * t / 6)
    )

    lag = 2
    marketing_influence = 0.35 * np.roll(marketing, lag)
    marketing_influence[:lag] = marketing_influence[lag]

    sales = baseline + marketing_influence + np.random.normal(0, 3.0, size=len(t))

    df = pd.DataFrame(
        {"date": dates, "t": t, "marketing": marketing, "sales": sales}
    ).set_index("date")

    # ---- Structure diagnostics ----
    trend_ma12 = df["sales"].rolling(window=12, center=True).mean()
    sales_detrended = signal.detrend(df["sales"].values)

    freqs = fft.fftfreq(len(sales_detrended), d=1)
    power = np.abs(fft.fft(sales_detrended)) ** 2
    pos = freqs > 0
    freqs_pos, power_pos = freqs[pos], power[pos]

    def acf(x, max_lag):
        x = np.asarray(x)
        x = x - x.mean()
        n = len(x)
        corr = np.correlate(x, x, mode="full")[n - 1 :]
        corr = corr / corr[0]
        return corr[: max_lag + 1]

    lags = 24
    acf_vals = acf(df["sales"].values, lags)

    # ---- Cross-correlation ----
    def xcorr(x, y, max_lag):
        x = np.asarray(x) - np.mean(x)
        y = np.asarray(y) - np.mean(y)
        n = len(x)
        corr = np.correlate(y, x, mode="full")
        mid = len(corr) // 2
        lags = np.arange(-max_lag, max_lag + 1)
        vals = []
        for k in lags:
            vals.append(corr[mid + k])
        vals = np.array(vals)
        vals = vals / (np.std(x) * np.std(y) * n)
        return lags, vals

    maxlag = 12
    cc_lags, cc_vals = xcorr(df["marketing"].values, df["sales"].values, maxlag)
    best_lag = cc_lags[np.argmax(cc_vals)]

    # ---- Seasonal regression model ----
    def seasonal_design(n, period):
        idx = np.arange(n)
        s = np.sin(2 * np.pi * idx / period)
        c = np.cos(2 * np.pi * idx / period)
        return s, c

    n = len(df)
    s12, c12 = seasonal_design(n, 12)
    s6, c6 = seasonal_design(n, 6)

    m_lag2 = np.roll(df["marketing"].values, 2)
    m_lag2[:2] = m_lag2[2]

    X = np.column_stack([np.ones(n), df["t"].values, s12, c12, s6, c6, m_lag2])
    y = df["sales"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    sigma = resid.std(ddof=X.shape[1])

    # ---- Forecast 24 months ----
    h = 24
    t_f = np.arange(n, n + h)
    s12_f, c12_f = seasonal_design(h, 12)
    s6_f, c6_f = seasonal_design(h, 6)
    marketing_future = np.tile(df["marketing"].values[-12:], 2)[:h]
    m_lag2_f = np.roll(np.concatenate([df["marketing"].values, marketing_future]), 2)[
        n : n + h
    ]

    X_f = np.column_stack([np.ones(h), t_f, s12_f, c12_f, s6_f, c6_f, m_lag2_f])
    y_f = X_f @ beta

    z = 1.28
    pi_low, pi_high = y_f - z * sigma, y_f + z * sigma

    # ---- Smoothing & filtering ----
    ma3 = df["sales"].rolling(window=3, center=True).mean()
    ma12 = df["sales"].rolling(window=12, center=True).mean()
    savgol = signal.savgol_filter(
        df["sales"].values, window_length=11, polyorder=3, mode="interp"
    )
    b, a = signal.butter(4, 0.15)
    sales_lowpass = signal.filtfilt(b, a, df["sales"].values)

    # ---- Plots ----
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["sales"], label="Sales")
    plt.plot(df.index, trend_ma12, label="Trend (MA12)")
    plt.title("Sales with 12-month Moving Average Trend")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.stem(np.arange(len(acf_vals)), acf_vals)
    plt.title("Autocorrelation (ACF) of Sales (up to 24 lags)")
    plt.xlabel("Lag (months)")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_pos, power_pos)
    plt.title("Periodogram of Detrended Sales (FFT Power)")
    plt.xlabel("Frequency (cycles/month)")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["marketing"], label="Marketing spend")
    plt.plot(df.index, df["sales"], label="Sales")
    plt.title(f"Marketing vs Sales (best cross-corr lag ≈ {best_lag} months)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["sales"], label="Actual")
    plt.plot(df.index, y_hat, label="Model fit")
    plt.title("Deterministic Seasonal Regression with Marketing Lag")
    plt.legend()
    plt.tight_layout()
    plt.show()

    future_idx = pd.date_range(
        df.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["sales"], label="History")
    plt.plot(future_idx, y_f, label="Forecast")
    plt.fill_between(future_idx, pi_low, pi_high, alpha=0.2, label="80% PI")
    plt.title("24-month Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["sales"], label="Original")
    plt.plot(df.index, ma3, label="MA(3)")
    plt.plot(df.index, ma12, label="MA(12)")
    plt.plot(df.index, savgol, label="Savitzky-Golay")
    plt.plot(df.index, sales_lowpass, label="Butterworth low-pass")
    plt.title("Smoothing & Filtering")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
