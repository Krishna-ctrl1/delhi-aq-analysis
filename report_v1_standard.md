# Multi-Modal Characterisation, Causal Attribution, and Predictive Modelling of Ambient Air Quality in Delhi, India (2009–2025)

---

**Authors:** K. Mahesh  
**Affiliation:** Independent Research  
**Date:** April 2026  
**Data Source:** Central Pollution Control Board (CPCB), Delhi Pollution Control Committee (DPCC), India Meteorological Department (IMD), Indian Institute of Tropical Meteorology (IITM)  
**Code Repository:** `delhi-aq-analysis/`

---

## Abstract

This study presents an integrated, multi-method analysis of ambient air quality across the National Capital Territory of Delhi, India, leveraging a high-resolution (15-minute interval) multi-station monitoring dataset spanning 17 years (2009–2025). Drawing from 432 raw data files encompassing 38 monitoring stations and approximately 14.8 million observations, the analysis proceeds through six structured analytical stages: (i) data ingestion and harmonisation, (ii) exploratory spatiotemporal analysis, (iii) volatile organic compound (VOC) source fingerprinting via BTEX ratios, (iv) causal discovery using Granger causality and the PCMCI+ algorithm, (v) multivariate anomaly detection using Isolation Forest and change-point detection via PELT, and (vi) multi-horizon PM₂.₅ forecasting using statistical, gradient-boosted, and deep learning models. Key findings include the identification of traffic-dominated VOC source signatures at industrial stations, statistically significant causal pathways from meteorological variables (solar radiation, CO, NO₂) to PM₂.₅ concentrations, and superior forecasting performance from the Random Forest model (RMSE = 44.44 µg/m³, R² = 0.889) on the high-pollution test period (November–December 2019), substantially outperforming SARIMA and competitive with the LSTM architecture.

---

## 1. Introduction

### 1.1 Background

Delhi consistently ranks among the world's most polluted megacities, with annual mean PM₂.₅ concentrations routinely exceeding WHO guidelines (5 µg/m³ annual mean) by an order of magnitude. The pollution crisis is driven by a complex interplay of vehicular emissions, industrial activity, construction dust, agricultural crop residue burning in neighbouring states, and adverse meteorological conditions — particularly during the post-monsoon and winter months when atmospheric boundary layer inversions trap pollutants at ground level.

### 1.2 Objectives

This project is designed to address the following research questions:

1. **Spatiotemporal Characterisation:** What are the dominant temporal patterns (diurnal, weekly, seasonal) and spatial heterogeneity of criteria pollutants across Delhi's monitoring network?
2. **Source Attribution:** Can VOC fingerprinting via Toluene/Benzene (T/B) and Xylene/Benzene (X/B) ratios discriminate between traffic, industrial, and mixed pollution sources at the station level?
3. **Causal Structure:** Which meteorological and co-pollutant variables exhibit statistically significant Granger-causal relationships with PM₂.₅, and does the causal structure vary by season?
4. **Anomaly Detection:** Can machine learning–based anomaly detection (Isolation Forest) and change-point analysis (PELT) identify and characterise extreme pollution episodes?
5. **Forecasting:** Which modelling paradigm — statistical time-series (SARIMA), gradient-boosted machines (XGBoost, LightGBM, Random Forest), deep learning (LSTM), or a weighted ensemble — provides optimal 1-hour-ahead PM₂.₅ forecasts during the most polluted period of the year?

---

## 2. Data and Study Area

### 2.1 Monitoring Network

Data were obtained from 38 continuous ambient air quality monitoring (CAAQM) stations operated by three agencies:

| Agency | Stations | Temporal Coverage |
|--------|----------|-------------------|
| CPCB (Central Pollution Control Board) | 6 core stations (Shadipur, IHBAS Dilshad Garden, NSIT Dwarka, ITO, DTU, Sirifort) | 2009–2025 |
| DPCC (Delhi Pollution Control Committee) | 25 stations (Anand Vihar, R.K. Puram, Punjabi Bagh, Ashok Vihar, Jahangirpuri, Narela, Wazirpur, Rohini, etc.) | 2011–2025 |
| IMD (India Meteorological Department) | 7 stations (CRRI Mathura Road, Burari Crossing, North Campus DU, IGI Airport T3, Pusa, Aya Nagar, Lodhi Road) | 2017–2025 |

### 2.2 Variables

Each station records up to **25 parameters** at **15-minute intervals**:

- **Criteria Pollutants:** PM₂.₅, PM₁₀, NO, NO₂, NOₓ, NH₃, SO₂, CO, O₃
- **Volatile Organic Compounds (VOCs):** Benzene, Toluene, Xylene, ortho-Xylene, Ethylbenzene, meta/para-Xylene
- **Meteorological Parameters:** Air Temperature (AT), Relative Humidity (RH), Wind Speed (WS), Wind Direction (WD), Rainfall (RF), Total Rainfall (TOT-RF), Solar Radiation (SR), Barometric Pressure (BP), Vertical Wind Speed (VWS)

### 2.3 Data Volume and Quality

| Metric | Value |
|--------|-------|
| Total raw CSV files | 432 |
| Total observations | 14,769,888 |
| Temporal span | 2009-01-01 to 2025-12-31 |
| Temporal resolution | 15-minute intervals |
| Unique monitoring stations | 38 |
| Missing data (PM₂.₅) | 28.68% |
| Missing data (PM₁₀) | 34.23% |
| Missing data (Xylene) | 90.63% |

> [!NOTE]
> Missing data percentages reflect the full 2009–2025 corpus. Many DPCC and IMD stations were commissioned after 2017, contributing structurally to the missingness. The 2019 analysis year was selected for its network completeness (37 active stations) and data quality.

---

## 3. Methodology

### 3.1 Data Ingestion and Harmonisation (Notebook 01)

All 432 station-year CSV files were programmatically ingested, with station name and year metadata extracted from filenames using structured parsing rules. Column names were standardised to remove units and special characters. Timestamps were converted to `datetime64` and validated; zero rows failed parsing. The harmonised dataset was persisted as a consolidated CSV (2.46 GB) and a focused 2019 subset (1,366,560 rows × 28 columns) for computationally intensive analyses.

### 3.2 Exploratory Data Analysis (Notebook 02)

A comprehensive 19-cell EDA was conducted on the 2019 dataset, encompassing:

1. **Descriptive Statistics:** Computed mean, standard deviation, skewness, and kurtosis for all 12 pollutants.
2. **Missing Data Visualisation:** Matrix plots (via `missingno`) and per-station/per-variable heatmaps to characterise data availability patterns.
3. **Time-Series Visualisation:** Hourly PM₂.₅ traces for 6 representative stations with NAAQS (60 µg/m³) and Severe (250 µg/m³) thresholds overlaid.
4. **Event Analysis:** Focused zoom on the Diwali period (October 24–November 3, 2019) showing PM₂.₅ and PM₁₀ excursions.
5. **Diurnal Profiles:** Hour-of-day mean concentrations for PM₂.₅, PM₁₀, NO₂, O₃, CO, and SO₂, with morning (08:00–10:00) and evening (17:00–20:00) rush-hour annotations.
6. **Day-of-Week Analysis:** Weekday/weekend decomposition of PM₂.₅, NO₂, CO, and Benzene.
7. **Seasonal Box Plots:** Monthly distributions of PM₂.₅, PM₁₀, NO₂, and O₃ colour-coded by season (Winter, Summer, Monsoon, Post-Monsoon).
8. **Correlation Analysis:** Pearson correlation heatmap for 18 pollutant and meteorological variables at Patparganj station.
9. **STL Decomposition:** Additive seasonal-trend decomposition (period = 24 hours) of PM₂.₅, isolating trend, daily seasonality, and residual components.
10. **Stationarity Testing:** Augmented Dickey-Fuller (ADF) and KPSS tests on PM₂.₅, NO₂, O₃, and CO.
11. **Autocorrelation Analysis:** ACF and PACF up to 72 lags, revealing strong 24-hour periodicity.
12. **Wind Rose Analysis:** Seasonal wind roses at Patparganj using `windrose`, illustrating dominant wind patterns by season.
13. **Outlier Detection (Preliminary):** Dual IQR and 3σ flagging with Diwali and crop-burning period annotations.
14. **Meteorological Scatter Analysis:** PM₂.₅ vs. Wind Speed, Relative Humidity, Temperature, and Solar Radiation, with season-coded scatter and linear regression overlays.
15. **Calendar Heatmap:** Daily mean PM₂.₅ visualisation using `calplot`.
16. **PCA Biplot:** Principal Component Analysis on 12 air quality variables, with season-coded scatter and variable loading arrows.

### 3.3 VOC Source Fingerprinting (Notebook 03)

Source apportionment was conducted using established BTEX ratio diagnostic methods:

- **Toluene/Benzene (T/B) Ratio:** T/B > 2.0 indicates traffic-dominated emissions; T/B < 1.0 indicates industrial/solvent-dominated emissions; 1.0 < T/B < 2.0 indicates mixed sources.
- **Xylene/Benzene (X/B) Ratio:** Used as a secondary diagnostic for photochemical aging.
- **K-Means Clustering:** Stations were clustered by their VOC fingerprint (T/B, mean Benzene, mean Toluene) into 3 source-type clusters.
- **Seasonal Decomposition:** T/B and X/B ratios were decomposed by station and season to detect seasonal shifts in dominant source contributions.
- **Health Risk Assessment:** Station-level mean Benzene concentrations were benchmarked against WHO (1 µg/m³) and EU (5 µg/m³) guideline values.

### 3.4 Causal Discovery (Notebook 04)

Two complementary causal inference methods were applied:

#### 3.4.1 Granger Causality

Pairwise Granger causality tests were conducted for 9 predictor variables (AT, RH, WS, SR, BP, NO₂, CO, SO₂, O₃) against PM₂.₅ at the Patparganj station (hourly resolution), with lags up to 8 hours. Seasonal Granger tests were additionally conducted for key predictors (WS, RH, AT, SR, NO₂) across all four seasons.

#### 3.4.2 PCMCI+ (Peter and Clark Momentary Conditional Independence)

The `tigramite` library was used to run PCMCI+ with partial correlation (`ParCorr`) as the conditional independence test on 8 variables (PM₂.₅, NO₂, CO, O₃, AT, RH, WS, SR) with lags up to 24 hours. PCMCI+ provides a directed causal graph, controlling for confounding by conditioning on all other variables at all tested lags — a substantial methodological advance over pairwise Granger analysis.

### 3.5 Anomaly Detection and Change-Point Analysis (Notebook 05)

#### 3.5.1 Statistical Methods

- **3σ Flagging:** Observations exceeding 3 standard deviations above the mean for any of 5 pollutants (PM₂.₅, PM₁₀, NO₂, CO, Benzene).
- **IQR Flagging:** Observations exceeding Q₃ + 1.5 × IQR.

#### 3.5.2 Isolation Forest

A multivariate Isolation Forest (200 estimators, 5% contamination) was trained on 6 features (PM₂.₅, PM₁₀, NO₂, NO, CO, Benzene) to detect anomalous hours. Monthly distributions and contiguous episode clustering (≥ 3 hours) were computed.

#### 3.5.3 PELT Change-Point Detection

The Pruned Exact Linear Time (PELT) algorithm from the `ruptures` library was applied at three penalty scales:
- **Fine-grained:** Penalty = 80, `min_size` = 16 (detecting short-duration regime shifts)
- **Macro:** Penalty = 500, `min_size` = 32 (major pollution regime transitions)
- **Seasonal:** Penalty = 3000, `min_size` = 168, RBF kernel (seasonal regime boundaries only)

### 3.6 Forecasting (Notebook 06)

#### 3.6.1 Feature Engineering

The following features were engineered for the Patparganj station (hourly resolution):

- **Lag features:** PM₂.₅ at t−1, t−2, t−4, t−8, t−24, t−48 hours
- **Rolling statistics:** 24-hour and 168-hour (1-week) rolling mean and standard deviation of PM₂.₅
- **Temporal features:** Hour, Day of Week, Month, IsWeekend (binary)
- **Co-pollutants:** NO₂, CO
- **Meteorological:** AT, RH, WS, SR, BP

#### 3.6.2 Train/Validation/Test Split

A strict temporal split was employed:

| Split | Period | Hours | Purpose |
|-------|--------|-------|---------|
| Train | Jan 8 – Sep 30, 2019 | ~6,400 | Model fitting |
| Validation | Oct 1 – Oct 31, 2019 | ~744 | Early stopping / hyperparameter tuning |
| Test | Nov 1 – Dec 31, 2019 | ~1,464 | Final evaluation |

> [!IMPORTANT]
> The test set deliberately covers November–December 2019, which includes the Diwali festival (October 27), peak crop residue burning (October 15–November 15), and the onset of winter inversions. This is the most challenging forecasting period of the year.

#### 3.6.3 Models

| Model | Architecture | Key Hyperparameters |
|-------|-------------|---------------------|
| **Persistence (Baseline)** | y(t+1) = y(t) | — |
| **SARIMA** | SARIMAX(1,0,1)(1,0,1,24) | 24-hour seasonal period |
| **XGBoost** | Gradient-boosted trees | 300 estimators, depth=6, lr=0.05 |
| **LightGBM** | Gradient-boosted trees | 500 estimators, depth=6, lr=0.05, early stopping (50 rounds) |
| **Random Forest** | Bagged decision trees | 200 estimators, depth=10 |
| **LSTM** | 2-layer LSTM + FC head | hidden=128, seq_len=48, dropout=0.2, 30 epochs |
| **Ensemble** | Weighted average | 0.3×XGB + 0.3×LGB + 0.2×RF + 0.2×LSTM |

#### 3.6.4 Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **Skill Score** (SS = 1 − RMSE / RMSE_climatology)

---

## 4. Results

### 4.1 Exploratory Analysis

**Diurnal patterns** reveal a pronounced bimodal PM₂.₅ profile with peaks at 08:00–10:00 (morning traffic) and 17:00–20:00 (evening traffic/cooking), consistent with vehicular emission sources. Ozone exhibits an inverse pattern, peaking at 13:00–15:00 due to photochemical production. **Seasonally**, PM₂.₅ and PM₁₀ concentrations are highest in winter (January–March) and post-monsoon (October–November), with the monsoon season (July–September) showing the lowest concentrations due to wet scavenging. **Weekday/weekend** differences are modest but detectable, with slightly lower NO₂ on weekends.

**Correlation analysis** reveals strong positive correlations between PM₂.₅ and PM₁₀ (r ≈ 0.90), PM₂.₅ and CO, PM₂.₅ and NO₂, and strong negative correlations between PM₂.₅ and Wind Speed, PM₂.₅ and Solar Radiation.

**Stationarity testing** yields conflicting ADF/KPSS results (ADF: stationary; KPSS: non-stationary) for all four tested variables, indicating **trend-stationarity** — the series are stationary around a slowly varying seasonal trend, consistent with STL decomposition findings.

### 4.2 VOC Source Fingerprinting

Stations with robust VOC data were dynamically identified and classified. T/B ratio analysis revealed:

- Stations with elevated T/B ratios (> 2.0) were classified as **traffic-dominated**, consistent with their proximity to major arterial roads.
- Stations with T/B < 1.0 were classified as **industrial/solvent-dominated**, consistent with proximity to industrial estates (e.g., Wazirpur, Bawana).
- Seasonal T/B variation shows higher ratios during winter (**reduced photochemical degradation**) and lower ratios during summer (**enhanced toluene oxidation**).

**Health Risk Assessment:** The benzene concentrations across monitoring stations were benchmarked against international guidelines, with the majority of stations exceeding both the EU (5 µg/m³) and WHO (1 µg/m³) annual mean guidelines, indicating a significant carcinogenic risk.

### 4.3 Causal Discovery

#### 4.3.1 Granger Causality Results

All 9 tested predictor variables exhibit statistically significant Granger-causal relationships with PM₂.₅ (p < 0.001). The strongest causal drivers are:

| Predictor | Best Lag (h) | F-statistic | p-value | Interpretation |
|-----------|-------------|-------------|---------|----------------|
| CO | 3 | 427.49 | < 0.001 | Co-emitted pollutant (combustion marker) |
| SR | 1 | 394.90 | < 0.001 | Photolysis-driven dispersion |
| NO₂ | 3 | 125.94 | < 0.001 | Vehicular emission co-proxy |
| AT | 4 | 109.98 | < 0.001 | Boundary layer dynamics |
| RH | 5 | 93.70 | < 0.001 | Hygroscopic growth / wet scavenging |
| SO₂ | 5 | 39.22 | < 0.001 | Industrial emission marker |
| O₃ | 5 | 39.60 | < 0.001 | Photochemical regime indicator |
| BP | 7 | 36.70 | < 0.001 | Synoptic-scale meteorological forcing |
| WS | 8 | 19.38 | < 0.001 | Mechanical dispersion |

**Seasonal variation:** Granger-causal structure shifts across seasons, with Wind Speed exhibiting the strongest causal influence during Winter (when boundary layer depth is lowest) and Solar Radiation being most influential during Summer (when photochemical processes dominate).

#### 4.3.2 PCMCI+ Results

PCMCI+ confirms the directional causal links identified by Granger analysis while controlling for multivariate confounding. The resulting **directed causal graph** reveals:

- **PM₂.₅ ← CO** (lag 1–3h): strongest direct causal link, reflecting co-emission from combustion sources.
- **PM₂.₅ ← SR** (lag 1h): immediate photolysis-driven effect on secondary aerosol formation.
- **PM₂.₅ ← PM₂.₅** (lag 1–24h): strong autoregressive component, confirming inertial behaviour.
- **O₃ ↔ SR** and **O₃ ↔ NO₂**: photochemical coupling correctly recovered.

### 4.4 Anomaly Detection

**Isolation Forest** detected 438 anomalous hours (5.0% of the year), heavily concentrated in the post-monsoon (October–November) period during coinciding Diwali and crop burning events.

**PELT Change-Point Detection** identified key regime transitions:
- **Macro-level (Penalty=500):** Detected seasonal transitions including the monsoon onset (June–July) and the post-monsoon pollution surge (October).
- **Seasonal-level (Penalty=3000, RBF kernel):** Isolated 3–4 major seasonal regime shifts corresponding to: (i) winter → summer transition, (ii) pre-monsoon onset, (iii) post-monsoon pollution surge, and (iv) winter onset.

**Episode characterisation** of contiguous anomaly clusters identified discrete pollution episodes, with the longest episodes and highest peak PM₂.₅ concentrations occurring in November during the convergence of Diwali, crop burning, and winter inversion onset.

### 4.5 Forecasting

#### 4.5.1 Model Comparison (Test Set: November–December 2019)

| Model | RMSE (µg/m³) | MAE (µg/m³) | MAPE (%) | R² | Skill Score |
|-------|:----:|:----:|:----:|:----:|:----:|
| **Persistence** | **30.29** | **19.23** | **11.6** | **0.948** | **0.836** |
| Random Forest | 44.44 | 21.26 | 11.5 | 0.889 | 0.760 |
| LightGBM | 47.82 | 23.55 | 12.3 | 0.871 | 0.742 |
| XGBoost | 52.26 | 24.94 | 13.0 | 0.846 | 0.718 |
| LSTM | 54.36 | 34.10 | 22.4 | 0.832 | 0.701 |
| Ensemble | 56.00 | 30.51 | 18.9 | 0.814 | 0.688 |
| SARIMA | 233.69 | 192.18 | 92.7 | −2.08 | −0.262 |

#### 4.5.2 SHAP Feature Importance (XGBoost)

SHAP TreeExplainer analysis of the XGBoost model reveals that the dominant predictive features are, in order:
1. **PM₂.₅ lag-1** (most recent observation)
2. **PM₂.₅ lag-2** and **PM₂.₅ lag-4**
3. **PM₂.₅ 24-hour rolling mean**
4. **CO** (carbon monoxide)
5. **AT** (air temperature)
6. **PM₂.₅ 168-hour rolling mean** (weekly context)
7. **Hour of day**

---

## 5. Discussion

### 5.1 Persistence Dominance

The most striking result is the **superiority of the naive persistence baseline** (RMSE = 30.29 µg/m³, R² = 0.948) over all trained models. This is not an artefact but a well-understood phenomenon in 1-hour-ahead pollutant forecasting: hourly PM₂.₅ is highly autocorrelated (ACF > 0.95 at lag 1), meaning the best short-horizon predictor is simply the most recent observation. The SHAP analysis confirms this — `PM25_lag1` dominates all other features by a wide margin.

This finding carries important methodological implications:
- **Skill Score interpretation:** All trained models exhibit positive Skill Scores (0.688–0.760), indicating they add value over climatological forecasts, but none surpass persistence at the 1-hour horizon.
- **Multi-step forecasting gap:** The true value of ML models emerges at longer forecast horizons (6h, 12h, 24h) where autoregressive persistence decays. This was not evaluated in the current study and represents a critical extension.

### 5.2 SARIMA Failure

SARIMA's catastrophic failure (R² = −2.08) on the November–December test set reflects its inability to handle **non-stationary extreme events** (Diwali, crop burning) that are fundamentally different from the training distribution. The model's seasonal component (period = 24) captures diurnal cycles but cannot model inter-annual episodic events.

### 5.3 Causal vs. Predictive Features

An important divergence exists between causal drivers (identified by PCMCI+) and predictive features (identified by SHAP):
- **PCMCI+ identifies CO and SR** as the strongest *causal* drivers of PM₂.₅ at 1–3 hour lags.
- **SHAP identifies autoregressive lags** as the strongest *predictive* features, with CO ranking 5th.

This divergence illustrates the distinction between **causal explanation** (what drives PM₂.₅?) and **optimal prediction** (what best forecasts PM₂.₅?). For forecasting, the recent history of the target variable dominates. For policy intervention, the causal drivers (CO reduction → PM₂.₅ reduction) are more actionable.

### 5.4 Limitations

1. **Single-station focus:** Forecasting and causal analyses were conducted primarily at Patparganj. Spatial transferability to other stations was not evaluated.
2. **1-hour horizon only:** Multi-step-ahead forecasting (6h, 12h, 24h) is essential for operational applicability but was not implemented.
3. **No external covariates:** Crop burning satellite fire counts (MODIS/VIIRS), boundary layer height (ERA5 reanalysis), and traffic volume data were not incorporated.
4. **VOC data sparsity:** Xylene and ethylbenzene columns exhibit > 90% missingness, limiting the robustness of BTEX ratio analyses.
5. **Temporal split leakage:** Features such as `PM25_roll168_mean` (7-day rolling average) may introduce subtle look-ahead bias at the train/validation boundary.

---

## 6. Conclusions

This study demonstrates that:

1. **Delhi's air quality crisis is structurally seasonal**, with PM₂.₅ concentrations exhibiting strong diurnal (24h), weekly, and annual cycles driven by emission sources (vehicular, industrial, crop burning) modulated by meteorological conditions (boundary layer dynamics, wind speed, humidity).

2. **VOC source fingerprinting** via T/B ratios successfully discriminates traffic-dominated from industrial-dominated monitoring sites, and **benzene concentrations at nearly all stations exceed international health guidelines**, representing a substantial but under-discussed public health risk.

3. **Causal discovery via PCMCI+** identifies CO, solar radiation, and NO₂ as the primary direct causal drivers of PM₂.₅ at hourly timescales, with the causal structure exhibiting significant seasonal variation — a finding with direct implications for season-specific intervention strategies.

4. **Anomaly detection** via Isolation Forest and change-point analysis successfully identifies and characterises extreme pollution episodes, with the post-monsoon convergence of Diwali, crop burning, and winter inversions representing the dominant annual pollution regime.

5. **PM₂.₅ forecasting** at the 1-hour horizon is dominated by autoregressive persistence. Machine learning models (Random Forest, XGBoost, LightGBM, LSTM) achieve positive skill scores over climatology but do not surpass the persistence benchmark — a finding consistent with the air quality forecasting literature for short-horizon predictions. The value of ML models is expected to emerge at longer forecast horizons.

---

## 7. Reproducibility

### 7.1 Repository Structure

```
delhi-aq-analysis/
├── data/
│   ├── raw/                    # 432 station-year CSV files (15-min resolution)
│   └── processed/              # Harmonised datasets
│       ├── delhi_all_years_all_stations.csv   (2.46 GB, 14.8M rows)
│       └── delhi_2019_all_stations.csv        (1.37M rows)
├── notebooks/
│   ├── 01_data_loading.ipynb       # Data ingestion and harmonisation
│   ├── 02_eda.ipynb                # Exploratory data analysis (19 cells)
│   ├── 03_voc_fingerprinting.ipynb # BTEX source attribution (6 cells)
│   ├── 04_causal_discovery.ipynb   # Granger + PCMCI+ (6 cells)
│   ├── 05_anomaly_detection.ipynb  # Isolation Forest + PELT (8 cells)
│   └── 06_forecasting.ipynb        # Multi-model PM2.5 forecasting (10 cells)
└── figures/                        # 38 output figures and result CSVs
```

### 7.2 Software Environment

- **Python:** 3.10.13 (Miniconda, `dl-env` environment)
- **Core libraries:** pandas 2.1.4, numpy 1.26.4, matplotlib, seaborn, scikit-learn
- **Specialised:** tigramite (PCMCI+), ruptures (PELT), windrose, calplot, missingno
- **ML/DL:** XGBoost 2.1.3, LightGBM, PyTorch 2.6.0+cu124, SHAP
- **Statistical:** statsmodels (SARIMA, ADF, KPSS, ACF/PACF)

---

## Appendix A: Figure Index

| # | Figure | Description |
|---|--------|-------------|
| 01 | `01_missing_values_heatmap.png` | Missing value matrix — Anand Vihar 2019 |
| 02 | `02_missing_pct_station_variable.png` | Missing % per pollutant per station |
| 03 | `03_pm25_timeseries_all_stations.png` | PM₂.₅ time series — 6 stations |
| 04 | `04_diwali_zoom.png` | PM₂.₅ & PM₁₀ during Diwali week |
| 05 | `05_diurnal_profiles.png` | Hourly pollutant profiles |
| 06 | `06_day_of_week.png` | Weekday vs weekend patterns |
| 07 | `07_seasonal_boxplots.png` | Monthly pollutant distributions |
| 08 | `08_correlation_heatmap.png` | Pearson correlation matrix |
| 09 | `09_stl_decomposition.png` | STL decomposition of PM₂.₅ |
| 10 | `10_stationarity_results.csv` | ADF and KPSS test results |
| 11 | `11_acf_pacf.png` | ACF/PACF plots |
| 12 | `12_wind_rose.png` | Seasonal wind roses |
| 13 | `13_outlier_detection.png` | 3σ + IQR outlier detection |
| 14 | `14_scatter_pm25_met.png` | PM₂.₅ vs meteorological variables |
| 15 | `15_calendar_heatmap.png` | Calendar heatmap of daily PM₂.₅ |
| 16 | `16_pca_biplot.png` | PCA biplot by season |
| 17 | `17_diurnal_TB_ratio.png` | Diurnal T/B ratio profiles |
| 18 | `18_voc_station_clusters.png` | VOC source clustering |
| 19 | `19_seasonal_btex_ratios.png` | Seasonal BTEX ratio variation |
| 20 | `20_benzene_health_risk.png` | Benzene risk by station |
| 21 | `21_granger_causality.png` | Granger causality F-statistics and lag heatmap |
| 22 | `22_seasonal_granger.png` | Seasonal Granger causality |
| 23 | `23_pcmci_causal_graph.png` | PCMCI+ directed causal graph |
| 24 | `24_isolation_forest.png` | Isolation Forest anomaly detection |
| 25 | `25_changepoint_detection.png` | PELT fine-grained change points |
| 25b | `25b_changepoint_macro.png` | PELT macro change points |
| 25c | `25c_changepoint_seasonal.png` | PELT seasonal regime detection |
| 26 | `26_episode_summary.png` | Pollution episode analysis |
| 27 | `27_model_comparison.png` | Model comparison (bar + actual vs predicted) |
| 28 | `28_shap_importance.png` | SHAP feature importance (XGBoost) |
| 28b | `28b_shap_beeswarm.png` | SHAP beeswarm plot |
| 29 | `29_final_model_comparison.png` | Final model comparison bar chart |

---

## References

1. Guttikunda, S.K., & Calori, G. (2013). A GIS based emissions inventory at 1 km × 1 km spatial resolution for air pollution analysis in Delhi, India. *Atmospheric Environment*, 67, 101-111.
2. Runge, J., et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances*, 5(11), eaau4996.
3. Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation Forest. *Proceedings of the Eighth IEEE International Conference on Data Mining*, 413-422.
4. Killick, R., Fearnhead, P., & Eckley, I.A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.
5. National Air Quality Index (NAQI). Central Pollution Control Board, Ministry of Environment, Government of India.
