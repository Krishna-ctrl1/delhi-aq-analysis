# Delhi Air Quality Presentation — Full Script & Viva Prep
### Advanced Data Analytics | IIIT Sri City | April 2026
### Krishna Gupta (S20230010124) & Kumarran Mahesh (S20230030387)

# PART 1 — KRISHNA GUPTA'S SCRIPT

---

## SLIDE 1 — TITLE (Krishna)
*[Stand, click to slide, pause 2 seconds, then speak]*

> "Good morning everyone. My name is Krishna Gupta, and I'm presenting alongside my partner Kumarran Mahesh. Our project is titled **'Multi-Modal Characterisation, Causal Attribution, and Predictive Modelling of Ambient Air Quality in Delhi, India — 2009 to 2025.'**
>
> This is a course project for Advanced Data Analytics. What makes this work distinct is not just what we analysed, but **how we integrated four different analytical streams** — source fingerprinting, causal inference, anomaly detection, and machine learning forecasting — into a single unified pipeline applied to the same dataset. No prior study on Delhi's air quality has done all four together.
>
> I'll cover the first half — the motivation, data, and the two source and causal research questions. Kumarran will take over from Slide 7."

**[Why you say this]:** Sets up the novelty claim immediately. Professors respect when students know *what* makes their work original.

---

## SLIDE 2 — THE CRISIS (Krishna)

> "Let me start with why Delhi's air quality is a systems-modelling problem, not just an environmental concern.
>
> PM2.5 refers to **Particulate Matter smaller than 2.5 micrometres** — small enough to bypass the nose and throat and lodge directly in the lungs and bloodstream. The WHO's safe annual guideline is 5 micrograms per cubic metre. Delhi's annual average exceeds **100 micrograms per cubic metre** — that is **20 times the safe limit**.
>
> More than **30 million people** are chronically exposed. And the problem isn't just chronic — it spikes catastrophically. During October and November, three sources converge simultaneously: **Diwali fireworks**, **agricultural residue burning in Punjab and Haryana** — farmers burn their stubble before the rabi season — and **meteorological temperature inversions** that trap pollutants at ground level like a lid on a pot. PM2.5 has been recorded above 800 micrograms per cubic metre during these episodes.
>
> The bottom line you can see from the time-series pattern across all six stations: **every single station chronically breaches India's own NAAQS standard of 60 micrograms per cubic metre**. The standard is aspirational, not operational.
>
> The research gap we identified: existing studies treat source attribution, causal inference, anomaly detection, and forecasting as isolated problems on different datasets. We close that gap."

**[Why this matters for viva]:** The prof may ask *"What is PM2.5 exactly?"* or *"What causes the October spike?"*. This script answers both.

---

## SLIDE 3 — RESEARCH FRAMEWORK (Krishna)

> "We structured our entire pipeline around four Research Questions, which you can see here.
>
> **RQ1** asks: can we use VOC chemical ratios to fingerprint whether a monitoring station is near traffic, industry, or mixed sources?
>
> **RQ2** asks: what is the *actual causal structure* linking weather and co-pollutants to PM2.5 — and crucially, does that structure change by season?
>
> **RQ3** asks: can machine learning identify the precise dates and durations of extreme pollution episodes automatically?
>
> **RQ4** asks: among seven forecasting models — from a naive baseline up to deep learning — which one actually wins, and does any ML model beat simply guessing that the next hour equals the current hour?
>
> These four questions are connected. The causal graph from RQ2 tells us *which variables matter* for RQ4. The anomaly episodes from RQ3 explain *why* some models fail in RQ4. The source fingerprints from RQ1 tell us *what to target* in policy.
>
> Our data pipeline has six stages: Harmonisation → EDA → VOC Fingerprinting → Causal Discovery → Anomaly Detection → Forecasting. All applied to the same 14.8 million observations from 38 stations."

**[Why this matters for viva]:** The prof may ask *"Why these four RQs specifically?"* Answer: they cover the four analytical gaps identified in the literature — fragmentation, causal naivety, missing anomaly catalogues, and inflated ML claims.

---

## SLIDE 4 — DATA & EDA (Krishna)

> "Our dataset comes from the Central Pollution Control Board, the Delhi Pollution Control Committee, and the India Meteorological Department. 38 stations, 25 parameters — 9 criteria pollutants, 6 VOCs, 10 meteorological — at 15-minute resolution from 2009 to 2025. After harmonisation that's **14.8 million observations**.
>
> For our primary analysis we focused on **2019** — it had the highest network completeness, with 37 of 38 stations active and the lowest missingness. Our focal station is **Patparganj** — selected systematically based on having the most complete data across all pollutant AND meteorological channels. That mattered for the PCMCI+ causal analysis, which needs all variables.
>
> Let me walk you through the two EDA visuals on this slide.
>
> **The Wind Rose** — this shows wind direction and speed broken into four seasons. The size of each petal tells you how frequently wind blows from that direction; the colour tells you wind speed. In **winter**, the dominant direction is **North-West** — this is directly from Punjab and Haryana, meaning the crop-burning smoke travels straight into Delhi. In **monsoon**, you see **South-West** winds at high speeds — the monsoon circulation both changes the direction and the high speeds help disperse and scavenge pollutants. That's why air quality is cleanest during monsoon.
>
> **The Calendar Heatmap** — each cell is one day of 2019, colour-coded by daily mean PM2.5. Pale yellow is clean air, dark red to maroon is severe. You can immediately see January is dark, then it fades through the summer, the monsoon months are pale yellow, and then October onward — especially November — the cells go very dark again. That single visual tells the whole seasonal story of Delhi's air quality."

**[Why this matters for viva]:** Prof may ask *"Why did you choose Patparganj?"* or *"What does a wind rose actually show?"* This script covers both.

---

## SLIDE 5 — RQ1: VOC SOURCE FINGERPRINTING (Krishna)

> "Now for our first result. RQ1 is about understanding *what kind* of pollution each monitoring station is exposed to — is it mostly from traffic, or from industrial/solvent sources?
>
> We used what are called **BTEX ratios** — Benzene, Toluene, Ethylbenzene, Xylene are a family of Volatile Organic Compounds that come from different combustion and industrial processes. The key ratio is **Toluene to Benzene, or T/B**.
>
> Why does this ratio work? **Traffic exhaust** produces relatively more toluene than benzene, so a T/B above 2.0 indicates traffic. **Industrial solvents and paints** produce more benzene relative to toluene, so T/B below 1.0 indicates industrial sources. These thresholds come from WHO and USEPA established guidelines.
>
> We applied **K-Means clustering** — an unsupervised machine learning algorithm — with k=3, meaning we asked the algorithm to find three natural groupings of stations based on their T/B ratio, mean benzene, and mean toluene. We didn't label the clusters in advance — the algorithm discovered them.
>
> The scatter plot on the left shows the result. The X-axis is T/B ratio, Y-axis is mean toluene concentration. The three clusters are clear: **green dots top-right** — Mundka and Sonia Vihar — are traffic-dominated, near major highways. **Red dots bottom-left** — Patparganj, NSIT Dwarka — are industrial/mixed with low overall VOC but low T/B. **Orange dots** are the majority, mixed low-VOC central stations.
>
> Seasonally: T/B ratios are higher in winter because cold temperatures slow down the photochemical oxidation of toluene, so it accumulates. In summer, toluene degrades faster, pulling the ratio down.
>
> But the finding that I want to emphasise — and which you can see in the bar chart on the right — is the **benzene carcinogen risk**. Benzene is a **Group 1 carcinogen** — meaning it is definitively proven to cause cancer, specifically leukaemia, at chronic exposure. WHO's safe annual mean is 1 microgram per cubic metre. The EU's limit is 5. Nearly every single station in Delhi exceeds the WHO limit. Several exceed even the EU limit. **DTU exceeds 12 micrograms per cubic metre** — twelve times the WHO safe level. And yet benzene does not appear in India's National Air Quality Index. That is a serious public health policy gap."

**[Why this matters for viva]:** Prof may ask *"What is K-Means?"*, *"What are VOCs?"*, *"Why T/B specifically?"*, *"What is benzene's health risk?"*. This script answers all four.

---

## SLIDE 6 — RQ2: CAUSAL DISCOVERY (Krishna)

> "This is the methodologically most sophisticated part of the study. We want to know not just *what correlates with* PM2.5, but *what actually causes it*.
>
> This is a crucial distinction. Correlation is easy to compute — we showed the Pearson heatmap in EDA. But correlation doesn't tell you what to *intervene on* to reduce pollution.
>
> We used two methods. First, **Granger Causality** — a test that asks: does knowing the past values of variable X help me predict variable Y *beyond what Y's own past already tells me*? If yes, we say X Granger-causes Y. This is a statistical definition of causation, not a philosophical one. We tested 9 predictors against PM2.5 at lags 1 through 8 hours. All 9 were significant at p less than 0.001. The top result: **CO at lag 3 hours, F-statistic 427**. Solar radiation at lag 1 hour, F=395.
>
> But Granger has a critical flaw — it does **pairwise** tests. So it will call something causal if it correlates with PM2.5, even if that correlation is *entirely due to a third variable* they share. For example, barometric pressure appeared to Granger-cause PM2.5. But is that actually true, or are both BP and PM2.5 just responding to temperature?
>
> To answer this, we applied **PCMCI+** — the Peter-Clark Momentary Conditional Independence Plus algorithm, developed by Jakob Runge at DLR. PCMCI+ performs conditional independence testing — it asks: is the X→Y link still significant *after conditioning on all other variables simultaneously*? This is the gold standard for time-series causal discovery.
>
> We ran it on 8 standardised variables at hourly resolution with lags up to 24 hours and a significance threshold of 0.05.
>
> **The confirmed causal links are:**
> - PM2.5 ← CO at lag 1 to 3 hours: this is the strongest direct causal link. CO is a co-emission from combustion — wherever there's vehicle exhaust or biomass burning producing PM2.5, CO comes with it, slightly ahead in time.
> - PM2.5 ← Solar Radiation at lag 1 hour: solar radiation drives photolytic chemistry that breaks down secondary aerosols. Higher SR reduces PM2.5 concentrations an hour later.
> - PM2.5 ← PM2.5 at lags 1 through 24 hours: the strong autoregressive component — the signal has enormous inertia.
> - O3 and SR and NO2 are bidirectionally coupled — this is the correct photochemical cycle being recovered automatically.
>
> And critically: the **BP→PM2.5 link was ELIMINATED** once we conditioned on temperature. It was a confounded relationship — spurious in the pairwise test but absent when we control for confounders.
>
> One more finding: **causal structure is NOT the same across seasons**. In winter, wind speed dominates — because the boundary layer is shallow and mechanical dispersion matters most. In monsoon, relative humidity dominates — wet scavenging. In summer, solar radiation dominates — photolysis. In post-monsoon, temperature and wind jointly dominate the inversion onset. This means season-generic pollution policies are systematically suboptimal.
>
> This is the first application of PCMCI+ to Delhi's CAAQM monitoring network — a novel contribution."

**[Why this matters for viva]:** This slide will attract the most questions. Likely viva questions and answers are in the Viva Prep section at the end of this document.

---
---

# PART 2 — KUMARRAN MAHESH'S SCRIPT

---

## SLIDE 7 — RQ3: ANOMALY DETECTION (Kumarran)

> "Thank you Krishna. I'll be covering the anomaly detection, forecasting, and our key conclusions.
>
> For RQ3, we wanted to automatically identify *when* Delhi experienced extreme pollution episodes, and *what* the structural pollution regimes are across the year — without manually labelling dates.
>
> We used a portfolio of four methods working at different scales.
>
> First, **3-sigma and IQR flagging** — these are classical statistical methods. 3-sigma flags any observation more than 3 standard deviations above the mean. IQR flags anything above Q3 plus 1.5 times the interquartile range. These are per-variable, univariate — they catch obvious extreme values but miss subtle multi-variable anomalies.
>
> Second, and most powerful: **Isolation Forest**. This is an unsupervised ML algorithm. The idea is elegant: *anomalies are easier to isolate than normal points*. The algorithm builds many random decision trees; an anomalous point gets isolated in very few splits because it's far from the main data cloud. We ran it with 200 trees and a contamination rate of 5% — meaning we expected about 5% of hours to be anomalous. It flagged **438 hours across 2019 — exactly 5.0%**.
>
> When we look at *when* those 438 hours occur: heavily concentrated in **October and November** — the Diwali and crop-burning convergence period. January also has a cluster. The monsoon months June through September have near-zero flagged hours. This is exactly what we'd expect physically, which validates the method.
>
> Third: **PELT — Pruned Exact Linear Time** change-point detection. PELT is a dynamic programming algorithm that finds the optimal set of change-points in a time series by minimising a penalised cost function. The penalty parameter controls how many change-points you detect — higher penalty means fewer, bigger changes.
>
> We ran PELT at **three penalty scales** to resolve different temporal structures simultaneously:
> - Fine scale, penalty 80: finds about 57 change-points — short-duration episode boundaries, individual pollution events
> - Macro scale, penalty 500: finds 6 to 8 change-points — major seasonal transitions
> - Seasonal scale, penalty 3000 with RBF kernel: finds 3 to 4 — fundamental annual regime boundaries
>
> The seasonal PELT result is striking: the October change-point coincides **precisely** with the Diwali and stubble-burning onset. This means we can use PELT on historical data to derive *data-driven* activation dates for GRAP — India's Graded Response Action Plan — rather than relying on calendar dates alone."

**[Why this matters for viva]:** Prof may ask *"How does Isolation Forest work?"*, *"What is PELT?"*, *"What is contamination parameter?"*. This script covers all three.

---

## SLIDE 8 — RQ4: FORECASTING (Kumarran)

> "For RQ4, we built and compared seven PM2.5 forecasting models at the 1-hour-ahead horizon, all trained and tested on identical chronological data splits with no data leakage.
>
> Before I show the results, let me briefly describe each model family:
>
> **Persistence baseline** — the simplest possible forecast: predict that the next hour equals the current hour. This is our benchmark. Any model that can't beat this has no practical value.
>
> **SARIMA** — Seasonal AutoRegressive Integrated Moving Average. A classical statistical model that combines a differenced autoregressive component with a moving average component, plus a seasonal component at period 24 hours. Good for stationary time series with regular seasonality.
>
> **XGBoost** — eXtreme Gradient Boosting. An ensemble of shallow decision trees trained sequentially, where each tree corrects the errors of the previous one. Very strong on tabular data.
>
> **LightGBM** — similar to XGBoost but uses a histogram-based approach that is faster and handles larger datasets better.
>
> **Random Forest** — an ensemble of deep decision trees trained in parallel on random subsets of data and features. Averages predictions for variance reduction.
>
> **LSTM** — Long Short-Term Memory network. A type of recurrent neural network with gating mechanisms that control what information to remember across time steps. Theoretically suited for long-range temporal dependencies.
>
> **Ensemble** — a weighted combination: 30% XGBoost + 30% LightGBM + 20% Random Forest + 20% LSTM.
>
> Our feature engineering included PM2.5 lags at 1, 2, 4, 8, 24, and 48 hours; 24-hour and 168-hour rolling statistics; hour, day-of-week, month, weekend indicator; and co-pollutants NO2 and CO plus meteorological variables.
>
> The test set was **November and December 2019** — the peak pollution period with Diwali and crop burning. This is deliberately the hardest possible test.
>
> **Now the results — and they are counterintuitive.**
>
> The **naive persistence model wins**. RMSE of 30.3 micrograms, R-squared 0.948, skill score 0.836. Random Forest comes second at RMSE 44.4. Every trained ML model is worse than just saying 'the next hour equals this hour.'
>
> SARIMA catastrophically fails — R-squared of negative 2.08. A model with negative R-squared is performing *worse than simply predicting the mean*. Why? Because SARIMA assumes stationarity — a stable mean and variance. The Diwali and crop-burning period violates this completely. The time series jumps from 80 micrograms to 900 micrograms in a matter of hours. SARIMA cannot model that.
>
> **Why does persistence win?** It comes down to autocorrelation. If you look at the ACF of PM2.5 at lag 1, it's above 0.95. The series barely moves hour-to-hour. Any trained model trying to learn patterns is fighting against raw inertia. The 'pattern' at 1 hour is just: it's still high because it was high. Persistence captures that perfectly.
>
> The implication: prior Delhi forecasting studies that claimed ML superiority without reporting a persistence baseline were making inflated claims. Our study is, to the best of our knowledge, the first to benchmark all 7 models together with the persistence baseline."

**[Why this matters for viva]:** This slide will get major questions. See viva prep section.

---

## SLIDE 9 — KEY INSIGHT: PREDICTION ≠ CAUSATION (Kumarran)

> "This slide connects RQ2 and RQ4 and contains what I consider the deepest insight of the study.
>
> We can ask two completely different questions about what 'matters' for PM2.5:
> 1. What features best *predict* PM2.5 one hour ahead?
> 2. What variables *causally drive* PM2.5?
>
> The SHAP beeswarm on the right answers question 1. **SHAP — SHapley Additive exPlanations** — is a game-theoretic framework from cooperative game theory. The idea: treat each feature as a 'player' in a coalition. The SHAP value for a feature is its average marginal contribution to the prediction across all possible orderings of the other features. This gives us a fair, mathematically grounded attribution of how much each feature moved the model's prediction.
>
> The beeswarm shows that **PM2.5_lag1 dominates completely** — it extends all the way to SHAP values of plus 300. Every other feature barely moves the needle by comparison. CO appears at rank 2 in SHAP, but look at its spread — it's tiny compared to lag1. PM2.5's own past value contributes more predictive power than all other features combined.
>
> Now look at the left table — the divergence matrix. PCMCI+ answers question 2: causal drivers. CO is rank 1 causally. Solar radiation is rank 2 causally. PM2.5 lag-1 appears at rank 3 causally — it's a genuine autoregressive self-cause, but it's not the top causal driver.
>
> These two rankings are not contradictory. They answer different questions. For a **forecaster**, PM2.5's own past is the right thing to use — it's highly predictable from itself. But for a **policymaker** who wants to *reduce* PM2.5, targeting PM2.5's past values is meaningless — you can't change the past. The actionable causal levers are **CO reduction** through vehicular emission controls and open burning bans, and **SR-mediated** ventilation through urban canopy design and air corridor planning.
>
> This is the core distinction: *effective prediction does not equal causal understanding*. They pull from different information structures in the data."

**[Why this matters for viva]:** *"What is SHAP?"*, *"Why is prediction different from causation?"*, *"What does autoregressive mean?"* — all answered here.

---

## SLIDE 10 — KEY CONTRIBUTIONS (Kumarran)

> "Let me quickly summarise our five novel contributions, because these are the things that make this work distinct from what already exists in the Delhi AQ literature.
>
> **Contribution 1**: First application of PCMCI+ to Delhi's CAAQM monitoring network. Every prior causal study we found used pairwise Granger tests, which we showed yield false positives. The directed causal graph with seasonal decomposition is new.
>
> **Contribution 2**: Honest persistence benchmarking. We are, to our knowledge, the first to compare all seven model families including the naive baseline in a single study on Delhi data. Prior studies made inflated claims.
>
> **Contribution 3**: The Prediction-Causation Divergence Matrix — explicitly linking SHAP feature importance to PCMCI+ causal rankings. This analytical cross-linkage is original.
>
> **Contribution 4**: Multi-scale PELT at three penalty levels simultaneously resolving episodic, transitional, and seasonal structure. Not just 'change points exist' but a hierarchical decomposition of the pollution year.
>
> **Contribution 5**: Benzene carcinogen risk mapping across all stations relative to WHO and EU guidelines. The finding that virtually all Delhi stations exceed WHO limits — and that benzene is absent from NAQI — is substantially underreported in the existing literature."

---

## SLIDE 11 — POLICY RECOMMENDATIONS (Kumarran)

> "Our analytical findings directly translate into five policy recommendations.
>
> First: **season-stratified emission controls**. Because our PCMCI+ analysis shows that the causal structure shifts fundamentally between seasons — wind speed in winter, solar radiation in summer, humidity in monsoon — emission control strategies need to be tailored by season, not generic year-round policies.
>
> Second: **CO as the number one intervention target**. With an F-statistic of 427.5 in Granger tests and the strongest cross-MCI link in PCMCI+, CO is the clearest causal pathway to PM2.5. Tighter vehicular emission norms — BS7 standards — and banning open biomass burning are the highest-ROI interventions.
>
> Third: **benzene must enter the NAQI**. India's National Air Quality Index currently includes PM2.5, PM10, NO2, SO2, CO, O3, NH3, and lead. Benzene is absent. Given that every Delhi station exceeds the WHO carcinogen limit, this is a policy failure.
>
> Fourth: **invest in 6 to 24 hour forecasting, not 1 hour**. At 1 hour, persistence is unbeatable. But at longer horizons, the autocorrelation decays, weather patterns have time to evolve, and ML models are expected to provide genuine added value over persistence. That is where operational early-warning systems should be built.
>
> Fifth: **use our PELT episode catalogue as GRAP trigger dates**. Currently GRAP — the Graded Response Action Plan — relies on forecast threshold breaches. Our data-driven episode catalogue, derived objectively from PELT change-points and Isolation Forest clustering, provides complementary activation evidence grounded in historical patterns."

---

## SLIDE 12 — THANK YOU (Kumarran)

> "To summarise: we applied a six-stage analytical pipeline to 14.8 million observations from 38 Delhi stations. Across four research questions we found that CO and solar radiation causally drive PM2.5 beyond correlation; that season-specific causal structure demands season-stratified policy; that the persistence model beats all trained ML at 1-hour horizons; that prediction and causation pull from different information; and that benzene carcinogen risk is present at nearly all Delhi stations but absent from India's regulatory framework.
>
> Our code is available on GitHub. We're happy to take questions. Thank you."

---
---

# VIVA PREPARATION — LIKELY PROFESSOR QUESTIONS

## Category 1: Conceptual Understanding

**Q: What exactly is PM2.5 and why is it more dangerous than PM10?**
> PM2.5 is particulate matter with an aerodynamic diameter below 2.5 micrometres. PM10 is below 10 micrometres. PM2.5 is more dangerous because particles this small can penetrate the lung's alveolar region — the gas-exchange surface — and even enter the bloodstream directly. PM10 is mostly filtered in the upper respiratory tract. PM2.5 is associated with cardiovascular disease, lung cancer, stroke, and premature mortality.

**Q: What is BTEX and why does T/B ratio discriminate source types?**
> BTEX stands for Benzene, Toluene, Ethylbenzene, and Xylene — a group of aromatic hydrocarbons. They come from different sources in different proportions. Vehicle exhaust — especially petrol engines — emits more toluene relative to benzene because toluene has higher vapour pressure and is a major component of petrol. Industrial solvents, paints, and manufacturing processes tend to emit more benzene relative to toluene. So the T/B ratio is a chemical fingerprint of the emission source. Ratios above 2.0 indicate traffic; below 1.0 indicate industrial or solvent sources.

**Q: What is the difference between Granger causality and PCMCI+?**
> Granger causality is a bivariate test — it tests one predictor against the target at a time, without accounting for other variables. This makes it vulnerable to confounding: if variable A correlates with both X and Y, Granger will call X→Y causal even if the real causal path is A→X and A→Y with no direct X→Y link. PCMCI+ is a conditional independence test. It asks: is the link X→Y still statistically significant *after conditioning on all other variables in the system simultaneously*? This eliminates confounded links. PCMCI+ also handles time lags explicitly and can detect instantaneous links. It uses the Peter-Clark algorithm for skeleton discovery followed by MCI tests for causal orientation.

**Q: Why did you choose Patparganj as your focal station?**
> Systematic selection based on data completeness. We assessed all 38 stations for missing data percentages across all 25 variables — both pollutant and meteorological channels. For PCMCI+ to work reliably, we needed a station where all 8 variables had high completeness for the full year. Patparganj had the lowest per-variable missingness in 2019 across both pollutant and meteorological channels simultaneously.

**Q: What is K-Means clustering and how did you choose k=3?**
> K-Means is an unsupervised algorithm that partitions N data points into k clusters by minimising within-cluster sum of squared distances to the cluster centroid. It assigns each point to the nearest centroid, recomputes centroids, and iterates until convergence. We chose k=3 based on the physical interpretation — the BTEX literature identifies three canonical source types (traffic, industrial, mixed) — and we verified this with elbow plot analysis of the within-cluster sum of squares, which showed a clear elbow at k=3.

---

## Category 2: Methodology Deep Dives

**Q: How does Isolation Forest work?**
> Isolation Forest is based on the observation that anomalies are *few and different* from normal points. The algorithm builds an ensemble of random isolation trees. Each tree randomly selects a feature and a random split value to divide the data. An anomalous point, being far from the main data cloud, gets isolated in very few splits — it has a short path length from root to leaf. A normal point requires many splits to isolate. The anomaly score is based on the average path length across all trees — shorter average path = more anomalous. The contamination parameter sets the expected fraction of anomalies, which determines the decision boundary.

**Q: What is PELT and what does the penalty parameter control?**
> PELT — Pruned Exact Linear Time — is a dynamic programming algorithm for optimal change-point detection. It minimises a penalised cost function: total fit error plus a penalty for each additional change-point. The penalty balances overfitting (too many change-points) against underfitting (missing real regime shifts). Higher penalty = fewer, larger changes. We used three scales: fine (80), macro (500), seasonal (3000), to simultaneously resolve short episodes, seasonal transitions, and annual boundaries.

**Q: What is SHAP and why use it instead of feature importance from the model directly?**
> Standard feature importance from tree models (like XGBoost's gain-based importance) is not consistent — it can assign high importance to correlated features arbitrarily. SHAP is based on Shapley values from cooperative game theory. For each prediction, it computes how much each feature contributed by averaging its marginal contribution across all possible subsets of features. This is the only attribution method that satisfies four desirable axioms: efficiency (contributions sum to prediction), symmetry (equal contributions → equal values), dummy (zero contribution → zero SHAP), and linearity. SHAP values are both locally faithful (per prediction) and globally comparable (average across test set).

**Q: Why SARIMA specifically, and why did it fail so catastrophically?**
> SARIMA(1,0,1)(1,0,1)24 was chosen because: the AR(1) and MA(1) components capture the short-range autocorrelation structure indicated by ACF/PACF analysis; the seasonal (1,0,1)24 component captures the 24-hour diurnal cycle we identified in STL decomposition; no differencing (I=0) because ADF tests indicated stationarity. It failed because ARIMA models assume the series is covariance-stationary — constant mean and variance over time. The Diwali/crop-burning period violates this completely: PM2.5 jumps from ~80 to ~900 µg/m³ in a matter of hours, creating a non-stationary shock. SARIMA has no mechanism to model this kind of episodic extreme. Its R² of −2.08 means it's worse than predicting the mean — it confidently predicted typical diurnal patterns while reality was 10× higher.

**Q: What is autocorrelation and why does it explain the persistence ceiling?**
> Autocorrelation is the correlation of a time series with its own past values at various lags. The ACF (AutoCorrelation Function) plot shows this. For hourly PM2.5 at Patparganj, the lag-1 autocorrelation exceeds 0.95. This means: knowing PM2.5 right now tells you 95% of what you need to know about PM2.5 in the next hour. There's only 5% of variance 'left over' for any other predictor to explain. The persistence model exploits this perfectly — it literally predicts PM2.5(t+1) = PM2.5(t). A trained ML model using dozens of features is competing to explain that residual 5%, which is largely noise. That's why persistence wins at 1 hour — the signal-to-noise ratio for 'extra information beyond lag-1' is nearly zero.

---

## Category 3: Results and Implications

**Q: Why is CO the strongest causal driver and not, say, wind speed?**
> CO is a **co-emission** — it is produced by the same combustion processes that produce PM2.5. When you burn biomass (crop residue, fireworks) or fuel (vehicles), you produce both CO and PM2.5 simultaneously. CO actually peaks slightly *before* PM2.5 at lag 1–3 hours because CO is a gas and disperses faster than particles. Wind speed is a *dispersal* mechanism — it affects how fast PM2.5 moves away — but it doesn't cause the pollution to be generated. The causal hierarchy is: emission source → CO + PM2.5 produced simultaneously (CO is a marker), then wind disperses both. CO is a cleaner causal signal because it's less confounded by meteorology than PM2.5 itself.

**Q: Can we actually say PCMCI+ finds causation, or just conditional independence?**
> This is an excellent philosophical question. PCMCI+ is a constraint-based causal discovery algorithm. It identifies the *Markov equivalence class* of causal graphs consistent with the conditional independence structure in the data. Under the faithfulness and causal Markov condition assumptions, this equivalence class contains the true causal graph. The key assumptions: (1) no hidden confounders that are unmeasured, (2) the true causal graph is acyclic at the measurement scale. For short lags in an atmospheric system, these are reasonable approximations. We cannot claim absolute causal truth — but PCMCI+ gives much stronger evidence than correlation or pairwise Granger, and is the best available tool for observational time-series causal discovery.

**Q: If persistence wins at 1 hour, why bother building ML models at all?**
> Three reasons. First, at longer horizons (6h, 12h, 24h) the autocorrelation has decayed significantly — lag-6 ACF is around 0.7, lag-24 is around 0.4 — so ML models should genuinely outperform persistence at those horizons. This is a future work direction. Second, the trained models carry causal and structural information — the SHAP analysis tells us something real about the system even if the model doesn't beat persistence at 1h. Third, for extreme event prediction (not just next-hour point forecasting but binary threshold exceedance), ML models that learn pollution episode signatures may outperform persistence even at 1h, because persistence can't 'see' an episode starting if PM2.5 is still normal at the current timestep.

**Q: What are the limitations of your study?**
> Five main limitations. First, our causal, anomaly, and forecasting analyses focus on a single station — Patparganj. We haven't validated spatial transferability. Second, we only evaluated the 1-hour forecast horizon — we expect ML to outperform persistence at 6–24h but didn't test this. Third, we didn't incorporate satellite data: MODIS AOD, VIIRS fire radiative power, or ERA5 boundary layer height, which could improve both causal and forecasting models. Fourth, VOC data had high missingness — Xylene was over 90% missing — which limits robustness of the X/B ratio analysis. Fifth, rolling window features computed near the train/validation boundary may introduce subtle look-ahead contamination.

---

## Category 4: Quick-fire technical questions

| Question | Short Answer |
|---|---|
| What does NAAQS stand for? | National Ambient Air Quality Standards — India's air quality guidelines |
| What is CAAQM? | Continuous Ambient Air Quality Monitoring — the real-time automated station network |
| What is GRAP? | Graded Response Action Plan — Delhi's emergency pollution response protocol with 4 alert levels |
| What is STL decomposition? | Seasonal-Trend decomposition using Loess — separates time series into trend, seasonal, and residual components |
| What is ADF test? | Augmented Dickey-Fuller test — tests for unit root (non-stationarity) in a time series |
| What is KPSS test? | Kwiatkowski–Phillips–Schmidt–Shin test — tests null hypothesis of stationarity, complementary to ADF |
| Why did you use both ADF and KPSS? | Conflicting results (ADF says stationary, KPSS says non-stationary) indicate trend-stationarity — stationary around a slowly varying seasonal trend |
| What is skill score? | (1 - MSE_model / MSE_persistence) — measures improvement over the persistence baseline |
| What is R²? | Coefficient of determination — fraction of variance explained. R²=1 is perfect, R²=0 means model predicts the mean, R²<0 means model is worse than predicting the mean |
| What is the WHO PM2.5 guideline? | 5 µg/m³ annual mean (revised 2021 from 10 µg/m³) |
| What is India's NAAQS for PM2.5? | 40 µg/m³ annual mean, 60 µg/m³ 24-hour mean |
| What is RMSE vs MAE? | RMSE (Root Mean Squared Error) penalises large errors more (squares them). MAE (Mean Absolute Error) treats all errors equally. RMSE is more sensitive to extreme events — relevant for pollution spikes |
