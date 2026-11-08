# Raqeeb | رقيب: AI triage for abnormal water quality results

Raqeeb reads each abnormal lab result together with lab QC, tank conditions, pipe repairs and customer complaints. It tells the lab:

- the **likely cause**: real network event, sampling error, or analytical (lab) error;
- **how sure** it is, and **why**;
- **what to do first**.

It also ranks the sites most likely to fail on chlorine tomorrow.

> All data in `data/` is **synthetic practice data**. The limits in `config.py` are **placeholders** until you set them to your lab's Omani standard edition.

---

## 1. Run it on Windows

1. Install **Python 3.11 or newer** from python.org. During setup, tick **"Add Python to PATH"**.
2. Unzip this folder somewhere simple, for example `C:\Raqeeb`.
3. Double-click **`setup.bat`**. This is needed once; it takes a few minutes and needs internet.
4. Double-click **`run.bat`**. Your browser opens Raqeeb at `http://localhost:8501`.
5. To stop Raqeeb, close the black window.

To create a **new practice dataset**, double-click **`new_data.bat`** and type any number as the seed. It regenerates the data and retrains the models.

---

## 2. The five pages

| Page | What it shows | What to say in the demo |
|---|---|---|
| **Inbox** | Abnormal results, most urgent first (by default Oct–Dec, the months the model never saw). Click one to see the likely cause, the reasons, the action, and the note in English and Arabic. | "Every morning the chemist opens this instead of a spreadsheet." |
| **Map** | Sites coloured by their worst status in the chosen dates | "Problems cluster: we can see where." |
| **Site history** | Measured chlorine against the model's expected chlorine, plus temperature and water age | "This is how Raqeeb knows what normal looks like for each site." |
| **Tomorrow's sampling** | Sites ranked by the chance their chlorine falls below the minimum tomorrow | Show a July date, then a January date: rural, hot, long-water-age sites rise to the top. |
| **Model report** | Accuracy on unseen months and on a brand-new dataset; which evidence the model uses | "We tested it honestly, on data it never saw." |

---

## 3. How it works (read in this order)

| File | What it does |
|---|---|
| `config.py` | **All settings**: limits, windows, QC tolerance, the action text for each cause. Start here. |
| `columns_map.py` | Maps Raqeeb's column names to the names in your files. On Day 2, change only this file. |
| `generate_data.py` | Makes the practice data. Chlorine follows C = C0·exp(−k_T·t), with k_T = k20·θ^(T−20) and θ = 1.07 (an assumption; change it in `CONFIG`). It injects three kinds of problem, each with a known answer. |
| `raqeeb/data.py` | Loads the six tables and flags abnormal results against the limits in `config.py`. |
| `raqeeb/features.py` | Builds the evidence for each result: surprise against the expected chlorine, batch QC deviation, share of the batch that is abnormal, repairs in the DMA in the last 5 days, quality complaints in the last 2 days, bacteria with normal chlorine and turbidity, and others. |
| `raqeeb/models.py` | **Model 1** (LightGBM regression) learns each site's normal chlorine from temperature, water age, source, month and the site's history. **Model 2** (LightGBM classifier) learns the cause from the evidence. The reasons shown come from SHAP contributions, which LightGBM computes itself. It also holds the priority rules and the tomorrow ranking. |
| `raqeeb/notes.py` | Writes the English and Arabic investigation note from a template. Works offline, with no API key. |
| `train.py` | Trains both models and saves them to `models/`. Then it makes a new dataset (seed 7) and tests on it (robustness check). |
| `app.py` | The Streamlit dashboard. |

### Safety rules that always win over the AI
- **Any E. coli result is "Urgent"**, whatever the model says.
- Raqeeb **never closes or downgrades** a failed result. It only orders the work and explains it. The chemist decides under the lab SOP.
- Confidence is shown as **99% at most**, because it is a model output, not a fact.

### Current scores (synthetic data)
- Chlorine expectation error: about **±0.10 mg/L**.
- Cause accuracy on unseen months Oct–Dec: about **94%** (17 cases).
- Cause accuracy on a brand-new dataset: about **100%** (92 cases).

**Be honest about these numbers:** the new dataset comes from the same generator rules, so real-world accuracy will be lower. On real data, the classifier keeps learning from each investigation the chemist closes.

---

## 4. Questions judges will ask, and the answers

| Question | Answer |
|---|---|
| "Why AI and not just a limit check?" | The limit check says *fail*. Raqeeb says *why* and *what first*, by combining five systems that today are checked by hand. It also warns about passing sites that are drifting toward failure. |
| "What if it is wrong?" | It never closes a result, and E. coli is always urgent. A wrong call changes the order of the work, not whether it gets done. |
| "You don't have our data." | Change `columns_map.py` to your column names and press **Retrain models**. Nothing else changes. |
| "Real failures are rare. How does it learn?" | The expectation and QC parts need no failure labels. The classifier starts from synthetic and expert-labelled cases, then learns from each closed investigation. |
| "Why LightGBM?" | Strong on small tables, handles missing values, fast on a laptop, and gives exact per-result explanations (SHAP) without extra libraries. |
| "How did you avoid overfitting?" | Time split (train Jan–Sep, test Oct–Dec), never random. Plus a test on a completely new dataset with different sites (seed 7). |

---

## 5. Day 2 at Nama: using real data

1. Put their CSV exports in `data/`, using the same six file names (or edit `FILES` in `columns_map.py`).
2. Edit `columns_map.py` so the right-hand names match their columns.
3. If they have no `true_cause` column, the dashboard still works. Raqeeb uses the model trained on the practice data until real labelled cases exist.
4. Set the real limits in `config.py`.
5. Click **Retrain models**.

---

## 6. Known limits
- The map background needs internet (OpenStreetMap). The points work offline.
- The site positions are approximate and synthetic.
- The practice data has about 3% abnormal results, against Nama's real ~0.2%, so there are enough cases to learn from.
