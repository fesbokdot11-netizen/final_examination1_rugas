# Deployment Report - Section B

**Student:** Kristheal Sofia Rugas  **GitHub repo:** https://github.com/fesbokdot11-netizen/final_examination1_rugas
**Streamlit URL:** https://finalexamination1rugas-lrmsjqxywqmmwqcutdpxx.streamlit.app

## 1. Reused midterm code
- Data loading / cleaning: `modules/data_processing.py`
- Feature engineering: `Queue_Category`, `Submission_Weekday`, and `SLA_Breached` logic in `modules/data_processing.py`
- Model: Decision Tree pipeline with OneHotEncoder and passthrough numerical features from `scripts/train_model.py`
- Inference and scoring: `modules/prediction.py`

## 2. Changes from the midterm
- The fitted pipeline is saved as `models/trained_model.joblib` and loaded with `@st.cache_resource` so it is not retrained on every refresh.
- The app wraps the model in a human-in-the-loop decision-support workflow instead of exposing raw model output alone.
- The dashboard and system-design pages translate the technical model into an information-system workflow for staff review.
- The saved model was regenerated with scikit-learn 1.9.1 for compatibility with the current project environment.
- The bundled dataset was replaced with the 400-row dataset supplied for this project. The model was retrained on
  completed requests only; pending rows are normalized to the app's `Open` workflow status.

## 3. Tests performed
| Test | Expected | Result |
|---|---|---|
| Supplied tab-separated sample loads and preprocessing completes | 400 rows loaded and derived features present | Passed |
| Model loads and accepts a valid one-row input | Prediction returns a class and probabilities | Passed |
| Streamlit Home and Dashboard render | No app exceptions in Streamlit smoke tests | Passed |
| `Processing` is accepted as a workflow status | Validation returns no errors | Passed |
| Invalid upload, missing columns, and empty CSV | Clear validation errors | Not yet verified end-to-end |
| Same cases compared with the submitted midterm notebook | Matching predictions | Not yet verified; original notebook cases were unavailable |
| Administrator cannot upload/change active data or record decisions | Read-only role behavior | Verified in role-gated UI; selector is a demo, not authentication |
| Public Streamlit deployment | URL opens successfully | Deployed; confirm after the latest rebuild |

## 4. Limitations
- Predictions are advisory only and do not authorize denial of service.
- The model is trained on completed requests only, because open/cancelled requests do not have an outcome label.
- The Decision Tree is shallow and the probability estimates are coarse.
- The supplied sample may not represent live operations. Validate on authorized operational data before relying on model output.

## 5. Reflection
The application structure converts the SLA prediction workflow into a decision-support experience with validation,
analysis, and human review. Before operational use, compare the bundled model against the original midterm notebook
and validate the supplied sample/model with properly authorized operational data.
