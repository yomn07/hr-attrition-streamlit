"""HR Attrition Analytics - compare all models, pick the best, tune hyperparameters.

Self-contained: no hr_features module and no saved .joblib needed.
Upload the IBM HR Attrition CSV (WA_Fn-UseC_-HR-Employee-Attrition.csv) and the
app trains the five models from the notebook with the same protocol.
"""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="HR Attrition - Model Comparison", page_icon="📊", layout="wide")

SEED = 42
TARGET = "Attrition"
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
MODEL_ORDER = ["Logistic Regression", "LR + interactions", "Decision Tree",
               "Random Forest", "XGBoost"]            # simple -> complex (parsimony order)
COLORS = {"Logistic Regression": "#4C72B0", "LR + interactions": "#64B5CD",
          "Decision Tree": "#8172B3", "Random Forest": "#55A868", "XGBoost": "#DD8452"}

# Defaults = best hyperparameters found in the notebook
DEFAULTS = {
    "Logistic Regression": dict(C=0.1, l1_ratio=0.0),
    "LR + interactions": dict(C=0.1, l1_ratio=0.0),
    "Decision Tree": dict(criterion="entropy", max_depth=5, min_samples_leaf=10, ccp_alpha=0.01),
    "Random Forest": dict(n_estimators=800, max_depth=0, min_samples_leaf=2, min_samples_split=10,
                          max_features=0.2, criterion="entropy", class_weight="None"),
    "XGBoost": dict(n_estimators=537, max_depth=1, learning_rate=0.1845, subsample=0.75,
                    colsample_bytree=0.486, min_child_weight=3, gamma=0.73,
                    reg_lambda=5.47, reg_alpha=2.25, scale_pos_weight=1.0),
}


# ------------------------------------------------------------------ data / pipeline
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns]).copy()
    if TARGET not in df.columns:
        raise ValueError("The CSV must contain an 'Attrition' column.")
    y = df[TARGET]
    df[TARGET] = y.astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
    if df.isna().any().any():
        raise ValueError("The data contains missing values; please fix them first.")
    return df


LOG_COLS = ["MonthlyIncome", "YearsAtCompany", "YearsSinceLastPromotion",
            "TotalWorkingYears", "NumCompaniesWorked"]


def add_interactions(d: pd.DataFrame) -> pd.DataFrame:
    """Interaction terms + log transforms (used only by 'LR + interactions')."""
    d = d.copy()
    ot = (d["OverTime"] == "Yes").astype(int)
    d["OT_x_Single"] = ot * (d["MaritalStatus"] == "Single").astype(int)
    d["OT_x_TravelFreq"] = ot * (d["BusinessTravel"] == "Travel_Frequently").astype(int)
    d["OT_x_Sales"] = ot * (d["Department"] == "Sales").astype(int)
    for c in LOG_COLS:
        if c in d.columns:
            d["log_" + c] = np.log1p(d[c])
    return d


def _cat(d): return [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]
def _num(d): return [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]


def build_pipeline(clf, scale=False, interactions=False):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), _cat),
        ("num", StandardScaler() if scale else "passthrough", _num),
    ])
    steps = []
    if interactions:
        steps.append(("fe", FunctionTransformer(add_interactions)))
    return Pipeline(steps + [("prep", prep), ("clf", clf)])


def make_model(name, p, pos_weight):
    if name in ("Logistic Regression", "LR + interactions"):
        clf = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                                 C=p["C"], l1_ratio=p["l1_ratio"], max_iter=3000, tol=1e-3,
                                 random_state=SEED)
        return build_pipeline(clf, scale=True, interactions=(name == "LR + interactions"))
    if name == "Decision Tree":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **p)
        return build_pipeline(clf)
    if name == "Random Forest":
        q = dict(p)
        q["max_depth"] = None if q["max_depth"] == 0 else q["max_depth"]
        q["class_weight"] = None if q["class_weight"] == "None" else q["class_weight"]
        return build_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1, **q))
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                        random_state=SEED, n_jobs=1, verbosity=0, **p)
    return build_pipeline(clf)


# ------------------------------------------------------------------ training
def train_all(df, params, n_repeats, test_size, tie_tol, progress):
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED + 1)

    models, folds, rows, curves = {}, {}, [], {}
    for i, name in enumerate(MODEL_ORDER):
        progress.progress(i / len(MODEL_ORDER), text=f"Training {name} ...")
        p = dict(params[name])
        if name == "XGBoost":
            p["scale_pos_weight"] = {"none": 1.0, "sqrt": float(np.sqrt(pos_weight)),
                                     "full": pos_weight}.get(str(p["scale_pos_weight"]), p["scale_pos_weight"])
        model = make_model(name, p, pos_weight)
        cv = cross_validate(model, X_tr, y_tr, cv=rcv, n_jobs=-1, return_train_score=True,
                            scoring={"roc": "roc_auc", "pr": "average_precision"})
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        models[name] = model
        folds[name] = cv["test_roc"]
        curves[name] = proba
        rows.append({
            "Model": name,
            "CV ROC-AUC": cv["test_roc"].mean(), "CV std": cv["test_roc"].std(),
            "CV PR-AUC": cv["test_pr"].mean(),
            "Train ROC-AUC (CV)": cv["train_roc"].mean(),
            "Test ROC-AUC": roc_auc_score(y_te, proba),
            "Test PR-AUC": average_precision_score(y_te, proba),
        })
    progress.progress(1.0, text="Done")

    summary = pd.DataFrame(rows).set_index("Model")
    summary["Overfit gap"] = summary["Train ROC-AUC (CV)"] - summary["CV ROC-AUC"]
    return dict(summary=summary, models=models, folds=folds, proba_test=curves,
                X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, params=params,
                columns=list(X.columns), n_repeats=n_repeats)


def pick_best(summary, tie_tol):
    top = summary["CV ROC-AUC"].max()
    eligible = [n for n in MODEL_ORDER if summary.loc[n, "CV ROC-AUC"] >= top - tie_tol]
    return summary["CV ROC-AUC"].idxmax(), eligible[0], eligible


# ------------------------------------------------------------------ UI helpers
def hyperparameter_ui():
    """Sidebar widgets; returns {model: params}."""
    d = DEFAULTS
    out = {}
    with st.sidebar.expander("Logistic Regression (+ interactions)", expanded=False):
        c = st.select_slider("C (inverse regularisation)", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                             value=d["Logistic Regression"]["C"], key="lr_c")
        l1 = st.select_slider("l1_ratio (0 = ridge, 1 = lasso)", [0.0, 0.25, 0.5, 0.75, 1.0],
                              value=0.0, key="lr_l1")
        out["Logistic Regression"] = out["LR + interactions"] = dict(C=c, l1_ratio=l1)
    with st.sidebar.expander("Decision Tree"):
        dd = d["Decision Tree"]
        out["Decision Tree"] = dict(
            criterion=st.selectbox("criterion", ["gini", "entropy"], index=1, key="dt_cr"),
            max_depth=st.slider("max_depth", 2, 12, dd["max_depth"], key="dt_md"),
            min_samples_leaf=st.slider("min_samples_leaf", 1, 60, dd["min_samples_leaf"], key="dt_ml"),
            ccp_alpha=st.select_slider("ccp_alpha", [0.0, 0.002, 0.005, 0.01, 0.02], value=0.01, key="dt_cc"))
    with st.sidebar.expander("Random Forest"):
        dd = d["Random Forest"]
        out["Random Forest"] = dict(
            n_estimators=st.slider("n_estimators", 100, 1000, dd["n_estimators"], 50, key="rf_n"),
            max_depth=st.slider("max_depth (0 = unlimited)", 0, 20, 0, key="rf_md"),
            min_samples_leaf=st.slider("min_samples_leaf", 1, 20, dd["min_samples_leaf"], key="rf_ml"),
            min_samples_split=st.slider("min_samples_split", 2, 30, dd["min_samples_split"], key="rf_ms"),
            max_features=st.select_slider("max_features", ["sqrt", "log2", 0.2, 0.35, 0.5], value=0.2, key="rf_mf"),
            criterion=st.selectbox("criterion", ["gini", "entropy"], index=1, key="rf_cr"),
            class_weight=st.selectbox("class_weight", ["None", "balanced", "balanced_subsample"], key="rf_cw"))
    with st.sidebar.expander("XGBoost"):
        dd = d["XGBoost"]
        out["XGBoost"] = dict(
            n_estimators=st.slider("n_estimators", 50, 800, dd["n_estimators"], 10, key="xg_n"),
            max_depth=st.slider("max_depth (1 = additive)", 1, 6, dd["max_depth"], key="xg_md"),
            learning_rate=st.slider("learning_rate", 0.01, 0.30, dd["learning_rate"], 0.005, key="xg_lr"),
            subsample=st.slider("subsample", 0.5, 1.0, dd["subsample"], 0.05, key="xg_ss"),
            colsample_bytree=st.slider("colsample_bytree", 0.3, 1.0, dd["colsample_bytree"], 0.05, key="xg_cs"),
            min_child_weight=st.slider("min_child_weight", 1, 15, dd["min_child_weight"], key="xg_mc"),
            gamma=st.slider("gamma", 0.0, 3.0, dd["gamma"], 0.05, key="xg_g"),
            reg_lambda=st.slider("reg_lambda", 1.0, 50.0, dd["reg_lambda"], 0.5, key="xg_rl"),
            reg_alpha=st.slider("reg_alpha", 0.0, 5.0, dd["reg_alpha"], 0.05, key="xg_ra"),
            scale_pos_weight=st.select_slider("scale_pos_weight", ["none", "sqrt", "full"], key="xg_sp"))
    return out


def bar_chart(summary, best):
    fig = go.Figure()
    for col, color in [("CV ROC-AUC", "#4C72B0"), ("Test ROC-AUC", "#C44E52")]:
        fig.add_bar(x=summary.index, y=summary[col], name=col, marker_color=color,
                    text=summary[col].round(3), textposition="outside")
    fig.update_layout(barmode="group", yaxis=dict(range=[0.5, 0.95], title="ROC-AUC"),
                      height=380, margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.12))
    return fig


def threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0)), (tn, fp, fn, tp)


# ------------------------------------------------------------------ app
st.title("📊 HR Attrition: model comparison")
st.caption("IBM HR Analytics · 5 models · repeated stratified CV · parsimony-based selection")

st.sidebar.header("1 · Data")
upload = st.sidebar.file_uploader("IBM HR Attrition CSV", type="csv")
st.sidebar.header("2 · Protocol")
n_repeats = st.sidebar.slider("CV repeats (5-fold each)", 1, 5, 3,
                              help="More repeats = more reliable ranking but slower. Notebook used 5.")
test_size = st.sidebar.slider("Test set share", 0.1, 0.4, 0.2, 0.05)
tie_tol = st.sidebar.slider("Tie tolerance (ROC-AUC)", 0.0, 0.05, 0.01, 0.005,
                            help="Models within this distance of the top CV score are 'tied'; "
                                 "the simplest tied model is selected.")
st.sidebar.header("3 · Hyperparameters")
st.sidebar.caption("Defaults are the best values found in your notebook. Change any of them, then retrain.")
params = hyperparameter_ui()
if st.sidebar.button("Reset to notebook values"):
    for k in [k for k in st.session_state if k.split("_")[0] in {"lr", "dt", "rf", "xg"}]:
        del st.session_state[k]
    st.rerun()

if upload is None:
    st.info("⬅️ Upload `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the sidebar to start.")
    st.stop()

try:
    df = clean(pd.read_csv(upload))
except Exception as exc:
    st.error(f"Could not read the data: {exc}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Employees", f"{len(df):,}")
c2.metric("Predictors", df.shape[1] - 1)
c3.metric("Attrition rate", f"{df[TARGET].mean():.1%}")

if st.button("🚀 Train & compare all models", type="primary"):
    bar = st.progress(0.0, text="Starting ...")
    try:
        st.session_state.run = train_all(df, params, n_repeats, test_size, tie_tol, bar)
    except Exception as exc:
        st.error(f"Training failed: {exc}")
    bar.empty()

run = st.session_state.get("run")
if run is None:
    st.stop()

summary = run["summary"]
strongest, best, eligible = pick_best(summary, tie_tol)

st.success(
    f"🏆 **Best model: {best}**  ·  CV ROC-AUC {summary.loc[best, 'CV ROC-AUC']:.3f} "
    f"± {summary.loc[best, 'CV std']:.3f}  ·  test ROC-AUC {summary.loc[best, 'Test ROC-AUC']:.3f}"
)
if strongest != best:
    st.caption(f"Highest raw CV score: {strongest}, but {best} is within {tie_tol} of it and simpler "
               f"(parsimony rule). Tied models: {', '.join(eligible)}.")
else:
    others = summary["CV ROC-AUC"].drop(best)
    gap = summary.loc[best, "CV ROC-AUC"] - others.max()
    st.caption(f"Margin over the runner-up ({others.idxmax()}): {gap:+.3f} CV ROC-AUC. "
               f"Fold-to-fold std is about {summary.loc[best, 'CV std']:.3f}, so gaps smaller than that are noise.")

tab_cmp, tab_curves, tab_best, tab_pred = st.tabs(
    ["🏁 Comparison", "📈 ROC / PR curves", "🔍 Best model detail", "🔮 Predict"])

with tab_cmp:
    show = summary.copy()
    show.insert(0, "Rank (CV)", show["CV ROC-AUC"].rank(ascending=False).astype(int))
    show.insert(1, "Selected", ["🏆" if n == best else "" for n in show.index])
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c not in ("Rank (CV)", "Selected")})
                 .highlight_max(subset=["CV ROC-AUC", "Test ROC-AUC", "Test PR-AUC"], color="#d7f0dc"),
                 width="stretch")
    st.plotly_chart(bar_chart(summary, best), width="stretch")
    fig = go.Figure()
    for n in MODEL_ORDER:
        fig.add_box(y=run["folds"][n], name=n, marker_color=COLORS[n], boxmean=True)
    fig.update_layout(title="Per-fold ROC-AUC (same folds for every model)", showlegend=False,
                      height=380, yaxis_title="ROC-AUC", margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")
    st.caption("Overfit gap = train AUC − validation AUC in CV. A large gap (typical for the Random Forest) means "
               "the model memorises training data.")

with tab_curves:
    y_te = run["y_te"]
    left, right = st.columns(2)
    f1 = go.Figure(); f2 = go.Figure()
    for n in MODEL_ORDER:
        p = run["proba_test"][n]
        fpr, tpr, _ = roc_curve(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        w = 4 if n == best else 2
        f1.add_scatter(x=fpr, y=tpr, name=f"{n} ({roc_auc_score(y_te, p):.3f})", line=dict(color=COLORS[n], width=w))
        f2.add_scatter(x=rc, y=pr, name=f"{n} ({average_precision_score(y_te, p):.3f})", line=dict(color=COLORS[n], width=w))
    f1.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="gray", dash="dash"), showlegend=False)
    f2.add_hline(y=y_te.mean(), line_dash="dash", line_color="gray")
    f1.update_layout(title="ROC curve (test)", xaxis_title="False positive rate", yaxis_title="True positive rate", height=430)
    f2.update_layout(title="Precision-Recall (test)", xaxis_title="Recall", yaxis_title="Precision", height=430)
    left.plotly_chart(f1, width="stretch")
    right.plotly_chart(f2, width="stretch")
    st.caption(f"The test set has only {int(y_te.sum())} leavers, so curves are noisy: "
               "rely on the repeated-CV table for the ranking.")

with tab_best:
    st.subheader(f"{best}")
    st.json({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in run["params"][best].items()})
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.30, 0.05,
                    help="Attrition is rare, so a threshold below 0.5 usually flags more at-risk employees.")
    p = run["proba_test"][best]
    m, (tn, fp, fn, tp) = threshold_metrics(y_te, p, thr)
    cols = st.columns(5)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")
    cols[3].metric("Flagged", f"{(p >= thr).mean():.0%}")
    cols[4].metric("Leavers caught", f"{tp}/{tp + fn}")
    cm = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Pred: Stayed", "Pred: Left"],
                              y=["Actual: Stayed", "Actual: Left"], colorscale="Blues",
                              text=[[tn, fp], [fn, tp]], texttemplate="%{text}", showscale=False))
    cm.update_layout(height=340, yaxis_autorange="reversed", margin=dict(t=20))
    st.plotly_chart(cm, width="stretch")

    est = run["models"][best].named_steps["clf"]
    prep = run["models"][best].named_steps["prep"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    imp = None
    if hasattr(est, "coef_"):
        imp = pd.Series(np.abs(est.coef_[0]), index=names)
    elif hasattr(est, "feature_importances_"):
        imp = pd.Series(est.feature_importances_, index=names)
    if imp is not None:
        top = imp.sort_values().tail(12)
        fi = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=COLORS[best]))
        fi.update_layout(title="Top 12 features (|coefficient| or importance)", height=420, margin=dict(t=50))
        st.plotly_chart(fi, width="stretch")

with tab_pred:
    st.write(f"Score new employees with **{best}** (trained on the training split above). "
             "Upload a CSV with the same predictor columns; `Attrition` is optional.")
    new = st.file_uploader("Employee CSV", type="csv", key="new_csv")
    thr_p = st.slider("Flag threshold", 0.05, 0.95, 0.30, 0.05, key="thr_pred")
    if new is not None:
        try:
            nd = pd.read_csv(new)
            missing = [c for c in run["columns"] if c not in nd.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            probs = run["models"][best].predict_proba(nd[run["columns"]])[:, 1]
            out = nd.copy()
            out["Attrition_Probability"] = probs
            out["Flagged"] = np.where(probs >= thr_p, "Yes", "No")
            st.metric("Flagged employees", f"{(probs >= thr_p).sum()} / {len(out)}")
            st.dataframe(out.sort_values("Attrition_Probability", ascending=False), width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not score the file: {exc}")

st.divider()
st.caption("Estimates support analysis; they must not be used for automated employment decisions.")
