"""Feature engineering + preprocessing for the HR attrition project.
Kept in a module so that a pickled sklearn pipeline can be re-loaded in any Python session."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

LOW_INCOME = 3000      #first quartile of MonthlyIncome in the IBM data
YOUNG_AGE = 30


def add_features(d, interactions=False):
    d = d.copy()
    d["IncomePerJobLevel"]  = d["MonthlyIncome"] / d["JobLevel"]
    d["TenureRatio"]        = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1)
    d["AvgYearsPerCompany"] = d["TotalWorkingYears"] / (d["NumCompaniesWorked"] + 1)
    d["PromotionGap"]       = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["ManagerStability"]   = d["YearsWithCurrManager"] / (d["YearsAtCompany"] + 1)
    d["RoleStability"]      = d["YearsInCurrentRole"] / (d["YearsAtCompany"] + 1)
    d["SatisfactionMean"]   = d[["EnvironmentSatisfaction", "JobSatisfaction",
                                 "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)
    d["IncomePerYearOfExp"] = d["MonthlyIncome"] / (d["TotalWorkingYears"] + 1)

    if interactions:
        for c in ("MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"):
            if c in d:
                d["Log" + c] = np.log1p(d[c])
        ot = (d["OverTime"] == "Yes").astype(int) if "OverTime" in d else None
        single = (d["MaritalStatus"] == "Single").astype(int) if "MaritalStatus" in d else None
        travel = (d["BusinessTravel"] == "Travel_Frequently").astype(int) if "BusinessTravel" in d else None
        if ot is not None and single is not None:
            d["OT_x_Single"] = ot * single
        if ot is not None and travel is not None:
            d["OT_x_TravelFreq"] = ot * travel
        if single is not None and travel is not None:
            d["Single_x_TravelFreq"] = single * travel
        if ot is not None:
            d["OT_x_LowIncome"] = ot * (d["MonthlyIncome"] < LOW_INCOME).astype(int)
            d["OT_x_Dissatisfaction"] = ot * (4 - d["SatisfactionMean"])
            if "Age" in d:
                d["OT_x_Young"] = ot * (d["Age"] < YOUNG_AGE).astype(int)
    return d


def select_cat(d):
    return [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]


def select_num(d):
    return [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]


def build_pipeline(clf, scale=False, interactions=False):
    """raw employees -> engineered features -> one-hot (+ optional scaling) -> classifier"""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), select_cat),
        ("num", StandardScaler() if scale else "passthrough", select_num),
    ])
    return Pipeline([
        ("fe", FunctionTransformer(add_features, kw_args={"interactions": interactions})),
        ("prep", prep),
        ("clf", clf),
    ])
