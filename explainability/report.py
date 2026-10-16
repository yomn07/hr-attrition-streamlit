import re

from explainability.shap_utils import shap_frame

PROTECTED = {"Age", "Gender", "MaritalStatus"}

LABELS = {"OverTime": "Overtime", "NumCompaniesWorked": "Number of previous employers"}

SATISFACTION = {1: "Low", 2: "Medium", 3: "High", 4: "Very high"}
LEVELS = {
    "JobSatisfaction": SATISFACTION,
    "EnvironmentSatisfaction": SATISFACTION,
    "RelationshipSatisfaction": SATISFACTION,
    "JobInvolvement": SATISFACTION,
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
    "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
    "Education": {1: "Below college", 2: "College", 3: "Bachelor",
                  4: "Master", 5: "Doctor"},
}

MONEY = {"MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate"}

TOPICS = {
    "OverTime": "Workload and overtime: is it sustainable, can it be rebalanced?",
    "MonthlyIncome": "Compensation: how does pay compare with the role and market?",
    "JobSatisfaction": "Role fit: what would make the day-to-day work more satisfying?",
    "EnvironmentSatisfaction": "Work environment: what would improve it?",
    "WorkLifeBalance": "Work-life balance: flexibility, schedule, time off.",
    "YearsSinceLastPromotion": "Career growth: promotion path and development plan.",
    "YearsInCurrentRole": "Career growth: is a new challenge or role change possible?",
    "BusinessTravel": "Travel load: can it be reduced or shared?",
    "DistanceFromHome": "Commute: remote or hybrid options.",
    "StockOptionLevel": "Long-term incentives and benefits.",
    "TrainingTimesLastYear": "Learning and development opportunities.",
    "JobInvolvement": "Engagement: how involved and valued does the person feel?",
    "RelationshipSatisfaction": "Team relationships and management support.",
}


def _label(feature):
    if feature in LABELS:
        return LABELS[feature]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", feature).capitalize()


def _format(feature, value):
    if feature in LEVELS:
        return LEVELS[feature].get(int(round(float(value))), str(value))
    if feature in MONEY:
        return f"{float(value):,.0f}"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _strength(share):
    if share >= 0.15:
        return "major factor"
    if share >= 0.07:
        return "moderate factor"
    return "minor factor"


def _describe(feature, value, share, data):
    text = f"**{_label(feature)}**: {_format(feature, value)}"

    if feature in data.numeric_features:
        median = float(data.X[feature].median())
        v = float(value)
        if v > median:
            relation = "above"
        elif v < median:
            relation = "below"
        else:
            relation = "equal to"
        text += f" ({relation} the company median of {_format(feature, median)})"

    return f"{text}, {_strength(share)}"


def build_report(result, data, probability, threshold, position=0, top=3):
    """Return a plain-language Markdown report for one explained employee."""
    contrib = shap_frame(
        result, data.numeric_features, data.categorical_features, grouped=True
    ).iloc[position]
    original = result["X_original"].iloc[position]

    total = float(contrib.abs().sum()) or 1.0
    share = contrib.abs() / total

    visible = contrib[~contrib.index.isin(PROTECTED)]
    raising = visible[visible > 0].sort_values(ascending=False).head(top)
    lowering = visible[visible < 0].sort_values().head(top)

    base_rate = float(data.y.mean())
    level = "Higher risk" if probability >= threshold else "Lower risk"
    ratio = probability / base_rate if base_rate > 0 else float("nan")

    lines = [
        "### Retention risk summary",
        "",
        f"**Risk level: {level}.** The model estimates a **{probability:.0%}** "
        f"chance of leaving, about **{ratio:.1f} times** the company average "
        f"of {base_rate:.0%}. Alert threshold used: {threshold:.0%}.",
        "",
        "**Main factors associated with HIGHER risk**",
    ]
    if raising.empty:
        lines.append("- No factor stands out as raising the risk.")
    for feature in raising.index:
        lines.append(f"- {_describe(feature, original[feature], share[feature], data)}")

    lines += ["", "**Main factors associated with LOWER risk**"]
    if lowering.empty:
        lines.append("- No factor stands out as lowering the risk.")
    for feature in lowering.index:
        lines.append(f"- {_describe(feature, original[feature], share[feature], data)}")

    topics = [TOPICS[f] for f in raising.index if f in TOPICS]
    if topics:
        lines += ["", "**Possible topics for a supportive conversation**"]
        lines += [f"- {t}" for t in topics]

    lines += [
        "",
        "**How to read this**",
        "- These are statistical associations in historical data, **not causes**.",
        "- Use the report to start a voluntary, supportive conversation. "
        "Do not use it for hiring, firing, promotion or pay decisions.",
        "- Age, gender and marital status are not listed as reasons, even if "
        "the model uses them. Review the Fairness tab.",
    ]
    return "\n".join(lines)
