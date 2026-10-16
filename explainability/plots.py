import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from explainability.shap_utils import shap_frame
RED, BLUE = "#d62728", "#1f77b4"


def global_bar(importance, units, top=15):
    shown = importance.head(top).iloc[::-1]
    fig = px.bar(
        x=shown.values, y=shown.index, orientation="h",
        labels={"x": f"Mean |SHAP value| ({units})", "y": ""},
        title="Global feature importance (SHAP)",
    )
    return fig


def beeswarm(shap_df, X_transformed, top=15, seed=42):
    order = shap_df.abs().mean().sort_values(ascending=False).head(top).index[::-1]
    rng = np.random.default_rng(seed)
    fig = go.Figure()

    for position, feature in enumerate(order):
        x = shap_df[feature].to_numpy()
        raw = X_transformed[feature].to_numpy(dtype=float)
        spread = raw.max() - raw.min()
        scaled = (raw - raw.min()) / spread if spread > 0 else np.zeros_like(raw)

        fig.add_trace(go.Scatter(
            x=x,
            y=position + rng.uniform(-0.3, 0.3, size=len(x)),
            mode="markers",
            marker=dict(
                size=5, color=scaled, colorscale="Bluered", cmin=0, cmax=1,
                showscale=(position == 0),
                colorbar=dict(title="Feature value", tickvals=[0, 1],
                              ticktext=["Low", "High"]),
            ),
            hovertemplate=f"{feature}<br>SHAP: %{{x:.3f}}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title="Direction of effect (right = pushes toward leaving)",
        xaxis_title="SHAP value",
        yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                   ticktext=list(order)),
        height=max(400, 28 * len(order) + 150),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    return fig


def _local_contributions(result, position, numeric_features,
                         categorical_features=None, grouped=False):
    """Labels and SHAP values for one employee, grouped or one-hot."""
    original = result["X_original"].iloc[position]

    if grouped and categorical_features is not None:
        # OverTime_Yes + OverTime_No -> OverTime
        contrib = shap_frame(
            result, numeric_features, categorical_features, grouped=True
        ).iloc[position]
        labels = [
            f"{f} = {original[f]:g}" if f in numeric_features
            else f"{f} = {original[f]}"
            for f in contrib.index
        ]
        return labels, contrib.to_numpy()

    transformed = result["X_transformed"].iloc[position]
    labels = [
        f"{f} = {original[f]:g}" if f in numeric_features
        else f"{f} = {int(transformed[f])}"
        for f in result["features"]
    ]
    return labels, np.asarray(result["values"][position])


def _top_with_rest(labels, values, top):
    """Largest contributions first, with the remainder merged into one row."""
    order = np.argsort(-np.abs(values))
    rows = [(labels[i], float(values[i])) for i in order[:top]]
    rest = values[order[top:]]
    if len(rest):
        rows.append((f"{len(rest)} other features", float(rest.sum())))
    return rows


def local_bar(result, position, numeric_features,
              categorical_features=None, grouped=False, top=10):
    """Why one employee got their score. Red raises risk, blue lowers it."""
    labels, values = _local_contributions(
        result, position, numeric_features, categorical_features, grouped
    )
    rows = _top_with_rest(labels, values, top)[::-1]  # largest on top

    fig = go.Figure(go.Bar(
        x=[v for _, v in rows], y=[l for l, _ in rows], orientation="h",
        marker_color=[RED if v > 0 else BLUE for _, v in rows],
        text=[f"{v:+.3f}" for _, v in rows], textposition="outside",
    ))
    fig.update_layout(
        title="Top contributions for this employee",
        xaxis_title=f"Effect on the model output ({result['units']})",
        showlegend=False,
    )
    return fig


def waterfall(result, position, numeric_features,
              categorical_features=None, grouped=True, top=8):
    """Start at the average output, add each feature, end at this employee."""
    labels, values = _local_contributions(
        result, position, numeric_features, categorical_features, grouped
    )
    rows = _top_with_rest(labels, values, top)

    base = float(result["base"])
    final = base + float(np.sum(values))

    names = ["Average (base value)"] + [l for l, _ in rows] + ["Final model output"]
    steps = [base] + [v for _, v in rows] + [final]
    measures = ["absolute"] + ["relative"] * len(rows) + ["total"]
    text = [f"{base:.3f}"] + [f"{v:+.3f}" for _, v in rows] + [f"{final:.3f}"]

    fig = go.Figure(go.Waterfall(
        orientation="h", y=names, x=steps, measure=measures,
        text=text, textposition="outside",
        connector=dict(line=dict(color="gray", width=1)),
        increasing=dict(marker=dict(color=RED)),
        decreasing=dict(marker=dict(color=BLUE)),
        totals=dict(marker=dict(color="#7f7f7f")),
    ))
    fig.update_yaxes(autorange="reversed")  # read top to bottom
    fig.update_layout(
        title="How this employee's score is built, step by step",
        xaxis_title=f"Model output ({result['units']})",
        showlegend=False,
        height=max(380, 42 * len(names) + 140),
    )
    return fig

def pdp_line(pdp_df, feature):
    fig = px.line(
        pdp_df, x=feature, y="Attrition probability", markers=True,
        title=f"Partial dependence: {feature}",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig
