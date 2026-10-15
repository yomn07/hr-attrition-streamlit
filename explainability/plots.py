import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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


def local_bar(result, position, numeric_features, top=10):
    features = result["features"]
    values = result["values"][position]
    transformed = result["X_transformed"].iloc[position]
    original = result["X_original"].iloc[position]

    labels = [
        f"{f} = {original[f]:g}" if f in numeric_features
        else f"{f} = {int(transformed[f])}"
        for f in features
    ]

    df = pd.DataFrame({"label": labels, "value": values})
    df = df.reindex(df["value"].abs().sort_values(ascending=False).index)
    shown, rest = df.head(top).iloc[::-1], df.iloc[top:]

    if len(rest):
        other = pd.DataFrame({
            "label": [f"{len(rest)} other features"],
            "value": [rest["value"].sum()],
        })
        shown = pd.concat([other, shown])

    fig = go.Figure(go.Bar(
        x=shown["value"], y=shown["label"], orientation="h",
        marker_color=[RED if v > 0 else BLUE for v in shown["value"]],
        text=[f"{v:+.3f}" for v in shown["value"]], textposition="outside",
    ))
    fig.update_layout(
        title="Top contributions for this employee",
        xaxis_title=f"Effect on the model output ({result['units']})",
        showlegend=False,
    )
    return fig


def pdp_line(pdp_df, feature):
    fig = px.line(
        pdp_df, x=feature, y="Attrition probability", markers=True,
        title=f"Partial dependence: {feature}",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig
