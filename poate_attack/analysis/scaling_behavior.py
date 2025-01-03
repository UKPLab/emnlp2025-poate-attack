import pandas as pd
# import matplotlib.pyplot as plt
import plotly.io as pio
pio.kaleido.scope.mathjax = None


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

sns.set_theme()
# sns.set_style("white")
sns.set_context("talk")
# plt.style.use("seaborn-deep")
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
sns.set_context("paper")
# from matplotlib import rc
# rc('font', family='serif')
plt.rcParams["text.color"] = "black"

import plotly.graph_objects as go
import plotly.io as pio


def plot_scaling_chart():

    # Input data
    datasets = ["AdvBench", "XSTest", "MaliciousInstructions"]

    # human
    # llama2_7b = [28, 26, 26]
    # llama2_70b = [40, 34, 34]
    # llama3_8b = [66, 74, 88]
    # llama3_70b = [76, 68, 80]
    # gemma2_9b = [42, 58, 72]
    # gemma2_27b = [52, 58, 64]
    # phi3_mini_4k = [58, 44, 68]
    # phi3_medium_4k = [58, 56, 62]
    # gpt3_5_turbo = [50, 58, 78]
    # gpt4_o = [60, 70, 86]

    # automatic
    llama2_7b = [46, 26, 40]
    llama2_70b = [44, 34, 66]
    llama3_8b = [72, 54, 76]
    llama3_70b = [88, 76, 80]
    gemma2_9b = [48, 48, 58]
    gemma2_27b = [70, 52, 68]
    phi3_mini_4k = [62, 50, 62]
    phi3_medium_4k = [70, 48, 58]
    gpt3_5_turbo = [58, 48, 74]
    gpt4_o = [74, 58, 86]

    zero_bar_scores = [0, 0, 0]

    # Define the models and their corresponding data
    model_scores = {
        "Llama2-7B-Chat": llama2_7b,
        "Llama2-70B-Chat": llama2_70b,
        "Spacer_llama2": zero_bar_scores,
        "Llama3.1-8B-instruct": llama3_8b,
        "Llama3.1-70B-instruct": llama3_70b,
        "Spacer_llama3": zero_bar_scores,
        "Gemma2-9B-it": gemma2_9b,
        "Gemma2-27B-it": gemma2_27b,
        "Spacer_gemma2": zero_bar_scores,
        "Phi3-mini-4k": phi3_mini_4k,
        "Phi3-medium-4k": phi3_medium_4k,
        "Spacer_phi3": zero_bar_scores,
        "GPT-3.5-turbo": gpt3_5_turbo,
        "GPT-4o": gpt4_o,
    }

    # Create a DataFrame for the data
    data = {
        "dataset": [],
        "model": [],
        "score": [],
        "family": []
    }

    # Populate the DataFrame with repeated datasets and model family names
    for model, scores in model_scores.items():
        family = model.split("-")[0]  # Extract the family name (e.g., "Llama2", "Llama3")
        data["dataset"].extend(datasets)  # Repeat datasets
        data["model"].extend([model] * len(datasets))  # Repeat the model for all datasets
        data["score"].extend(scores)  # Add scores for this model
        data["family"].extend([family] * len(datasets))  # Add family name for all datasets

    df = pd.DataFrame(data)

    # Initialize figure
    fig = go.Figure()

    # Define colors for each model
    model_colors = {
        "Llama2-7B-Chat": "#aec7e8", "Llama2-70B-Chat": "#1f77b4",
        "Llama3.1-8B-instruct": "#ffbb78", "Llama3.1-70B-instruct": "#ff7f0e",
        "Gemma2-9B-it": "#98df8a", "Gemma2-27B-it": "#2ca02c",
        "Phi3-mini-4k": "#ff9896", "Phi3-medium-4k": "#d62728",
        "GPT-3.5-turbo": "#c5b0d5", "GPT-4o": "#9467bd",
        "Spacer_llama2": "rgba(0, 0, 0, 0)",
        "Spacer_llama3": "rgba(0, 0, 0, 0)",
        "Spacer_gemma2": "rgba(0, 0, 0, 0)",
        "Spacer_phi3": "rgba(0, 0, 0, 0)"
    }

    family_colors = {
        "Llama2-7B-Chat": "blue", "Llama2-70B-Chat": "blue",
        "Llama3.1-8B-instruct": "orange", "Llama3.1-70B-instruct": "orange",
        "Gemma2-9B-it": "green", "Gemma2-27B-it": "green",
        "Phi3-mini-4k": "red", "Phi3-medium-4k": "red",
        "GPT-3.5-turbo": "purple", "GPT-4o": "purple",
        "Spacer_llama2": "rgba(0, 0, 0, 0)",
        "Spacer_llama3": "rgba(0, 0, 0, 0)",
        "Spacer_gemma2": "rgba(0, 0, 0, 0)",
        "Spacer_phi3": "rgba(0, 0, 0, 0)"
    }

    # keep unique but leave the spacer bar
    for model in df["model"].unique():
        filtered_df = df[df["model"] == model]

        if model.__contains__("Spacer"):
            fig.add_trace(
                go.Bar(
                    name=model,
                    showlegend=False,  # Hide from legend
                    x=filtered_df["dataset"],
                    y=filtered_df["score"],
                    marker=dict(
                        line=dict(color=family_colors[model], width=0),  # Invisible border
                        color=model_colors[model],
                    ),
                    width=0.01  # Make the spacer bar narrower
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=filtered_df["dataset"],
                    y=filtered_df["score"],
                    marker=dict(
                        line=dict(color=family_colors[model], width=1),  # Border color and width
                        color=model_colors[model]
                    ),
                    width=0.06,
                )
            )

    # Update layout for repeated datasets
    fig.update_layout(
        barcornerradius=1,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=[
            dict(
                type='line',
                x0=1.5, x1=1.5,
                y0=0, y1=1,
                xref='x', yref='paper',
                line=dict(color="gray", width=1, dash="dot")
            ),
            # Spacer after XSTest (group 2)
            dict(
                type='line',
                x0=0.5, x1=0.5,
                y0=0, y1=1,
                xref='x', yref='paper',
                line=dict(color="gray", width=1, dash="dot")
            ),
        ],
        # title="Dataset Comparison Across Model Families",
        font=dict(family='Times New Roman', size=16, color='black'),
        xaxis=dict(
            title="Datasets",
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
        ),
        yaxis=dict(
            title="Attack Success Rate (%)",
            ticks="outside", mirror=True, showline=True,
            range=[0, 100],
            linewidth=1.5,
            linecolor='black',
        ),
        barmode="group",  # Group bars by model
        bargap=0.15,  # Add some gap between bars
        legend=dict(title="Models", orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.4,
                    font=dict(family='Times New Roman', size=16, color='black'),
                    ),
        # legend=dict(
        #     title="Models",
        #     orientation="v",  # Vertical orientation
        #     yanchor="middle",  # Align vertically in the middle
        #     y=0.6,  # Centered vertically
        #     xanchor="left",  # Anchor horizontally to the left of the x-position
        #     x=1.2,  # Position legend further right (beyond the plot)
        #     font=dict(family='Times New Roman', size=8, color='black'),
        # ),
        template="plotly_white",
        width=620,
        height=360
    )

    # Save the plot as PDF
    pio.write_image(fig, "./data/plots/scaling_behavior_auto_small.pdf", scale=1)


def plot_human_comparison():

    # Input data
    datasets = ["AdvBench", "XSTest", "MaliciousInstructions"]

    # human
    # llama2_7b = [28, 26, 26]
    # llama2_70b = [40, 34, 34]
    # llama3_8b = [66, 74, 88]
    # llama3_70b = [76, 68, 80]
    # gemma2_9b = [42, 58, 72]
    # gemma2_27b = [52, 58, 64]
    # phi3_mini_4k = [58, 44, 68]
    # phi3_medium_4k = [58, 56, 62]
    # gpt3_5_turbo = [50, 58, 78]
    # gpt4_o = [60, 70, 86]

    # automatic
    llama2_7b_potee = [46, 26, 40]
    llama2_7b_human = [28, 26, 26]
    llama3_8b_potee = [72, 54, 76]
    llama3_8b_human = [66, 74, 88]
    gemma2_9b_potee = [48, 48, 58]
    gemma2_9b_human = [42, 58, 72]
    phi3_mini_4k_potee = [62, 50, 62]
    phi3_mini_4k_human = [58, 44, 68]
    gpt4_o_potee = [74, 58, 86]
    gpt4_o_human = [60, 70, 86]

    zero_bar_scores = [0, 0, 0]

    # Define the models and their corresponding data
    model_scores = {
        "Llama2-7B-Chat (POATE)": llama2_7b_potee,
        "Llama2-7B-Chat (Human)": llama2_7b_human,
        "Spacer_llama2": zero_bar_scores,
        "Llama3.1-8B-instruct (POATE)": llama3_8b_potee,
        "Llama3.1-8B-instruct (Human)": llama3_8b_human,
        "Spacer_llama3": zero_bar_scores,
        "Gemma2-9B-it (POATE)": gemma2_9b_potee,
        "Gemma2-9B-it (Human)": gemma2_9b_human,
        "Spacer_gemma2": zero_bar_scores,
        "Phi3-mini-4k (POATE)": phi3_mini_4k_potee,
        "Phi3-mini-4k (Human)": phi3_mini_4k_human,
        "Spacer_phi3": zero_bar_scores,
        "GPT-4o (POATE)": gpt4_o_potee,
        "GPT-4o (Human)": gpt4_o_human,
    }

    # Create a DataFrame for the data
    data = {
        "dataset": [],
        "model": [],
        "score": [],
        "family": []
    }

    # Populate the DataFrame with repeated datasets and model family names
    for model, scores in model_scores.items():
        family = model.split("-")[0]  # Extract the family name (e.g., "Llama2", "Llama3")
        data["dataset"].extend(datasets)  # Repeat datasets
        data["model"].extend([model] * len(datasets))  # Repeat the model for all datasets
        data["score"].extend(scores)  # Add scores for this model
        data["family"].extend([family] * len(datasets))  # Add family name for all datasets

    df = pd.DataFrame(data)

    # Initialize figure
    fig = go.Figure()

    # Define colors for each model
    model_colors = {
        "Llama2-7B-Chat (POATE)": "#aec7e8", "Llama2-7B-Chat (Human)": "#1f77b4",
        "Llama3.1-8B-instruct (POATE)": "#ffbb78", "Llama3.1-8B-instruct (Human)": "#ff7f0e",
        "Gemma2-9B-it (POATE)": "#98df8a", "Gemma2-9B-it (Human)": "#2ca02c",
        "Phi3-mini-4k (POATE)": "#ff9896", "Phi3-mini-4k (Human)": "#d62728",
        "GPT-4o (POATE)": "#c5b0d5", "GPT-4o (Human)": "#9467bd",
        "Spacer_llama2": "rgba(0, 0, 0, 0)",
        "Spacer_llama3": "rgba(0, 0, 0, 0)",
        "Spacer_gemma2": "rgba(0, 0, 0, 0)",
        "Spacer_phi3": "rgba(0, 0, 0, 0)"
    }

    family_colors = {
        "Llama2-7B-Chat (POATE)": "blue", "Llama2-7B-Chat (Human)": "blue",
        "Llama3.1-8B-instruct (POATE)": "orange", "Llama3.1-8B-instruct (Human)": "orange",
        "Gemma2-9B-it (POATE)": "green", "Gemma2-9B-it (Human)": "green",
        "Phi3-mini-4k (POATE)": "red", "Phi3-mini-4k (Human)": "red",
        "GPT-4o (POATE)": "purple", "GPT-4o (Human)": "purple",
        "Spacer_llama2": "rgba(0, 0, 0, 0)",
        "Spacer_llama3": "rgba(0, 0, 0, 0)",
        "Spacer_gemma2": "rgba(0, 0, 0, 0)",
        "Spacer_phi3": "rgba(0, 0, 0, 0)"
    }

    # keep unique but leave the spacer bar
    for model in df["model"].unique():
        filtered_df = df[df["model"] == model]

        if model.__contains__("Spacer"):
            fig.add_trace(
                go.Bar(
                    name=model,
                    showlegend=False,  # Hide from legend
                    x=filtered_df["dataset"],
                    y=filtered_df["score"],
                    marker=dict(
                        line=dict(color=family_colors[model], width=0),  # Invisible border
                        color=model_colors[model],
                    ),
                    width=0.01  # Make the spacer bar narrower
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    name=model,
                    x=filtered_df["dataset"],
                    y=filtered_df["score"],
                    marker=dict(
                        line=dict(color=family_colors[model], width=1),  # Border color and width
                        color=model_colors[model]
                    ),
                    width=0.06,
                )
            )

    # Update layout for repeated datasets
    fig.update_layout(
        barcornerradius=1,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=[
            dict(
                type='line',
                x0=1.5, x1=1.5,
                y0=0, y1=1,
                xref='x', yref='paper',
                line=dict(color="gray", width=1, dash="dot")
            ),
            # Spacer after XSTest (group 2)
            dict(
                type='line',
                x0=0.5, x1=0.5,
                y0=0, y1=1,
                xref='x', yref='paper',
                line=dict(color="gray", width=1, dash="dot")
            ),
        ],
        # title="Dataset Comparison Across Model Families",
        font=dict(family='Times New Roman', size=16, color='black'),
        xaxis=dict(
            title="Datasets",
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
        ),
        yaxis=dict(
            title="Attack Success Rate (%)",
            ticks="outside", mirror=True, showline=True,
            range=[0, 100],
            linewidth=1.5,
            linecolor='black',
        ),
        barmode="group",  # Group bars by model
        bargap=0.15,  # Add some gap between bars
        legend=dict(title="Models", orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.4,
                    font=dict(family='Times New Roman', size=16, color='black'),
                    ),
        # legend=dict(
        #     title="Models",
        #     orientation="v",  # Vertical orientation
        #     yanchor="middle",  # Align vertically in the middle
        #     y=0.6,  # Centered vertically
        #     xanchor="left",  # Anchor horizontally to the left of the x-position
        #     x=1.2,  # Position legend further right (beyond the plot)
        #     font=dict(family='Times New Roman', size=8, color='black'),
        # ),
        template="plotly_white",
        width=620,
        height=360
    )

    # Save the plot as PDF
    pio.write_image(fig, "./data/plots/potee_human.pdf", scale=1)

if __name__ == '__main__':
    # plot_scaling_chart()
    plot_human_comparison()