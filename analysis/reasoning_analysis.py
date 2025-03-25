import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import re
import os
from typing import List, Optional


REASONING_RULES = {
    "Positional": ["center column", "center square", "corner", "edge"],
    "Blocking": ["block", "prevent", "stop opponent", "avoid opponent", "counter"],
    "Opponent Modeling": ["opponent", "they are trying", "their strategy", "their move"],
    "Winning Logic": ["win", "winning move", "connect", "fork", "threat", "chance of winning"],
    "Heuristic": ["best move", "most likely", "advantageous", "better chance"],
    "Rule-Based": ["according to", "rule", "strategy"],
    "Random/Unjustified": ["random", "guess"]
}


class LLMReasoningAnalyzer:
    def __init__(self, csv_path: str):
        """Initialize the analyzer with a path to the LLM game log CSV.

        Args:
            csv_path: Path to the reasoning CSV file.
        """
        self.df = pd.read_csv(csv_path)
        self._preprocess()

    def _preprocess(self) -> None:
        """Prepare the DataFrame by filling NaNs and stripping whitespace."""
        self.df['reasoning'] = self.df['reasoning'].fillna("").astype(str)
        self.df['reasoning'] = self.df['reasoning'].str.strip()

    def categorize_reasoning(self) -> None:
        """Assign a reasoning category to each reasoning entry."""
        def classify(reasoning: str, agent: str) -> str:
            if not reasoning or agent.startswith("random"):
                return "Uncategorized"
            for label, keywords in REASONING_RULES.items():
                for kw in keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", reasoning.lower()):
                        return label
            return "Uncategorized"

        self.df['reasoning_type'] = self.df.apply(
            lambda row: classify(row['reasoning'], row['agent_name']), axis=1
        )

    def summarize_reasoning(self) -> None:
        """Generate a short summary for each reasoning entry (simple heuristic).

        OBS: Later we can replace this with LLM compression later.
        """
        def summarize(reasoning: str) -> str:
            if "." in reasoning:
                first = reasoning.split(".")[0]
            else:
                first = reasoning
            return " ".join(first.strip().split()[:10])

        self.df['summary'] = self.df['reasoning'].apply(summarize)

    def compute_metrics(self, output_csv: str = "agent_metrics_summary.csv", plot_dir: str = "plots") -> None:
            """Compute reasoning metrics per agent and save CSV + plots.

            Args:
                output_csv: Where to store the final summary CSV.
                plot_dir: Directory to store visualizations.
            """
            os.makedirs(plot_dir, exist_ok=True)
            rows = []
            for agent in self.df['agent_name'].unique():
                if agent.startswith("random"):
                    continue
                agent_df = self.df[self.df['agent_name'] == agent]
                total = len(agent_df)
                opponent_mentions = agent_df['reasoning'].str.lower().str.contains("opponent").sum()
                reasoning_len_avg = agent_df['reasoning'].apply(lambda r: len(r.split())).mean()
                unique_types = agent_df['reasoning_type'].nunique()
                type_counts = agent_df['reasoning_type'].value_counts(normalize=True).to_dict()
                entropy = -sum(p * np.log2(p) for p in type_counts.values() if p > 0)

                rows.append({
                    "agent_name": agent,
                    "total_moves": total,
                    "avg_reasoning_length": reasoning_len_avg,
                    "%_opponent_mentions": opponent_mentions / total,
                    "reasoning_diversity": unique_types,
                    "reasoning_entropy": entropy
                })

                # Plot reasoning type distribution pie chart
                plt.figure()
                agent_df['reasoning_type'].value_counts().plot.pie(autopct='%1.1f%%')
                plt.title(f"Reasoning Type Distribution - {agent}")
                plt.ylabel("")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"pie_reasoning_type_{agent}.png"))
                plt.close()

           # pd.DataFrame(rows).to_csv(output_csv, index=False) # Commented out to avoid writing to disk


    def plot_heatmaps_by_agent(self, output_dir: str = "plots") -> None:
        """Plot heatmaps of reasoning types per turn for each agent.

        Args:
            output_dir: Directory to save the heatmap plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        for agent in self.df['agent_name'].unique():
            if agent.startswith("random"):
                continue
            df_agent = self.df[self.df['agent_name'] == agent]
            pivot = df_agent.pivot_table(
                index="turn", columns="reasoning_type", values="agent_name",
                aggfunc="count", fill_value=0
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, cmap="YlGnBu", annot=True)
            plt.title(f"Reasoning Type by Turn - {agent}")
            plt.ylabel("Turn")
            plt.xlabel("Reasoning Type")
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"heatmap_{agent}.png")
            plt.savefig(out_path)
            plt.close()

    def plot_wordclouds_by_agent(self, output_dir: str = "plots") -> None:
        """Generate word clouds for each agent's reasoning.

        Args:
            output_dir: Directory to save the plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        for agent in self.df['agent_name'].unique():
            if agent.startswith("random"):
                continue
            agent_df = self.df[self.df['agent_name'] == agent]
            text = " ".join(agent_df['reasoning'].tolist())
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            title = f"Reasoning Word Cloud - {agent}"
            plt.title(title)
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"wordcloud_{agent}.png")
            plt.savefig(out_path)
            plt.close()

    def save_output(self, path: str) -> None:
        """Save the augmented DataFrame to a CSV.

        Args:
            path: Output file path.
        """
        self.df.to_csv(path, index=False)


if __name__ == "__main__":
    fpath= "/p/project/ccstdl/cipolina-kun1/open_spiel_arena/results/merged_logs_20250325_221921.csv"
    analyzer = LLMReasoningAnalyzer(fpath)
    analyzer.categorize_reasoning()
    analyzer.summarize_reasoning() # makes phrases shorter for ease of analysis (not super needed)
    analyzer.compute_metrics()
    analyzer.plot_heatmaps_by_agent()
    analyzer.plot_wordclouds_by_agent()
    analyzer.save_output("augmented_reasoning_output.csv")
    print('Done reasoning analysis.')
