import pandas as pd
import matplotlib.pyplot as plt

save_dir = "data/output"
csv_path = "data/selected_eval_results.csv"
df = pd.read_csv(csv_path)
df = df.copy()
for col in df.columns:
    if df[col].dtype != 'object':
        df[col] = df[col].apply(lambda x: f"{x:.8g}")

fig, ax = plt.subplots(figsize = (len(df.columns) * 2.2, len(df) * 0.5))
ax.axis('off')

table = ax.table(
    cellText = df.values,
    colLabels = df.columns,
    loc = 'center',
    cellLoc = 'center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)
plt.title("Evaluation Results at Different Steps", fontsize = 14, pad = 20)
plt.tight_layout()
plt.savefig(f"{save_dir}/eval_table.png")