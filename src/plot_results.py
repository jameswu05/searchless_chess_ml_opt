import pandas as pd
import matplotlib.pyplot as plt

save_dir = "data/output"
csv_path = "data/selected_eval_results.csv"
df = pd.read_csv(csv_path)

print(df)

df = df.sort_values("step")

#Accuracy vs step
plt.figure(figsize=(7, 4))
plt.plot(df["step"], df["eval_action_accuracy"], marker="o")
plt.xlabel("Step")
plt.ylabel("Eval Action Accuracy")
plt.title("Eval Action Accuracy vs Step")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/accuracy.png")

#Log loss vs step
plt.figure(figsize=(7, 4))
plt.plot(df["step"], df["eval_output_log_loss"], marker="o")
plt.xlabel("Step")
plt.ylabel("Eval Output Log Loss")
plt.title("Eval Output Log Loss vs Step")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/log_loss.png")

#Kendall tau vs step
plt.figure(figsize=(7, 4))
plt.plot(df["step"], df["eval_kendall_tau"], marker="o")
plt.xlabel("Step")
plt.ylabel("Eval Kendall Tau")
plt.title("Eval Kendall Tau vs Step")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/kendall_tau.png")

#Entropy vs step
plt.figure(figsize=(7, 4))
plt.plot(df["step"], df["eval_entropy"], marker="o")
plt.xlabel("Step")
plt.ylabel("Eval Entropy")
plt.title("Eval Entropy vs Step")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/entropy.png")

#L2 win prob loss vs step
plt.figure(figsize=(7, 4))
plt.plot(df["step"], df["eval_l2_win_prob_loss"], marker="o")
plt.xlabel("Step")
plt.ylabel("Eval L2 Win Prob Loss")
plt.title("Eval L2 Win Prob Loss vs Step")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/l2_win_prob_loss.png")