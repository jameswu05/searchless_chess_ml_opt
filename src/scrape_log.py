import re
import csv
import pandas as pd
import matplotlib.pyplot as plt

log_file = "data/log.txt" 
output_csv = "data/output/output_log.csv"
save_dir = "data/output"
split_step = 100000
STEP_INTERVAL = 5000

#plotting helpers
#-------------------
def plot_full(df, save_dir):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9))

    #Loss
    axs[0].plot(df["step"], df["loss"])
    axs[0].set_title("Loss (Linear)")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")

    axs[1].plot(df["step"], df["loss"])
    axs[1].set_yscale("log")
    axs[1].set_title("Loss (Log Scale)")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_full.png", dpi=300)
    plt.close()

    # Grad
    fig, axs = plt.subplots(2, 1, figsize=(7, 9))

    axs[0].plot(df["step"], df["grad_norm_unclipped"])
    axs[0].set_title("Gradient Norm (Linear)")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Gradient")

    axs[1].plot(df["step"], df["grad_norm_unclipped"])
    axs[1].set_yscale("log")
    axs[1].set_title("Gradient Norm (Log Scale)")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Gradient")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/grad_full.png", dpi=300)
    plt.close()

def plot_early(df, save_dir, split_step=100000):
    df_early = df[df["step"] <= split_step]

    #Loss
    fig, axs = plt.subplots(2, 1, figsize=(7, 9))

    axs[0].plot(df_early["step"], df_early["loss"])
    axs[0].set_title("Loss (0–100k Linear)")

    axs[1].plot(df_early["step"], df_early["loss"])
    axs[1].set_yscale("log")
    axs[1].set_title("Loss (0–100k Log)")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_early.png", dpi=300)
    plt.close()

    #Grad
    fig, axs = plt.subplots(2, 1, figsize=(7, 9))

    axs[0].plot(df_early["step"], df_early["grad_norm_unclipped"])
    axs[0].set_title("Grad Norm (0–100k Linear)")

    axs[1].plot(df_early["step"], df_early["grad_norm_unclipped"])
    axs[1].set_yscale("log")
    axs[1].set_title("Grad Norm (0–100k Log)")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/grad_early.png", dpi=300)
    plt.close()

def plot_late(df, save_dir, split_step=100000):
    df_late = df[df["step"] > split_step]

    # Loss
    fig, axs = plt.subplots(2, 1, figsize=(7, 9))

    axs[0].plot(df_late["step"], df_late["loss"])
    axs[0].set_title("Loss (100k+ Linear)")

    axs[1].plot(df_late["step"], df_late["loss"])
    axs[1].set_yscale("log")
    axs[1].set_title("Loss (100k+ Log)")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_late.png", dpi=300)
    plt.close()

    #Grad
    fig, axs = plt.subplots(2, 1, figsize=(7, 9))

    axs[0].plot(df_late["step"], df_late["grad_norm_unclipped"])
    axs[0].set_title("Grad Norm (100k+ Linear)")

    axs[1].plot(df_late["step"], df_late["grad_norm_unclipped"])
    axs[1].set_yscale("log")
    axs[1].set_title("Grad Norm (100k+ Log)")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/grad_late.png", dpi=300)
    plt.close()

#Parse file
#-----------------------
pattern = re.compile(
    r"step:\s*(\d+)\s*\|\s*loss:\s*([0-9.eE+-]+)\s*\|\s*grad_norm_unclipped:\s*([0-9.eE+-]+)"
)

rows = []
with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            grad = float(match.group(3))

            if step % STEP_INTERVAL == 0 and step <= 1000000:
                rows.append([step, loss, grad])

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss", "grad_norm_unclipped"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {output_csv}")


#Make plots
#--------------------------
df = pd.read_csv(output_csv)
plot_full(df, save_dir)
plot_early(df, save_dir)
plot_late(df, save_dir)