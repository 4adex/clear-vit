import re
import matplotlib.pyplot as plt

# Path to the log file
log_path = "SSSAttnLog.log"  # Replace with actual file path

# Regular expressions for train/test lines
train_pattern = re.compile(
    r"Train:.*?iter:\s*(\d+).*?loss:\s*([\d\.e+-]+).*?acc@1:\s*([\d\.e+-]+).*?acc@5:\s*([\d\.e+-]+)"
)
test_star_pattern = re.compile(
    r"\* Acc@1 ([\d\.e+-]+) Acc@5 ([\d\.e+-]+)"
)
test_full_pattern = re.compile(
    r"Test:.*?Acc@1:\s*([\d\.e+-]+)\s*Acc@5:\s*([\d\.e+-]+)"
)

# Storage lists
iterations, train_loss, train_a1, train_a5 = [], [], [], []
test_x, test_a1, test_a5 = [], [], []

# Iteration tracker
current_iter = 0

# Parse the log file
with open(log_path) as f:
    for line in f:
        m = train_pattern.search(line)
        if m:
            current_iter = int(m.group(1))  # update global iter
            iterations.append(current_iter)
            train_loss.append(float(m.group(2)))
            train_a1.append(float(m.group(3)))
            train_a5.append(float(m.group(4)))
            continue

        mt = test_star_pattern.search(line) or test_full_pattern.search(line)
        if mt:
            test_x.append(current_iter)  # associate with most recent train iter
            test_a1.append(float(mt.group(1)))
            test_a5.append(float(mt.group(2)))

# --- Plotting ---
plt.figure(figsize=(12, 6))

plt.plot(iterations, train_loss, label="Train Loss", color='steelblue', linewidth=1)
plt.plot(iterations, train_a1, label="Train Acc@1", color='orange', linewidth=1)
plt.plot(iterations, train_a5, label="Train Acc@5", color='green', linewidth=1)

plt.scatter(test_x, test_a1, label="Test Acc@1", color='cyan', marker='D', s=50)
plt.scatter(test_x, test_a5, label="Test Acc@5", color='magenta', marker='X', s=50)

plt.xlabel("Iteration")
plt.ylabel("Metric")
plt.title("Training and Test Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot.png")
plt.show()
