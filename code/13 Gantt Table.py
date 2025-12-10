import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime as dt

# -----------------------
# Configuration
# -----------------------

# Start and end dates
start_date = dt.date(2025, 9, 22)
end_date   = dt.date(2025, 12, 7)

# Generate date column (one date per week)
week_dates = [start_date + dt.timedelta(weeks=i) for i in range(12)]
week_labels = [d.strftime("%Y/%m/%d") for d in week_dates]  # Date format column names

# Row names (task numbers)
tasks = [f"{i+1}" for i in range(11)]

# Gantt bars for each task (start week, duration in weeks)
task_schedule = [
    (0, 3),   # Task 1: Week1–Week2
    (2, 2),   # Task 2
    (4, 1),   # Task 3
    (5, 1),   # Task 4
    (6, 1),   # Task 5
    (7, 1),   # Task 6
    (8, 1),   # Task 7
    (9, 1),   # Task 8
    (9, 2),   # Task 9
    (10, 2),   # Task 10
    (8, 4)   # Task 11 
]

# -----------------------
# Plotting
# -----------------------
fig, ax = plt.subplots(figsize=(15, 6))

ax.axis("off")

num_rows = len(tasks)
num_cols = len(week_labels)

cell_width = 1
cell_height = 0.5

# Background grid
for row in range(num_rows):
    for col in range(num_cols):
        ax.add_patch(
            patches.Rectangle(
                (col * cell_width, -row * cell_height),
                cell_width,
                cell_height,
                edgecolor="white",
                facecolor="#E6EEFA" if (row + col) % 2 == 0 else "#F1F6FF"
            )
        )

# Blue column headers
for col, label in enumerate(week_labels):
    ax.add_patch(
        patches.Rectangle(
            (col * cell_width, cell_height),
            cell_width,
            cell_height,
            edgecolor="white",
            facecolor="#3A78C3"
        )
    )
    ax.text(
        col * cell_width + cell_width / 2,
        cell_height + cell_height / 2,
        label,
        va="center", ha="center",
        color="white", fontsize=9, weight="bold"
    )

# Left-side task number column (blue)
for row, task in enumerate(tasks):
    ax.add_patch(
        patches.Rectangle(
            (-cell_width, -row * cell_height),
            cell_width,
            cell_height,
            edgecolor="white",
            facecolor="#3A78C3"
        )
    )
    ax.text(
        -cell_width / 2,
        -row * cell_height + cell_height / 2,
        task,
        va="center", ha="center",
        color="white", fontsize=10
    )

# Green Gantt bars
for row, (start_week, duration) in enumerate(task_schedule):
    ax.add_patch(
        patches.Rectangle(
            (start_week * cell_width, -row * cell_height),
            duration * cell_width,
            cell_height,
            edgecolor="white",
            facecolor="#8BC34A"   # Green color
        )
    )

# Boundary settings
ax.set_xlim(-cell_width, num_cols * cell_width)
ax.set_ylim(-num_rows * cell_height, cell_height * 2)

plt.title("Gantt Chart(2025/09/22 — 2025/12/11)", fontsize=14, pad=20)
plt.tight_layout()
plt.show()