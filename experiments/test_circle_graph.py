import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Test drawing circular nodes and connecting arrows
def draw_circle_node(ax, x, y, r, label, sublabel="", color="#3182CE", text_color="white"):
    circle = patches.Circle((x, y), r, facecolor=color, edgecolor="#1A365D", linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    if sublabel:
        ax.text(x, y + 1.2, label, ha='center', va='center', fontsize=8, fontweight='bold', color=text_color, zorder=4)
        ax.text(x, y - 1.8, sublabel, ha='center', va='center', fontsize=6.5, color=text_color, zorder=4)
    else:
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color=text_color, zorder=4)

draw_circle_node(ax, 20, 50, 6, "Conv1", "7x7, s2", "#4299E1")
draw_circle_node(ax, 45, 50, 6, "ResBlock", "Bottleneck", "#48BB78")
draw_circle_node(ax, 70, 50, 6, "FC Head", "1000 Class", "#ED8936")

# Connecting lines
ax.annotate("", xy=(39, 50), xytext=(26, 50),
            arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))
ax.annotate("", xy=(64, 50), xytext=(51, 50),
            arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"))

plt.savefig("/tmp/test_circle.png")
print("Circle test passed!")
