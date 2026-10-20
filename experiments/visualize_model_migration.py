import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import shutil

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

WORKSPACE = Path("/storage/xuedh_workspace/mopar-impl")
OUTPUT_IMG_DIR = WORKSPACE / "docs" / "images"
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/xuedh/.gemini/antigravity-cli/brain/01cca443-6fe0-4f36-8937-df4e3a3ef25a")

def draw_circle_node(ax, x, y, r, label, sublabel="", color="#3182CE", border_color="#1A365D", text_color="white", fs=7.5, sub_fs=6.0, aspect=2.133):
    # Perfect geometric circle compensation for rectangular canvas
    w = 2 * r / aspect
    h = 2 * r
    c = patches.Ellipse((x, y), w, h, facecolor=color, edgecolor=border_color, linewidth=1.5, zorder=4)
    ax.add_patch(c)
    if sublabel:
        ax.text(x, y + r*0.24, label, ha='center', va='center', fontsize=fs, fontweight='bold', color=text_color, zorder=5)
        ax.text(x, y - r*0.35, sublabel, ha='center', va='center', fontsize=sub_fs, color=text_color, zorder=5)
    else:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs, fontweight='bold', color=text_color, zorder=5)

def draw_arrow(ax, x1, y1, x2, y2, color="#4A5568", lw=1.6, label="", label_pos=0.5, label_offset=(0, 2), fs=6.5, curve=0):
    if curve == 0:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=12), zorder=3)
    else:
        rad = f"rad={curve}"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, connectionstyle=f"arc3,{rad}", mutation_scale=12), zorder=3)
    if label:
        lx = x1 + (x2 - x1) * label_pos + label_offset[0]
        ly = y1 + (y2 - y1) * label_pos + label_offset[1]
        ax.text(lx, ly, label, ha='center', va='center', fontsize=fs, fontweight='bold', color=color, zorder=5)

def draw_resnet50_circle_graph(ax, aspect=2.133):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 96, "ResNet-50: Computational Graph (Circles & Edges) $\\rightarrow$ Migrated Serverless FaaS Graph", 
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#1A202C')
    
    # ==================== 1. LEFT PANEL: MONOLITH DAG ====================
    box1 = patches.FancyBboxPatch((2, 6), 38, 86, boxstyle="round,pad=1.2", 
                                  facecolor="#F7FAFC", edgecolor="#A0AEC0", linewidth=1.5, linestyle="--")
    ax.add_patch(box1)
    ax.text(21, 88.5, "[Monolith DAG] 158 Layers in 1 Container", ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#2D3748')
    ax.text(21, 85.5, "Unified Memory Pool: 1000 MB Quota (MC: 165.0 MB·s)", ha='center', va='center', 
            fontsize=7.5, color='#718096')

    mono_nodes = [
        ("m_in", 21, 78, 3.2, "Input", "Image", "#E2E8F0", "#718096", "#2D3748"),
        ("m_c1", 21, 68, 3.8, "Conv1", "7x7, s2", "#4299E1", "#2B6CB0", "white"),
        ("m_mp", 21, 58, 3.5, "MaxPool", "3x3, s2", "#63B3ED", "#2B6CB0", "white"),
        ("m_l1", 21, 48, 4.0, "Layer 1", "3 Blocks", "#3182CE", "#1A365D", "white"),
        ("m_l2", 21, 38, 4.0, "Layer 2", "4 Blocks", "#3182CE", "#1A365D", "white"),
        ("m_l3", 21, 28, 4.0, "Layer 3", "6 Blocks", "#3182CE", "#1A365D", "white"),
        ("m_l4_0", 14, 18, 3.8, "L4.0", "Downsample", "#2B6CB0", "#1A365D", "white"),
        ("m_l4_1", 28, 18, 3.8, "L4.1-2", "Residual", "#2B6CB0", "#1A365D", "white"),
        ("m_avg", 21, 9, 3.5, "AvgPool", "7x7", "#48BB78", "#22543D", "white"),
        ("m_fc", 33, 9, 3.5, "FC", "1000-d", "#ED8936", "#C05621", "white"),
    ]
    coords = {}
    for nid, nx, ny, nr, nlab, nsub, ncol, nbor, ntxt in mono_nodes:
        coords[nid] = (nx, ny, nr)
        draw_circle_node(ax, nx, ny, nr, nlab, nsub, ncol, nbor, ntxt, aspect=aspect)

    # Directed Edges between Monolith nodes
    draw_arrow(ax, coords["m_in"][0], coords["m_in"][1]-coords["m_in"][2], coords["m_c1"][0], coords["m_c1"][1]+coords["m_c1"][2])
    draw_arrow(ax, coords["m_c1"][0], coords["m_c1"][1]-coords["m_c1"][2], coords["m_mp"][0], coords["m_mp"][1]+coords["m_mp"][2])
    draw_arrow(ax, coords["m_mp"][0], coords["m_mp"][1]-coords["m_mp"][2], coords["m_l1"][0], coords["m_l1"][1]+coords["m_l1"][2])
    draw_arrow(ax, coords["m_l1"][0], coords["m_l1"][1]-coords["m_l1"][2], coords["m_l2"][0], coords["m_l2"][1]+coords["m_l2"][2])
    draw_arrow(ax, coords["m_l2"][0], coords["m_l2"][1]-coords["m_l2"][2], coords["m_l3"][0], coords["m_l3"][1]+coords["m_l3"][2])
    
    # Layer 3 to Layer 4
    draw_arrow(ax, coords["m_l3"][0]-1.5, coords["m_l3"][1]-coords["m_l3"][2], coords["m_l4_0"][0], coords["m_l4_0"][1]+coords["m_l4_0"][2])
    draw_arrow(ax, coords["m_l4_0"][0]+coords["m_l4_0"][2]/aspect, coords["m_l4_0"][1], coords["m_l4_1"][0]-coords["m_l4_1"][2]/aspect, coords["m_l4_1"][1])
    draw_arrow(ax, coords["m_l4_1"][0]-1.5, coords["m_l4_1"][1]-coords["m_l4_1"][2], coords["m_avg"][0]+1.5, coords["m_avg"][1]+coords["m_avg"][2])
    draw_arrow(ax, coords["m_avg"][0]+coords["m_avg"][2]/aspect, coords["m_avg"][1], coords["m_fc"][0]-coords["m_fc"][2]/aspect, coords["m_fc"][1])

    # Residual Shortcut skip curves
    draw_arrow(ax, coords["m_l1"][0]+coords["m_l1"][2]/aspect, coords["m_l1"][1], coords["m_l2"][0]+coords["m_l2"][2]/aspect, coords["m_l2"][1], 
               color="#3182CE", lw=1.2, curve=0.4, label="skip", label_offset=(3.5, 0))
    draw_arrow(ax, coords["m_l2"][0]+coords["m_l2"][2]/aspect, coords["m_l2"][1], coords["m_l3"][0]+coords["m_l3"][2]/aspect, coords["m_l3"][1], 
               color="#3182CE", lw=1.2, curve=0.4, label="skip", label_offset=(3.5, 0))

    # Cut Lines
    ax.plot([4, 38], [23, 23], color="#E53E3E", linestyle=":", linewidth=2, zorder=6)
    ax.text(6, 24.5, "✂ Cut 1 (Decouple Residual Channels)", fontsize=7, color="#E53E3E", fontweight='bold')

    ax.plot([4, 38], [13.5, 13.5], color="#E53E3E", linestyle=":", linewidth=2, zorder=6)
    ax.text(6, 14.8, "✂ Cut 2 (Pooling Tensor Compression)", fontsize=7, color="#E53E3E", fontweight='bold')

    # ==================== TRANSITION ARROW ====================
    ax.annotate("", xy=(46, 50), xytext=(41.5, 50),
                arrowprops=dict(facecolor='#2B6CB0', edgecolor='#2B6CB0', width=3, headwidth=8))
    ax.text(43.8, 54, "Ours DP\nSlicing", ha='center', va='center', fontsize=8, fontweight='bold', color='#2B6CB0')

    # ==================== 2. RIGHT PANEL: MIGRATED FAAS SERVICES ====================
    # Cluster 1: Lambda 1 (Ingress)
    b_lam1 = patches.FancyBboxPatch((48, 62), 50, 30, boxstyle="round,pad=1.0", 
                                    facecolor="#EBF8FF", edgecolor="#3182CE", linewidth=1.5)
    ax.add_patch(b_lam1)
    ax.text(50, 89, "Lambda 1: `ResNet_Slice0_Api`", fontsize=9, fontweight='bold', color="#2B6CB0")
    ax.text(88, 89, "Quota: 128 MB (Footprint: 55.6 MB)", fontsize=7.5, color="#2B6CB0", fontweight='bold')

    # Nodes in Lambda 1
    r1 = 4.2
    draw_circle_node(ax, 54, 75, r1, "API Gate", "POST /pred", "#ED8936", "#C05621", "white", aspect=aspect)
    draw_circle_node(ax, 66, 75, r1, "Conv1", "7x7, s2", "#4299E1", "#2B6CB0", "white", aspect=aspect)
    draw_circle_node(ax, 79, 75, r1, "Layer 1-3", "13 Residual", "#3182CE", "#1A365D", "white", aspect=aspect)
    draw_circle_node(ax, 92, 75, r1, "L4.0", "Exit Act", "#2B6CB0", "#1A365D", "white", aspect=aspect)

    draw_arrow(ax, 54 + r1/aspect, 75, 66 - r1/aspect, 75)
    draw_arrow(ax, 66 + r1/aspect, 75, 79 - r1/aspect, 75)
    draw_arrow(ax, 79 + r1/aspect, 75, 92 - r1/aspect, 75)

    # Inter-Service Flow 1
    draw_arrow(ax, 92, 75 - r1, 92, 57, color="#38A169", lw=2.2, label="Act Tensor (EventBridge/SQS)", label_pos=0.5, label_offset=(-14, 0))

    # Cluster 2: Lambda 2 (Deep Residuals)
    b_lam2 = patches.FancyBboxPatch((48, 34), 50, 22, boxstyle="round,pad=1.0", 
                                    facecolor="#F0FFF4", edgecolor="#38A169", linewidth=1.5)
    ax.add_patch(b_lam2)
    ax.text(50, 53, "Lambda 2: `ResNet_Slice1_Worker`", fontsize=9, fontweight='bold', color="#22543D")
    ax.text(88, 53, "Quota: 128 MB (Footprint: 34.1 MB)", fontsize=7.5, color="#22543D", fontweight='bold')

    draw_circle_node(ax, 60, 43, r1, "Event In", "Tensor", "#38A169", "#22543D", "white", aspect=aspect)
    draw_circle_node(ax, 76, 43, r1, "L4.1 - L4.2", "19 Layers", "#2F855A", "#22543D", "white", aspect=aspect)
    draw_circle_node(ax, 91, 43, r1, "Out Feat", "2048-dim", "#276749", "#1C4532", "white", aspect=aspect)

    draw_arrow(ax, 60 + r1/aspect, 43, 76 - r1/aspect, 43)
    draw_arrow(ax, 76 + r1/aspect, 43, 91 - r1/aspect, 43)

    # Inter-Service Flow 2
    draw_arrow(ax, 91, 43 - r1, 91, 25, color="#38A169", lw=2.2, label="Feature Vector (Direct)", label_pos=0.5, label_offset=(-12, 0))

    # Cluster 3: Lambda 3 (Classifier)
    b_lam3 = patches.FancyBboxPatch((48, 6), 50, 20, boxstyle="round,pad=1.0", 
                                    facecolor="#FFFAF0", edgecolor="#DD6B20", linewidth=1.5)
    ax.add_patch(b_lam3)
    ax.text(50, 23, "Lambda 3: `ResNet_Slice2_Worker`", fontsize=9, fontweight='bold', color="#7B341E")
    ax.text(88, 23, "Quota: 128 MB (Footprint: 7.8 MB)", fontsize=7.5, color="#7B341E", fontweight='bold')

    draw_circle_node(ax, 60, 14, r1, "AvgPool", "Global", "#ED8936", "#C05621", "white", aspect=aspect)
    draw_circle_node(ax, 75, 14, r1, "FC 1000", "Softmax", "#DD6B20", "#9C4221", "white", aspect=aspect)
    draw_circle_node(ax, 90, 14, r1, "JSON", "Response", "#48BB78", "#22543D", "white", aspect=aspect)

    draw_arrow(ax, 60 + r1/aspect, 14, 75 - r1/aspect, 14)
    draw_arrow(ax, 75 + r1/aspect, 14, 90 - r1/aspect, 14)

    # Overall Metrics Banner
    ax.text(50, 2.5, "Result: MC 165.0 -> 41.3 MB·s (-74.96% Savings) | Tensor Diff Δ = 0.00e+00 (Strictly Lossless)", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#22543D")


def draw_qwen3_circle_graph(ax, aspect=2.133):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 96, "Qwen3-1.7B: Computational Graph $\\rightarrow$ 4-GPU NVLink Pipeline Microservices", 
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#1A202C')

    # ==================== 1. LEFT PANEL: MONOLITH DAG ====================
    box1 = patches.FancyBboxPatch((2, 6), 34, 86, boxstyle="round,pad=1.2", 
                                  facecolor="#F7FAFC", edgecolor="#A0AEC0", linewidth=1.5, linestyle="--")
    ax.add_patch(box1)
    ax.text(19, 88.5, "[Monolith LLM] 3.8 GB Weights", ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#2D3748')
    ax.text(19, 85.5, "GPU 0 Bottleneck: 3.21 GB VRAM (TP: 27.81 char/s)", ha='center', va='center', 
            fontsize=7.2, color='#718096')

    q_nodes = [
        ("q_in", 19, 78, 3.2, "Prompt", "Tokens", "#E2E8F0", "#718096", "#2D3748"),
        ("q_emb", 19, 68, 4.2, "Embed", "152k Vocab", "#FEB2B2", "#E53E3E", "#742A2A"),
        ("q_d0", 19, 57, 4.2, "Dec 0..3", "4 Layers", "#BEE3F8", "#2B6CB0", "#1A365D"),
        ("q_d1", 19, 46, 4.2, "Dec 4..11", "8 Layers", "#90CDF4", "#2B6CB0", "#1A365D"),
        ("q_d2", 19, 35, 4.2, "Dec 12..19", "8 Layers", "#63B3ED", "#2B6CB0", "white"),
        ("q_d3", 19, 24, 4.2, "Dec 20..27", "8 Layers", "#3182CE", "#1A365D", "white"),
        ("q_head", 19, 13, 4.2, "LM-Head", "152k Proj", "#FEB2B2", "#E53E3E", "#742A2A"),
    ]
    q_coords = {}
    for qid, qx, qy, qr, qlab, qsub, qcol, qbor, qtxt in q_nodes:
        q_coords[qid] = (qx, qy, qr)
        draw_circle_node(ax, qx, qy, qr, qlab, qsub, qcol, qbor, qtxt, aspect=aspect)

    # Monolith edges
    draw_arrow(ax, q_coords["q_in"][0], q_coords["q_in"][1]-q_coords["q_in"][2], q_coords["q_emb"][0], q_coords["q_emb"][1]+q_coords["q_emb"][2])
    draw_arrow(ax, q_coords["q_emb"][0], q_coords["q_emb"][1]-q_coords["q_emb"][2], q_coords["q_d0"][0], q_coords["q_d0"][1]+q_coords["q_d0"][2])
    draw_arrow(ax, q_coords["q_d0"][0], q_coords["q_d0"][1]-q_coords["q_d0"][2], q_coords["q_d1"][0], q_coords["q_d1"][1]+q_coords["q_d1"][2])
    draw_arrow(ax, q_coords["q_d1"][0], q_coords["q_d1"][1]-q_coords["q_d1"][2], q_coords["q_d2"][0], q_coords["q_d2"][1]+q_coords["q_d2"][2])
    draw_arrow(ax, q_coords["q_d2"][0], q_coords["q_d2"][1]-q_coords["q_d2"][2], q_coords["q_d3"][0], q_coords["q_d3"][1]+q_coords["q_d3"][2])
    draw_arrow(ax, q_coords["q_d3"][0], q_coords["q_d3"][1]-q_coords["q_d3"][2], q_coords["q_head"][0], q_coords["q_head"][1]+q_coords["q_head"][2])

    # Tied-weight loop (curved dashed line connecting Embed and LM-Head)
    draw_arrow(ax, q_coords["q_emb"][0]-q_coords["q_emb"][2]/aspect, q_coords["q_emb"][1],
               q_coords["q_head"][0]-q_coords["q_head"][2]/aspect, q_coords["q_head"][1],
               color="#E53E3E", lw=1.5, curve=-0.6, label="Tied Weight Sharing", label_offset=(-7, 0))

    # Cut Lines
    ax.plot([4, 34], [62.5, 62.5], color="#E53E3E", linestyle=":", linewidth=2, zorder=6)
    ax.text(5, 63.8, "✂ Cut 0: Embed + 4 Layers Only", fontsize=6.8, color="#E53E3E", fontweight='bold')

    ax.plot([4, 34], [51.5, 51.5], color="#E53E3E", linestyle=":", linewidth=2, zorder=6)
    ax.plot([4, 34], [40.5, 40.5], color="#E53E3E", linestyle=":", linewidth=2, zorder=6)
    ax.plot([4, 34], [29.5, 29.5], color="#E53E3E", linestyle=":", linewidth=2, zorder=6)

    # ==================== TRANSITION ARROW ====================
    ax.annotate("", xy=(42, 50), xytext=(38, 50),
                arrowprops=dict(facecolor='#805AD5', edgecolor='#805AD5', width=3, headwidth=8))
    ax.text(40, 54, "Asymmetric\nPipeline", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#805AD5')

    # ==================== 2. RIGHT PANEL: 4-GPU MICROSERVICES ====================
    qr = 4.2
    gpus = [
        ("GPU 0 (Ingress): `Qwen_Slice0_Api`", 44, 72, 54, 18, "#FAF5FF", "#805AD5",
         [("API Gate", "POST /chat", "#ED8936", "#C05621", "white", 52),
          ("Embed", "152k Vocab", "#FEB2B2", "#E53E3E", "#742A2A", 68),
          ("Dec 0..3", "4 Layers", "#9F7AEA", "#6B46C1", "white", 84)],
         "VRAM: 2.20 GB (Quota: 3072 MB)"),
        
        ("GPU 1 (Worker 1): `Qwen_Slice1_Worker`", 44, 50, 54, 18, "#F7FAFC", "#4A5568",
         [("NVLink In", "Hidden", "#CBD5E0", "#718096", "#2D3748", 54),
          ("Dec 4..11", "8 Layers", "#805AD5", "#553C9A", "white", 73),
          ("NVLink Out", "States", "#CBD5E0", "#718096", "#2D3748", 92)],
         "VRAM: 1.41 GB (-56.1% Load Reduction)"),
        
        ("GPU 2 (Worker 2): `Qwen_Slice2_Worker`", 44, 28, 54, 18, "#F7FAFC", "#4A5568",
         [("NVLink In", "Hidden", "#CBD5E0", "#718096", "#2D3748", 54),
          ("Dec 12..19", "8 Layers", "#6B46C1", "#44337A", "white", 73),
          ("NVLink Out", "States", "#CBD5E0", "#718096", "#2D3748", 92)],
         "VRAM: 1.41 GB (-56.1% Load Reduction)"),
        
        ("GPU 3 (Worker 3): `Qwen_Slice3_Worker`", 44, 6, 54, 18, "#FFF5F5", "#E53E3E",
         [("Dec 20..27", "8 Layers", "#553C9A", "#322659", "white", 52),
          ("RMSNorm", "Layer", "#CBD5E0", "#718096", "#2D3748", 66),
          ("LM-Head", "152k Proj", "#FEB2B2", "#E53E3E", "#742A2A", 80),
          ("Stream", "Token Out", "#48BB78", "#22543D", "white", 94)],
         "VRAM: 1.41 GB (Quota: 2048 MB)"),
    ]

    for gtitle, gx, gy, gw, gh, gbg, gbor, gnodes, gmetrics in gpus:
        b_gpu = patches.FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=1.0",
                                       facecolor=gbg, edgecolor=gbor, linewidth=1.5)
        ax.add_patch(b_gpu)
        ax.text(gx + 2, gy + gh - 3.5, gtitle, fontsize=8.5, fontweight='bold', color=gbor)
        ax.text(gx + gw - 2, gy + gh - 3.5, gmetrics, fontsize=7.2, fontweight='bold', color=gbor, ha='right')

        # Draw circular nodes inside GPU cluster
        for nlab, nsub, ncol, nbor, ntxt, nx in gnodes:
            draw_circle_node(ax, nx, gy + 6.5, qr, nlab, nsub, ncol, nbor, ntxt, aspect=aspect)

        # Connect circular nodes inside cluster
        for i in range(len(gnodes) - 1):
            x1 = gnodes[i][5] + qr/aspect
            x2 = gnodes[i+1][5] - qr/aspect
            draw_arrow(ax, x1, gy + 6.5, x2, gy + 6.5)

    # Inter-GPU NVLink P2P Edges
    draw_arrow(ax, 84, 72, 84, 68, color="#DD6B20", lw=2.5, label="NVLink 47.5 GB/s", label_offset=(11, 0))
    draw_arrow(ax, 84, 50, 84, 46, color="#DD6B20", lw=2.5, label="NVLink 47.5 GB/s", label_offset=(11, 0))
    draw_arrow(ax, 84, 28, 84, 24, color="#DD6B20", lw=2.5, label="NVLink 47.5 GB/s", label_offset=(11, 0))

    # Overall Metrics Banner
    ax.text(50, 2.5, "Result: 4-GPU Generation TP 33.42 -> 39.11 char/s (+17.0% Speedup) | Communication Bubble: -35.7%", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#9C4221")


def main():
    # 1. Combined Dual-Panel Diagram (aspect = 16 / 7.5 = 2.133)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 15))
    plt.subplots_adjust(hspace=0.18, top=0.96, bottom=0.03, left=0.03, right=0.97)
    
    draw_resnet50_circle_graph(ax1, aspect=2.133)
    draw_qwen3_circle_graph(ax2, aspect=2.133)
    
    combined_path = OUTPUT_IMG_DIR / "model_migration_visualization.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()
    shutil.copy(combined_path, ARTIFACT_DIR / "model_migration_visualization.png")
    print(f"Generated combined circle graph: {combined_path}")

    # 2. ResNet-50 Standalone Diagram
    fig_r, ax_r = plt.subplots(figsize=(16, 7.5))
    plt.subplots_adjust(top=0.94, bottom=0.06, left=0.03, right=0.97)
    draw_resnet50_circle_graph(ax_r, aspect=2.133)
    r_path = OUTPUT_IMG_DIR / "resnet50_migration_architecture.png"
    plt.savefig(r_path, dpi=300, bbox_inches="tight")
    plt.close()
    shutil.copy(r_path, ARTIFACT_DIR / "resnet50_migration_architecture.png")
    print(f"Generated ResNet-50 circle graph: {r_path}")

    # 3. Qwen3 Standalone Diagram
    fig_q, ax_q = plt.subplots(figsize=(16, 7.5))
    plt.subplots_adjust(top=0.94, bottom=0.06, left=0.03, right=0.97)
    draw_qwen3_circle_graph(ax_q, aspect=2.133)
    q_path = OUTPUT_IMG_DIR / "qwen3_migration_architecture.png"
    plt.savefig(q_path, dpi=300, bbox_inches="tight")
    plt.close()
    shutil.copy(q_path, ARTIFACT_DIR / "qwen3_migration_architecture.png")
    print(f"Generated Qwen3 circle graph: {q_path}")

if __name__ == "__main__":
    main()
