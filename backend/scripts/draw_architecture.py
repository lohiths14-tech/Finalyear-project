import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    # Setup figure and axes
    fig, ax = plt.subplots(figsize=(7, 11), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Styling constants
    box_params = dict(boxstyle="round,pad=0.3", lw=1.5, alpha=0.9)
    font_title = dict(fontsize=10, fontweight="bold", family="sans-serif")
    font_body = dict(fontsize=8, family="sans-serif")
    arrow_params = dict(arrowstyle="-|>", lw=1.5, color="#2C3E50", mutation_scale=15)

    # ── STAGE 0 ──
    # Position: center=(5, 15), size=(6, 0.8)
    rect0 = patches.FancyBboxPatch((2.0, 14.4), 6.0, 0.9, boxstyle="round,pad=0.1",
                                   edgecolor="#2980B9", facecolor="#EBF5FB", lw=1.5)
    ax.add_patch(rect0)
    ax.text(5, 15.0, "STAGE 0: INGESTION & GATEWAY", ha="center", va="center", **font_title)
    ax.text(5, 14.65, "FastAPI Endpoint • MIME Validation • 10MB Limit Check", ha="center", va="center", **font_body)

    # Arrow 0 -> 1
    ax.annotate("", xy=(5, 13.1), xytext=(5, 14.4), arrowprops=arrow_params)

    # ── STAGE 1 ──
    rect1 = patches.FancyBboxPatch((2.0, 12.1), 6.0, 1.0, boxstyle="round,pad=0.1",
                                   edgecolor="#D35400", facecolor="#FDF2E9", lw=1.5)
    ax.add_patch(rect1)
    ax.text(5, 12.8, "STAGE 1: IMAGE QUALITY ASSESSMENT (IQA)", ha="center", va="center", **font_title)
    ax.text(5, 12.4, "7-Dimension IQI Gate Check (Blur, Noise, Contrast, etc.)\nRejects if IQI < 0.55 with retake guidance", ha="center", va="center", **font_body)

    # Reject Branch
    arrow_reject = dict(arrowstyle="-|>", lw=1.5, color="#C0392B", mutation_scale=15)
    ax.annotate("", xy=(9.2, 12.6), xytext=(8.0, 12.6), arrowprops=arrow_reject)
    rect_reject = patches.FancyBboxPatch((9.2, 12.1), 1.6, 1.0, boxstyle="round,pad=0.1",
                                         edgecolor="#C0392B", facecolor="#FDEDEC", lw=1.5)
    ax.add_patch(rect_reject)
    ax.text(10.0, 12.6, "REJECT\nHTTP 422", ha="center", va="center", color="#C0392B", fontsize=8, fontweight="bold")

    # Arrow 1 -> 2
    ax.annotate("", xy=(5, 10.8), xytext=(5, 12.1), arrowprops=arrow_params)

    # ── STAGE 2 ──
    rect2 = patches.FancyBboxPatch((2.0, 9.3), 6.0, 1.5, boxstyle="round,pad=0.1",
                                   edgecolor="#8E44AD", facecolor="#F5EEF8", lw=1.5)
    ax.add_patch(rect2)
    ax.text(5, 10.45, "STAGE 2: ADVANCED PREPROCESSING", ha="center", va="center", **font_title)
    ax.text(5, 9.7, "1. NLM Denoising (LAB space)    2. DCP Dehazing (t=0.95)\n3. CLAHE Contrast Normalization  4. Gray World White Balance\n5. Veg Enhancement (ExG+CIVE) 6. Shadow & Soil Suppression",
            ha="center", va="center", **font_body)

    # Arrow 2 -> 3
    ax.annotate("", xy=(5, 7.8), xytext=(5, 9.3), arrowprops=arrow_params)

    # ── STAGE 3 ──
    # Container for parallel branches
    rect3 = patches.FancyBboxPatch((0.5, 3.8), 9.0, 4.0, boxstyle="round,pad=0.1",
                                   edgecolor="#16A085", facecolor="#E8F8F5", lw=1.5, ls="--")
    ax.add_patch(rect3)
    ax.text(5, 7.5, "STAGE 3: SPATIO-SPECTRAL-TEMPORAL (SST) FEATURE EXTRACTOR", ha="center", va="center", **font_title)

    # Sub-box 3A: Spectral
    rect3a = patches.FancyBboxPatch((0.8, 4.2), 2.6, 2.7, boxstyle="round,pad=0.1",
                                    edgecolor="#27AE60", facecolor="#E8F8F5", lw=1.2)
    ax.add_patch(rect3a)
    ax.text(2.1, 6.6, "NGVCNet\n(Spectral Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#1E8449")
    ax.text(2.1, 5.2, "128-bin NGVC\nHistogram\n+\n1D-CNN Blocks\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    # Sub-box 3B: Spatial
    rect3b = patches.FancyBboxPatch((3.7, 4.2), 2.6, 2.7, boxstyle="round,pad=0.1",
                                    edgecolor="#2980B9", facecolor="#EBF5FB", lw=1.2)
    ax.add_patch(rect3b)
    ax.text(5.0, 6.6, "SpatialGraphNet\n(Spatial Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#21618C")
    ax.text(5.0, 5.2, "Centroid Adjacency\nGraph (1.5x stride)\n+\n2-Layer GraphSAGE\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    # Sub-box 3C: Temporal
    rect3c = patches.FancyBboxPatch((6.6, 4.2), 2.6, 2.7, boxstyle="round,pad=0.1",
                                    edgecolor="#D35400", facecolor="#FDF2E9", lw=1.2)
    ax.add_patch(rect3c)
    ax.text(7.9, 6.6, "TemporalProgNet\n(Temporal Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#A04000")
    ax.text(7.9, 5.2, "5-Step Historical\nEmbedding Sequence\n+\n2-Layer BiLSTM\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    # Arrow 3 -> 4 (Merging arrows)
    ax.annotate("", xy=(5, 2.3), xytext=(2.1, 4.2), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 2.3), xytext=(5.0, 4.2), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 2.3), xytext=(7.9, 4.2), arrowprops=arrow_params)

    # ── STAGE 4 ──
    rect4 = patches.FancyBboxPatch((2.0, 1.3), 6.0, 1.0, boxstyle="round,pad=0.1",
                                   edgecolor="#16A085", facecolor="#E8F8F5", lw=1.5)
    ax.add_patch(rect4)
    ax.text(5, 2.0, "STAGE 4: AGRIVISIONENSEMBLE META-LEARNER", ha="center", va="center", **font_title)
    ax.text(5, 1.6, "Concatenated Embeddings (192-dim) • 3-Layer MLP\nDual Heads: Classification (Softmax) & Severity (Sigmoid)", ha="center", va="center", **font_body)

    # Arrow 4 -> 5
    ax.annotate("", xy=(5, 0.0), xytext=(5, 1.3), arrowprops=arrow_params)

    # Shift diagram up for stage 5 & 6
    plt.close(fig)

    # To fit Stage 5 and Stage 6 cleanly, let's adjust coordinate space.
    # We will draw a final beautiful diagram with 18 units of height.
    fig, ax = plt.subplots(figsize=(7, 12.5), dpi=300)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 18.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Stage 0 (y: 16.5 to 17.5)
    rect0 = patches.FancyBboxPatch((2.0, 16.5), 6.0, 1.0, boxstyle="round,pad=0.1",
                                   edgecolor="#2980B9", facecolor="#EBF5FB", lw=1.5)
    ax.add_patch(rect0)
    ax.text(5, 17.15, "STAGE 0: INGESTION & GATEWAY", ha="center", va="center", **font_title)
    ax.text(5, 16.8, "FastAPI Endpoint • MIME Validation • 10MB Limit Check", ha="center", va="center", **font_body)
    ax.annotate("", xy=(5, 15.0), xytext=(5, 16.5), arrowprops=arrow_params)

    # Stage 1 (y: 13.6 to 15.0)
    rect1 = patches.FancyBboxPatch((2.0, 13.6), 6.0, 1.4, boxstyle="round,pad=0.1",
                                   edgecolor="#D35400", facecolor="#FDF2E9", lw=1.5)
    ax.add_patch(rect1)
    ax.text(5, 14.5, "STAGE 1: IMAGE QUALITY ASSESSMENT (IQA)", ha="center", va="center", **font_title)
    ax.text(5, 14.0, "7-Dimension IQI Gate Check (Blur, Noise, Contrast, etc.)\nRejects if IQI < 0.55 with retake guidance", ha="center", va="center", **font_body)
    ax.annotate("", xy=(9.0, 14.3), xytext=(8.0, 14.3), arrowprops=arrow_reject)
    rect_reject = patches.FancyBboxPatch((9.0, 13.6), 1.8, 1.4, boxstyle="round,pad=0.1",
                                         edgecolor="#C0392B", facecolor="#FDEDEC", lw=1.5)
    ax.add_patch(rect_reject)
    ax.text(9.9, 14.3, "REJECT\nHTTP 422", ha="center", va="center", color="#C0392B", fontsize=8, fontweight="bold")
    ax.annotate("", xy=(5, 11.8), xytext=(5, 13.6), arrowprops=arrow_params)

    # Stage 2 (y: 9.8 to 11.8)
    rect2 = patches.FancyBboxPatch((2.0, 9.8), 6.0, 2.0, boxstyle="round,pad=0.1",
                                   edgecolor="#8E44AD", facecolor="#F5EEF8", lw=1.5)
    ax.add_patch(rect2)
    ax.text(5, 11.2, "STAGE 2: ADVANCED PREPROCESSING PIPELINE", ha="center", va="center", **font_title)
    ax.text(5, 10.4, "• Non-Local Means Denoising (CIE LAB space)\n• Dark Channel Prior Dehazing (transmission mapping)\n• CLAHE Contrast Normalization & Gray World White Balance\n• Multi-Index Veg Enhancement & Shadow Inpainting",
            ha="left", va="center", **font_body)
    ax.annotate("", xy=(5, 8.2), xytext=(5, 9.8), arrowprops=arrow_params)

    # Stage 3 (y: 4.0 to 8.2)
    rect3 = patches.FancyBboxPatch((0.2, 4.0), 9.6, 4.2, boxstyle="round,pad=0.1",
                                   edgecolor="#16A085", facecolor="#E8F8F5", lw=1.5, ls="--")
    ax.add_patch(rect3)
    ax.text(5, 7.8, "STAGE 3: SPATIO-SPECTRAL-TEMPORAL (SST) FEATURE EXTRACTORS", ha="center", va="center", **font_title)

    rect3a = patches.FancyBboxPatch((0.5, 4.3), 2.8, 3.0, boxstyle="round,pad=0.1",
                                    edgecolor="#27AE60", facecolor="#E8F8F5", lw=1.2)
    ax.add_patch(rect3a)
    ax.text(1.9, 6.8, "NGVCNet\n(Spectral Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#1E8449")
    ax.text(1.9, 5.2, "128-bin NGVC\nHistogram\n+\n1D-CNN Blocks\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    rect3b = patches.FancyBboxPatch((3.6, 4.3), 2.8, 3.0, boxstyle="round,pad=0.1",
                                    edgecolor="#2980B9", facecolor="#EBF5FB", lw=1.2)
    ax.add_patch(rect3b)
    ax.text(5.0, 6.8, "SpatialGraphNet\n(Spatial Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#21618C")
    ax.text(5.0, 5.2, "Centroid Adjacency\nGraph (1.5x stride)\n+\n2-Layer GraphSAGE\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    rect3c = patches.FancyBboxPatch((6.7, 4.3), 2.8, 3.0, boxstyle="round,pad=0.1",
                                    edgecolor="#D35400", facecolor="#FDF2E9", lw=1.2)
    ax.add_patch(rect3c)
    ax.text(8.1, 6.8, "TemporalProgNet\n(Temporal Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#A04000")
    ax.text(8.1, 5.2, "5-Step Historical\nEmbedding Sequence\n+\n2-Layer BiLSTM\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    ax.annotate("", xy=(5, 2.3), xytext=(1.9, 4.3), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 2.3), xytext=(5.0, 4.3), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 2.3), xytext=(8.1, 4.3), arrowprops=arrow_params)

    # Stage 4 (y: 2.3 to 3.5)
    rect4 = patches.FancyBboxPatch((2.0, 2.3), 6.0, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor="#16A085", facecolor="#E8F8F5", lw=1.5)
    ax.add_patch(rect4)
    ax.text(5, 3.0, "STAGE 4: AGRIVISIONENSEMBLE META-LEARNER", ha="center", va="center", **font_title)
    ax.text(5, 2.6, "Concatenated Embeddings (192-dim) • 3-Layer MLP\nDual Heads: Classification (Softmax) & Severity (Sigmoid)", ha="center", va="center", **font_body)
    ax.annotate("", xy=(5, 0.7), xytext=(5, 2.3), arrowprops=arrow_params)

    # Stage 5 (y: 0.7 to 1.9)
    rect5 = patches.FancyBboxPatch((2.0, 0.7), 6.0, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor="#C0392B", facecolor="#FDEDEC", lw=1.5)
    ax.add_patch(rect5)
    ax.text(5, 1.4, "STAGE 5: RELIABILITY & CALIBRATION PIPELINE", ha="center", va="center", **font_title)
    ax.text(5, 1.0, "PAV Isotonic Calibration (100% ECE Red.) • MSP OOD Detection\nEMA Temporal Prediction Stabilizer • Reliability Score (R) Calculation", ha="center", va="center", **font_body)
    ax.annotate("", xy=(5, -0.9), xytext=(5, 0.7), arrowprops=arrow_params)

    # Adjust vertical space to include Stage 6 at the very bottom
    # We will close this figure and create a perfect final figure with height 20 units!
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 13.2), dpi=300)
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(-1.8, 18.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Re-draw everything with y-offsets
    # Stage 0: 16.6 to 17.8
    rect0 = patches.FancyBboxPatch((2.0, 16.6), 6.0, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor="#2980B9", facecolor="#EBF5FB", lw=1.5)
    ax.add_patch(rect0)
    ax.text(5, 17.3, "STAGE 0: INGESTION & GATEWAY", ha="center", va="center", **font_title)
    ax.text(5, 16.9, "FastAPI Endpoint • MIME Validation • 10MB Limit Check", ha="center", va="center", **font_body)
    ax.annotate("", xy=(5, 14.8), xytext=(5, 16.6), arrowprops=arrow_params)

    # Stage 1: 13.0 to 14.8
    rect1 = patches.FancyBboxPatch((2.0, 13.0), 6.0, 1.8, boxstyle="round,pad=0.1",
                                   edgecolor="#D35400", facecolor="#FDF2E9", lw=1.5)
    ax.add_patch(rect1)
    ax.text(5, 14.1, "STAGE 1: IMAGE QUALITY ASSESSMENT (IQA)", ha="center", va="center", **font_title)
    ax.text(5, 13.5, "7-Dimension IQI Gate Check (Blur, Noise, Contrast, etc.)\nRejects if IQI < 0.55 with retake guidance", ha="center", va="center", **font_body)
    ax.annotate("", xy=(9.0, 13.9), xytext=(8.0, 13.9), arrowprops=arrow_reject)
    rect_reject = patches.FancyBboxPatch((9.0, 13.0), 1.8, 1.8, boxstyle="round,pad=0.1",
                                         edgecolor="#C0392B", facecolor="#FDEDEC", lw=1.5)
    ax.add_patch(rect_reject)
    ax.text(9.9, 13.9, "REJECT\nHTTP 422", ha="center", va="center", color="#C0392B", fontsize=8, fontweight="bold")
    ax.annotate("", xy=(5, 11.0), xytext=(5, 13.0), arrowprops=arrow_params)

    # Stage 2: 9.0 to 11.0
    rect2 = patches.FancyBboxPatch((2.0, 9.0), 6.0, 2.0, boxstyle="round,pad=0.1",
                                   edgecolor="#8E44AD", facecolor="#F5EEF8", lw=1.5)
    ax.add_patch(rect2)
    ax.text(5, 10.3, "STAGE 2: ADVANCED PREPROCESSING PIPELINE", ha="center", va="center", **font_title)
    ax.text(5, 9.5, "• Non-Local Means Denoising (CIE LAB space)\n• Dark Channel Prior Dehazing (transmission mapping)\n• CLAHE Contrast Normalization & Gray World White Balance\n• Multi-Index Veg Enhancement & Shadow Inpainting",
            ha="left", va="center", **font_body)
    ax.annotate("", xy=(5, 7.2), xytext=(5, 9.0), arrowprops=arrow_params)

    # Stage 3: 3.0 to 7.2
    rect3 = patches.FancyBboxPatch((0.2, 3.0), 9.6, 4.2, boxstyle="round,pad=0.1",
                                   edgecolor="#16A085", facecolor="#E8F8F5", lw=1.5, ls="--")
    ax.add_patch(rect3)
    ax.text(5, 6.8, "STAGE 3: SPATIO-SPECTRAL-TEMPORAL (SST) FEATURE EXTRACTORS", ha="center", va="center", **font_title)

    rect3a = patches.FancyBboxPatch((0.5, 3.3), 2.8, 3.0, boxstyle="round,pad=0.1",
                                    edgecolor="#27AE60", facecolor="#E8F8F5", lw=1.2)
    ax.add_patch(rect3a)
    ax.text(1.9, 5.8, "NGVCNet\n(Spectral Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#1E8449")
    ax.text(1.9, 4.2, "128-bin NGVC\nHistogram\n+\n1D-CNN Blocks\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    rect3b = patches.FancyBboxPatch((3.6, 3.3), 2.8, 3.0, boxstyle="round,pad=0.1",
                                    edgecolor="#2980B9", facecolor="#EBF5FB", lw=1.2)
    ax.add_patch(rect3b)
    ax.text(5.0, 5.8, "SpatialGraphNet\n(Spatial Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#21618C")
    ax.text(5.0, 4.2, "Centroid Adjacency\nGraph (1.5x stride)\n+\n2-Layer GraphSAGE\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    rect3c = patches.FancyBboxPatch((6.7, 3.3), 2.8, 3.0, boxstyle="round,pad=0.1",
                                    edgecolor="#D35400", facecolor="#FDF2E9", lw=1.2)
    ax.add_patch(rect3c)
    ax.text(8.1, 5.8, "TemporalProgNet\n(Temporal Branch)", ha="center", va="center", fontsize=8, fontweight="bold", color="#A04000")
    ax.text(8.1, 4.2, "5-Step Historical\nEmbedding Sequence\n+\n2-Layer BiLSTM\n(64-dim embedding)", ha="center", va="center", fontsize=7.5)

    ax.annotate("", xy=(5, 1.3), xytext=(1.9, 3.3), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 1.3), xytext=(5.0, 3.3), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 1.3), xytext=(8.1, 3.3), arrowprops=arrow_params)

    # Stage 4: 1.3 to 2.5
    rect4 = patches.FancyBboxPatch((2.0, 1.3), 6.0, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor="#16A085", facecolor="#E8F8F5", lw=1.5)
    ax.add_patch(rect4)
    ax.text(5, 2.1, "STAGE 4: AGRIVISIONENSEMBLE META-LEARNER", ha="center", va="center", **font_title)
    ax.text(5, 1.7, "Concatenated Embeddings (192-dim) • 3-Layer MLP\nDual Heads: Classification (Softmax) & Severity (Sigmoid)", ha="center", va="center", **font_body)
    ax.annotate("", xy=(5, -0.4), xytext=(5, 1.3), arrowprops=arrow_params)

    # Stage 5: -0.4 to -1.6
    rect5 = patches.FancyBboxPatch((2.0, -0.4), 6.0, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor="#C0392B", facecolor="#FDEDEC", lw=1.5)
    ax.add_patch(rect5)
    ax.text(5, 0.4, "STAGE 5: RELIABILITY & CALIBRATION PIPELINE", ha="center", va="center", **font_title)
    ax.text(5, 0.0, "PAV Isotonic Calibration (100% ECE Red.) • MSP OOD Detection\nEMA Temporal Prediction Stabilizer • Reliability Score (R) Calculation", ha="center", va="center", **font_body)
    ax.annotate("", xy=(5, -2.2), xytext=(5, -0.4), arrowprops=arrow_params)

    # Stage 6: -2.2 to -3.4
    rect6 = patches.FancyBboxPatch((2.0, -2.1), 6.0, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor="#7F8C8D", facecolor="#F2F3F4", lw=1.5)
    ax.add_patch(rect6)
    ax.text(5, -1.3, "STAGE 6: RESPONSE ASSEMBLY & MONITORING", ha="center", va="center", **font_title)
    ax.text(5, -1.7, "PostgreSQL Persistence • Prometheus & Grafana Metrics\nLocal AI Field Agronomist Consultation Advisory Route", ha="center", va="center", **font_body)

    # Ensure output directories exist
    os.makedirs("model_weights/publication_results/figures", exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("model_weights/publication_results/figures/fig1_architecture.png", bbox_inches="tight")
    plt.savefig("fig1_architecture.png", bbox_inches="tight")
    plt.close()
    print("Successfully generated and saved fig1_architecture.png")

if __name__ == "__main__":
    draw_architecture()
