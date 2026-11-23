"""
Generate high-quality AgriPulse 2.0 architecture diagram for IEEE paper.
Fixes from reviewer feedback:
  - Removed duplicate Stage 0 block
  - Fixed typo: "not a smart rote" → correct label
  - Single clean pipeline flow
  - High DPI (600) for print quality
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


def draw_v2_architecture():
    fig, ax = plt.subplots(figsize=(7.5, 12.5), dpi=600)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.5, 18.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Styling
    font_stage = dict(fontsize=9, fontweight="bold", family="sans-serif")
    font_body = dict(fontsize=7.5, family="sans-serif")
    font_small = dict(fontsize=7, family="sans-serif")
    arrow_params = dict(arrowstyle="-|>", lw=1.5, color="#2C3E50", mutation_scale=14)
    arrow_reject = dict(arrowstyle="-|>", lw=1.5, color="#C0392B", mutation_scale=14)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 0: Ingestion & Gateway (y: 17.0–18.0)
    # ═══════════════════════════════════════════════════════════════════
    rect0 = patches.FancyBboxPatch((1.8, 17.0), 6.4, 1.0,
                                   boxstyle="round,pad=0.15",
                                   edgecolor="#2980B9", facecolor="#EBF5FB", lw=1.5)
    ax.add_patch(rect0)
    ax.text(5, 17.65, "Stage 0: Ingestion & Gateway", ha="center", va="center", **font_stage)
    ax.text(5, 17.25, "FastAPI Endpoint · MIME Validation · File Size Check",
            ha="center", va="center", **font_body)

    # Arrow 0 → 1
    ax.annotate("", xy=(5, 16.0), xytext=(5, 17.0), arrowprops=arrow_params)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: Image Quality Assessment (y: 14.8–16.0)
    # ═══════════════════════════════════════════════════════════════════
    rect1 = patches.FancyBboxPatch((1.8, 14.8), 6.4, 1.2,
                                   boxstyle="round,pad=0.15",
                                   edgecolor="#D35400", facecolor="#FDF2E9", lw=1.5)
    ax.add_patch(rect1)
    ax.text(5, 15.55, "Stage 1: Image Quality Assessment (IQA)", ha="center", va="center", **font_stage)
    ax.text(5, 15.1, "7-Dimension IQI Score (Blur, Noise, Contrast, Resolution, ...)\nRejects if IQI < 0.55 with farmer-facing retake guidance",
            ha="center", va="center", **font_small)

    # Reject branch
    ax.annotate("", xy=(9.0, 15.4), xytext=(8.2, 15.4), arrowprops=arrow_reject)
    rect_rej = patches.FancyBboxPatch((9.0, 15.0), 1.3, 0.8,
                                      boxstyle="round,pad=0.1",
                                      edgecolor="#C0392B", facecolor="#FDEDEC", lw=1.3)
    ax.add_patch(rect_rej)
    ax.text(9.65, 15.4, "REJECT", ha="center", va="center",
            color="#C0392B", fontsize=8, fontweight="bold")

    # Arrow 1 → 2
    ax.annotate("", xy=(5, 13.6), xytext=(5, 14.8), arrowprops=arrow_params)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 2: Preprocessing Pipeline (y: 12.8–13.6)
    # ═══════════════════════════════════════════════════════════════════
    rect2 = patches.FancyBboxPatch((1.8, 12.8), 6.4, 0.8,
                                   boxstyle="round,pad=0.15",
                                   edgecolor="#8E44AD", facecolor="#F5EEF8", lw=1.5)
    ax.add_patch(rect2)
    ax.text(5, 13.35, "Stage 2: Preprocessing Pipeline", ha="center", va="center", **font_stage)
    ax.text(5, 13.0, "Denoise · Dehaze · CLAHE · White Balance · Veg. Enhancement",
            ha="center", va="center", **font_small)

    # Arrow 2 → 3
    ax.annotate("", xy=(5, 12.0), xytext=(5, 12.8), arrowprops=arrow_params)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 3: DINOv2 Feature Extraction + Adaptive Grid (y: 10.6–12.0)
    # ═══════════════════════════════════════════════════════════════════
    rect3_bg = patches.FancyBboxPatch((1.0, 10.6), 8.0, 1.4,
                                      boxstyle="round,pad=0.15",
                                      edgecolor="#F39C12", facecolor="#FEF9E7", lw=1.5)
    ax.add_patch(rect3_bg)
    ax.text(5, 11.65, "Stage 3: DINOv2 Feature Extraction", ha="center", va="center", **font_stage)

    # Sub-blocks: Frozen ViT and Adaptive Grid
    rect3a = patches.FancyBboxPatch((1.3, 10.75), 3.2, 0.7,
                                    boxstyle="round,pad=0.08",
                                    edgecolor="#E67E22", facecolor="#FDEBD0", lw=1.0)
    ax.add_patch(rect3a)
    ax.text(2.9, 11.1, "Frozen ViT-B/14\n(86M params)", ha="center", va="center", **font_small)

    rect3b = patches.FancyBboxPatch((5.5, 10.75), 3.2, 0.7,
                                    boxstyle="round,pad=0.08",
                                    edgecolor="#E67E22", facecolor="#FDEBD0", lw=1.0)
    ax.add_patch(rect3b)
    ax.text(7.1, 11.1, "Adaptive Grid\n(14px stride, N patches)", ha="center", va="center", **font_small)

    # Arrow: Split into 3 branches
    ax.annotate("", xy=(2.5, 9.4), xytext=(5, 10.6), arrowprops=arrow_params)
    ax.annotate("", xy=(5.0, 9.4), xytext=(5, 10.6), arrowprops=arrow_params)
    ax.annotate("", xy=(7.5, 9.4), xytext=(5, 10.6), arrowprops=arrow_params)

    # ═══════════════════════════════════════════════════════════════════
    # Three Branches (y: 7.8–9.4)
    # ═══════════════════════════════════════════════════════════════════
    # Branch A: Graph Transformer (Spatial)
    rect_ba = patches.FancyBboxPatch((1.0, 7.8), 3.0, 1.6,
                                     boxstyle="round,pad=0.12",
                                     edgecolor="#27AE60", facecolor="#EAFAF1", lw=1.3)
    ax.add_patch(rect_ba)
    ax.text(2.5, 9.0, "Branch A:", ha="center", va="center",
            fontsize=7, fontweight="bold", color="#1E8449")
    ax.text(2.5, 8.7, "Graph Transformer", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#1E8449")
    ax.text(2.5, 8.2, "Adjacency-masked\nself-attention (4 heads)\n→ Spatial embedding\n$\\mathbf{z}_{spat} \\in \\mathbb{R}^{64}$",
            ha="center", va="center", **font_small)

    # Branch B: Temporal Transformer
    rect_bb = patches.FancyBboxPatch((4.0, 7.8), 2.0, 1.6,
                                     boxstyle="round,pad=0.12",
                                     edgecolor="#2980B9", facecolor="#EBF5FB", lw=1.3)
    ax.add_patch(rect_bb)
    ax.text(5.0, 9.0, "Branch B:", ha="center", va="center",
            fontsize=7, fontweight="bold", color="#21618C")
    ax.text(5.0, 8.7, "Temporal\nTransformer", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#21618C")
    ax.text(5.0, 8.15, "CLS token\n+ Progression\n$\\mathbf{z}_{temp} \\in \\mathbb{R}^{64}$",
            ha="center", va="center", **font_small)

    # Branch C: Environmental Data (optional)
    rect_bc = patches.FancyBboxPatch((6.2, 7.8), 3.0, 1.6,
                                     boxstyle="round,pad=0.12",
                                     edgecolor="#95A5A6", facecolor="#F2F4F4", lw=1.0, ls="--")
    ax.add_patch(rect_bc)
    ax.text(7.7, 9.0, "Branch C (optional):", ha="center", va="center",
            fontsize=7, fontweight="bold", color="#566573")
    ax.text(7.7, 8.55, "Environmental", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#566573")
    ax.text(7.7, 8.15, "Temp, Humidity,\nRainfall → Linear\n$\\mathbf{z}_{env} \\in \\mathbb{R}^{64}$",
            ha="center", va="center", **font_small)

    # Arrows: Branches → Stage 4
    ax.annotate("", xy=(5, 6.6), xytext=(2.5, 7.8), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 6.6), xytext=(5.0, 7.8), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 6.6), xytext=(7.7, 7.8),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#95A5A6", mutation_scale=12, ls="--"))

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 4: Cross-Attention Fusion (y: 5.2–6.6)
    # ═══════════════════════════════════════════════════════════════════
    rect4 = patches.FancyBboxPatch((1.8, 5.2), 6.4, 1.4,
                                   boxstyle="round,pad=0.15",
                                   edgecolor="#E74C3C", facecolor="#FDEDEC", lw=1.5)
    ax.add_patch(rect4)
    ax.text(5, 6.25, "Stage 4: Cross-Attention Fusion", ha="center", va="center", **font_stage)
    ax.text(5, 5.8, "Query: Visual embedding ($\\mathbf{z}_{vis}$)\n"
                     "Key/Value: Spatial + Temporal (+ Environmental) context",
            ha="center", va="center", **font_small)
    ax.text(5, 5.4, "Multi-head cross-attention → FFN → Residual + LayerNorm",
            ha="center", va="center", **font_small, color="#7B241C")

    # Arrow 4 → Split
    ax.annotate("", xy=(3.5, 4.4), xytext=(5, 5.2), arrowprops=arrow_params)
    ax.annotate("", xy=(6.5, 4.4), xytext=(5, 5.2), arrowprops=arrow_params)

    # ═══════════════════════════════════════════════════════════════════
    # Dual Heads (y: 3.6–4.4)
    # ═══════════════════════════════════════════════════════════════════
    rect_cls = patches.FancyBboxPatch((1.8, 3.6), 3.0, 0.8,
                                      boxstyle="round,pad=0.1",
                                      edgecolor="#2C3E50", facecolor="#D5D8DC", lw=1.2)
    ax.add_patch(rect_cls)
    ax.text(3.3, 4.0, "Classification Head\nSoftmax → $\\hat{\\mathbf{y}} \\in \\mathbb{R}^C$",
            ha="center", va="center", **font_small)

    rect_sev = patches.FancyBboxPatch((5.2, 3.6), 3.0, 0.8,
                                      boxstyle="round,pad=0.1",
                                      edgecolor="#2C3E50", facecolor="#D5D8DC", lw=1.2)
    ax.add_patch(rect_sev)
    ax.text(6.7, 4.0, "Severity Head\nSigmoid → $\\hat{S} \\in [0,100]$",
            ha="center", va="center", **font_small)

    # Arrow Dual Heads → Stage 5
    ax.annotate("", xy=(5, 2.6), xytext=(3.5, 3.6), arrowprops=arrow_params)
    ax.annotate("", xy=(5, 2.6), xytext=(6.5, 3.6), arrowprops=arrow_params)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 5: Reliability & Calibration (y: 1.2–2.6)
    # ═══════════════════════════════════════════════════════════════════
    rect5 = patches.FancyBboxPatch((1.0, 1.2), 8.0, 1.4,
                                   boxstyle="round,pad=0.15",
                                   edgecolor="#2471A3", facecolor="#D4E6F1", lw=1.5)
    ax.add_patch(rect5)
    ax.text(5, 2.25, "Stage 5: Reliability & Calibration Pipeline", ha="center", va="center", **font_stage)
    ax.text(5, 1.65, "Temperature Scaling → Isotonic Regression (PAV)\n"
                     "→ OOD Detection (Energy + Mahalanobis) → EMA Stabilization\n"
                     "→ Composite Reliability Score $R = 0.25 \\cdot IQI + 0.35 \\cdot \\min(1, c/0.8) + ...$",
            ha="center", va="center", **font_small)

    # Arrow 5 → 6
    ax.annotate("", xy=(5, 0.2), xytext=(5, 1.2), arrowprops=arrow_params)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 6: Response Assembly (y: -1.0–0.2)
    # ═══════════════════════════════════════════════════════════════════
    rect6 = patches.FancyBboxPatch((1.8, -1.0), 6.4, 1.2,
                                   boxstyle="round,pad=0.15",
                                   edgecolor="#1B4F72", facecolor="#1B4F72", lw=1.5)
    ax.add_patch(rect6)
    ax.text(5, -0.2, "Stage 6: Response Assembly & Monitoring",
            ha="center", va="center", color="white", **font_stage)
    ax.text(5, -0.65, "Disease Class + Severity + Reliability Score + Advisory\n"
                      "PostgreSQL · Prometheus · Grafana",
            ha="center", va="center", color="#D6EAF8", **font_small)

    # ═══════════════════════════════════════════════════════════════════
    # Side annotation: Graceful degradation
    # ═══════════════════════════════════════════════════════════════════
    ax.text(-0.3, 8.6, "Graceful\nDegradation:\nZero-init if\nno history",
            ha="center", va="center", fontsize=6.5, fontstyle="italic",
            color="#7F8C8D",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDFEFE",
                      edgecolor="#BDC3C7", lw=0.8))

    # Title annotation
    ax.text(5, 18.3, "AgriPulse 2.0 — Seven-Stage Inference Pipeline",
            ha="center", va="center", fontsize=11, fontweight="bold",
            family="sans-serif", color="#1B2631")

    plt.tight_layout()

    # Save at multiple locations
    out_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(out_dir, "..", "..")

    out_path = os.path.join(project_root, "fig1_v2_architecture.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=600, facecolor="white")
    print(f"Saved: {out_path}")

    # Also save as PDF vector for LaTeX
    out_pdf = os.path.join(project_root, "fig1_v2_architecture.pdf")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_pdf}")

    plt.close()
    print("Successfully generated v2 architecture diagram (600 DPI + PDF)")


if __name__ == "__main__":
    draw_v2_architecture()
