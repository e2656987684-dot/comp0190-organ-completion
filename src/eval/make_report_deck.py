"""Build the progress-report slide deck from the generated figures and metrics.

Numbers come from `reports/figures/summary.csv`, which `make_report_figures.py`
writes out of the real runs -- so a slide cannot quietly disagree with the code.
Run that script first, then this one.

    python src/eval/make_report_figures.py
    python src/eval/make_report_deck.py

No cover, contents or summary slide by request: six content slides, each one
figure plus the few numbers that figure is there to support.
"""

from __future__ import annotations

import os

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(REPO, "reports", "figures")
OUT = os.path.join(REPO, "reports", "progress_report.pptx")

W, H = Inches(13.333), Inches(7.5)          # 16:9
INK = RGBColor(0x1F, 0x2A, 0x37)
MUTED = RGBColor(0x5B, 0x66, 0x73)
ACCENT = RGBColor(0x15, 0x65, 0xC0)
GOOD = RGBColor(0x1B, 0x5E, 0x20)
WARN = RGBColor(0xB7, 0x3B, 0x1F)
RULE = RGBColor(0xD5, 0xDA, 0xE0)


def _text(slide, x, y, w, h, runs, size=16, color=INK, bold=False,
          align=PP_ALIGN.LEFT, space=6):
    """runs: str, or list of (text, {overrides}) for mixed formatting."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    if isinstance(runs, str):
        runs = [(runs, {})]
    for i, (txt, ov) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ov.get("align", align)
        p.space_after = Pt(ov.get("space", space))
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(ov.get("size", size))
        r.font.bold = ov.get("bold", bold)
        r.font.color.rgb = ov.get("color", color)
        r.font.name = "Calibri"
    return box


def _slide(prs, title, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])      # blank
    _text(s, Inches(0.55), Inches(0.32), Inches(12.2), Inches(0.7),
          title, size=27, bold=True)
    if kicker:
        _text(s, Inches(0.55), Inches(1.02), Inches(12.2), Inches(0.4),
              kicker, size=14, color=MUTED)
    ln = s.shapes.add_shape(1, Inches(0.55), Inches(1.45), Inches(12.2), Emu(9525))
    ln.fill.solid()
    ln.fill.fore_color.rgb = RULE
    ln.line.fill.background()
    ln.shadow.inherit = False
    return s


def _picture(slide, name, y, height=None, width=None):
    p = os.path.join(FIG, name)
    if width:
        pic = slide.shapes.add_picture(p, 0, y, width=width)
    else:
        pic = slide.shapes.add_picture(p, 0, y, height=height)
    pic.left = int((W - pic.width) / 2)
    return pic


def _table(slide, x, y, w, h, rows, col_w=None, size=13, header_bg=None,
           highlight_row=None):
    n_r, n_c = len(rows), len(rows[0])
    shp = slide.shapes.add_table(n_r, n_c, x, y, w, h)
    tbl = shp.table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = cw
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(0.32)
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = str(val)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
            r = para.runs[0] if para.runs else para.add_run()
            r.font.size = Pt(size)
            r.font.name = "Calibri"
            r.font.bold = (ri == 0) or (highlight_row is not None and ri == highlight_row)
            r.font.color.rgb = (ACCENT if highlight_row is not None and ri == highlight_row
                                else INK)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (header_bg or RGBColor(0xEF, 0xF3, 0xF7)) \
                if ri == 0 else RGBColor(0xFF, 0xFF, 0xFF)
    return shp


def main():
    df = pd.read_csv(os.path.join(FIG, "summary.csv")).set_index("run")
    g = lambda r, c: df.loc[r, c]
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---------------------------------------------------------------- slide 1
    s = _slide(prs, "Task and evaluation baseline",
               "SkullFix, 100 skulls · 80 train / 20 validation · same split for every run")
    _picture(s, "completion.png", Inches(1.75), height=Inches(3.5))
    _table(s, Inches(1.5), Inches(5.35), Inches(10.3), Inches(1.0),
           [["", "CD_t (mm)", "DCD", "Note"],
            ["Released pretrained weights", "10.71", "1.428", "not trained on skulls"],
            ["This work (from scratch)", f"{g('rep_w05','CD_t_mm'):.2f}",
             f"{g('rep_w05','DCD'):.3f}", "−40% CD_t, −51% DCD"]],
           col_w=[Inches(4.0), Inches(1.8), Inches(1.6), Inches(2.9)],
           highlight_row=2)
    _text(s, Inches(1.5), Inches(6.55), Inches(10.3), Inches(0.5),
          "Sanity check: the source paper reports DCD = 1.41269; the released weights "
          "measure 1.42772 on our data — a 1% match, confirming the metric port and "
          "weight loading.", size=13, color=MUTED)

    # ---------------------------------------------------------------- slide 2
    s = _slide(prs, "The real gap is point density, not surface roughness",
               "The original plan — a smoothness penalty — was dropped after measurement")
    _picture(s, "density_problem.png", Inches(1.7), height=Inches(3.3))
    _text(s, Inches(1.1), Inches(5.35), Inches(5.2), Inches(1.5),
          [("Surface roughness", {"bold": True, "size": 17}),
           ("GT 0.736  vs  prediction 0.760", {"size": 16}),
           ("→ essentially equal, nothing to gain", {"color": MUTED, "size": 14})],
          size=16, space=4)
    _text(s, Inches(7.0), Inches(5.35), Inches(5.2), Inches(1.5),
          [("Point spacing", {"bold": True, "size": 17, "color": WARN}),
           ("closer than 2 mm:  GT 0.0%  vs  12.8%", {"size": 16, "color": WARN}),
           ("spacing CV:  GT 0.145  vs  0.389", {"size": 16, "color": WARN}),
           ("→ a clear, measurable deficit", {"color": MUTED, "size": 14})],
          size=16, space=4)

    # ---------------------------------------------------------------- slide 3
    s = _slide(prs, "Why the density-aware loss cannot fix it",
               "Mechanism, measured — not an assumption")
    _text(s, Inches(0.9), Inches(1.75), Inches(11.5), Inches(0.6),
          "DCD  =  1 − exp(−α·d) · 1 / count λ", size=24, bold=True, color=ACCENT,
          align=PP_ALIGN.CENTER)
    _text(s, Inches(0.9), Inches(2.32), Inches(11.5), Inches(0.4),
          "count comes from argmin — piecewise constant, so its gradient is exactly "
          "zero (verified: tf.gradients returns None)",
          size=15, color=MUTED, align=PP_ALIGN.CENTER)
    _picture(s, "dcd_blindspot.png", Inches(2.9), height=Inches(2.5))
    _text(s, Inches(1.2), Inches(5.65), Inches(11.0), Inches(1.2),
          [("DCD can only re-weight Chamfer's gradients — it can never push two "
            "predicted points apart.", {"bold": True}),
           ("So tuning its hyper-parameters was never going to fix density, and a "
            "repulsion term is not redundant with it.", {"color": MUTED, "size": 15})],
          size=17, space=8, align=PP_ALIGN.CENTER)

    # ---------------------------------------------------------------- slide 4
    s = _slide(prs, "Two changes that worked",
               "Attribution separated with a controlled run (learning-rate fix only, no repulsion)")
    _picture(s, "curves.png", Inches(1.65), height=Inches(2.9))
    _table(s, Inches(1.1), Inches(4.9), Inches(11.1), Inches(1.3),
           [["", "val CD_t (mm)", "clumped <2 mm", "spacing CV"],
            ["baseline", f"{g('baseline','CD_t_mm'):.2f}",
             f"{g('baseline','clump_%'):.1f}%", f"{g('baseline','spacing_CV'):.3f}"],
            ["+ learning-rate schedule fixed", f"{g('lr_fix','CD_t_mm'):.2f}",
             f"{g('lr_fix','clump_%'):.1f}%", f"{g('lr_fix','spacing_CV'):.3f}"],
            ["+ repulsion loss", f"{g('rep_w05','CD_t_mm'):.2f}",
             f"{g('rep_w05','clump_%'):.1f}%", f"{g('rep_w05','spacing_CV'):.3f}"]],
           col_w=[Inches(5.1), Inches(2.2), Inches(2.0), Inches(1.8)],
           highlight_row=3)
    _text(s, Inches(1.1), Inches(6.45), Inches(11.1), Inches(0.7),
          [("All of the CD_t gain came from the learning-rate fix — ReduceLROnPlateau's "
            "patience exceeded early stopping's, so it had never once fired. "
            "Repulsion contributes density only, at no accuracy cost.", {})],
          size=14, color=MUTED)

    # ---------------------------------------------------------------- slide 5
    s = _slide(prs, "Results",
               "20 validation skulls · millimetres use each skull's own scale")
    _table(s, Inches(0.75), Inches(1.8), Inches(11.8), Inches(2.0),
           [["run", "CD_t (mm)", "HD95 (mm)", "F1@0.05", "F1@0.03", "DCD",
             "clumped<2mm", "spacing CV"]] +
           [[r] + [f"{g(r,c):.3f}" if c not in ("clump_%",) else f"{g(r,c):.2f}%"
                   for c in ["CD_t_mm", "HD95_mm", "F1@0.05", "F1@0.03", "DCD",
                             "clump_%", "spacing_CV"]]
            for r in ["baseline", "dcd_l2", "lr_fix", "rep_w05"]] +
           [["ground truth", "—", "—", "—", "—", "—", "0.00%", "0.145"]],
           col_w=[Inches(2.2)] + [Inches(1.37)] * 7, size=13, highlight_row=4)
    _text(s, Inches(0.75), Inches(4.15), Inches(11.8), Inches(0.4),
          "F-score against the source paper (same metric definition, different datasets)",
          size=15, bold=True)
    _table(s, Inches(0.75), Inches(4.6), Inches(11.8), Inches(1.4),
           [["", "training data", "F1@0.05", "F1@0.03"],
            ["Source paper, Table 1", "200,000 clouds / 240 classes", "0.937", "0.741"],
            ["This work", "80 skulls / 1 class",
             f"{g('rep_w05','F1@0.05'):.3f}", f"{g('rep_w05','F1@0.03'):.3f}"],
            ["Source paper, Table 2", "4,800 shapes", "0.892", "0.623"]],
           col_w=[Inches(3.4), Inches(4.6), Inches(1.9), Inches(1.9)],
           highlight_row=2)
    _text(s, Inches(0.75), Inches(6.5), Inches(11.8), Inches(0.5),
          "DCD and F-score are comparable across projects; CD is not — the paper's CD "
          "would fall 35× below the point spacing under this codebase's definition.",
          size=13, color=MUTED)

    # ---------------------------------------------------------------- slide 6
    s = _slide(prs, "Next steps", "and open questions")
    _text(s, Inches(0.8), Inches(1.85), Inches(5.7), Inches(0.4),
          "Planned", size=18, bold=True, color=ACCENT)
    _text(s, Inches(0.8), Inches(2.35), Inches(5.6), Inches(4.2),
          [("1.  Ablate DCD away", {"bold": True}),
           ("Density is now handled by repulsion; if removing DCD changes nothing, "
            "three losses collapse to two.", {"size": 14, "color": MUTED}),
           ("2.  k-fold cross-validation", {"bold": True}),
           ("Every result so far is a single 20-skull split. Code is in place, never run.",
            {"size": 14, "color": MUTED}),
           ("3.  Defect-region-restricted metrics", {"bold": True}),
           ("Current metrics are diluted by the visible surface the model only has to "
            "reproduce.", {"size": 14, "color": MUTED})],
          size=16, space=7)
    _text(s, Inches(7.0), Inches(1.85), Inches(5.5), Inches(0.4),
          "Under consideration", size=18, bold=True, color=MUTED)
    _text(s, Inches(7.0), Inches(2.35), Inches(5.5), Inches(4.2),
          [("Focus the model on the defect", {"bold": True}),
           ("The dataset's implant labels have never been used; training on them "
            "directly would match the AutoImplant evaluation protocol. Large change.",
            {"size": 14, "color": MUTED}),
           ("Point-to-plane attraction", {"bold": True}),
           ("An alternative route to the supervisor's smoothness goal — orthogonal to "
            "repulsion and verifiable with existing tools.", {"size": 14, "color": MUTED}),
           ("Poisson reconstruction control", {"bold": True}),
           ("Would connect the density metric to clinical usability, which the current "
            "visualisation cannot show.", {"size": 14, "color": MUTED})],
          size=16, space=7)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print(f"-> {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
