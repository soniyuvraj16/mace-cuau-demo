"""Revise the partner's deck around the actual demo and add backup slides.

Input  MACE_Agentic_Discovery.pptx   (kept untouched)
Output MACE_Agentic_Discovery_v2.pptx
"""

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "MACE_Agentic_Discovery.pptx"
OUT = ROOT / "MACE_Agentic_Discovery_v2.pptx"

MAROON, ROSE, PINK = "500000", "B87878", "E3C4C4"
DARK, GREY, LIGHT = "1E293B", "64748B", "94A3B8"
CARD, WHITE = "F5F1F1", "FFFFFF"
CU, AU = "B87333", "C9A227"


def rgb(h):
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- text helpers
def set_lines(shape, lines):
    """Replace paragraph texts while keeping each paragraph's first-run formatting."""
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    for i, para in enumerate(paras):
        if i < len(lines):
            runs = para.runs
            if runs:
                runs[0].text = lines[i]
                for r in runs[1:]:
                    r._r.getparent().remove(r._r)
            else:
                para.text = lines[i]
        else:
            para._p.getparent().remove(para._p)
    for extra in lines[len(paras):]:
        p = tf.add_paragraph()
        p.text = extra


def shape_by_text(slide, startswith):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith(startswith):
            return sh
    raise KeyError(startswith)


def remove(shape):
    shape._element.getparent().remove(shape._element)


NOTES_BODY_TEMPLATE = None  # set in main(): the body placeholder of an existing notes slide


def set_notes(slide, text):
    import copy

    ns = slide.notes_slide
    if ns.notes_text_frame is None and NOTES_BODY_TEMPLATE is not None:
        # the deck's notes master has no body placeholder, so new notes slides get none; borrow one
        ns.shapes._spTree.append(copy.deepcopy(NOTES_BODY_TEMPLATE))
    tf = ns.notes_text_frame
    if tf is not None:
        tf.text = text


# ---------------------------------------------------------------- drawing helpers for new slides
def text(slide, x, y, w, h, s, size=14, bold=False, color=DARK, align=PP_ALIGN.LEFT, font="Arial", italic=False, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = rgb(color)
    return tb


def box(slide, x, y, w, h, fill=CARD, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = rgb(fill)
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = rgb(line)
        sp.line.width = Pt(1)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sp.adjustments[0] = 0.08
    sp.shadow.inherit = False
    return sp


def circle(slide, x, y, d, fill, label=None, size=14, color=WHITE, font="Arial"):
    c = box(slide, x, y, d, d, fill, MSO_SHAPE.OVAL)
    if label is not None:
        tf = c.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.color.rgb = rgb(color)
    return c


def line(slide, x1, y1, x2, y2, color=LIGHT, width=1.5, arrow=False):
    ln = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = rgb(color)
    ln.line.width = Pt(width)
    if arrow:
        from pptx.oxml.ns import qn
        lnEl = ln.line._get_or_add_ln()
        tail = lnEl.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        lnEl.append(tail)
    return ln


def header(slide, label, title, subtitle, num):
    text(slide, 0.6, 0.38, 9.0, 0.3, label, 11, True, ROSE)
    text(slide, 0.6, 0.62, 12.1, 0.7, title, 34, True, MAROON)
    text(slide, 0.6, 1.32, 12.1, 0.4, subtitle, 15, False, GREY, italic=True)
    text(slide, 12.2, 7.0, 0.6, 0.25, f"{num:02d}", 10, True, MAROON, PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, title, body, fill=CARD, title_color=DARK, body_color=GREY, title_size=16, body_size=12.5, icon=None):
    box(slide, x, y, w, h, fill)
    ty = y + 0.22
    if icon:
        circle(slide, x + 0.25, y + 0.22, 0.55, MAROON if fill != MAROON else WHITE, icon, 13, WHITE if fill != MAROON else MAROON, "Cambria Math")
        ty = y + 0.95
    text(slide, x + 0.25, ty, w - 0.5, 0.45, title, title_size, True, title_color)
    text(slide, x + 0.25, ty + 0.5, w - 0.5, h - (ty - y) - 0.6, body, body_size, False, body_color)


def footer(slide, s):
    text(slide, 0.6, 7.03, 10.5, 0.25, s, 9, False, LIGHT)


# ---------------------------------------------------------------- data from the run
def run_numbers():
    disc = json.loads((ROOT / "results" / "discovery.json").read_text())
    val = json.loads((ROOT / "results" / "validation.json").read_text())
    gens = [json.loads(l) for l in (ROOT / "results" / "progress.jsonl").read_text().splitlines() if l.strip()]
    gens = [g for g in gens if g["type"] == "generation"]
    pop_mean = [sum(p["ef"] for p in g["population"]) / len(g["population"]) for g in gens]
    best = [min(p["ef"] for p in g["population"]) for g in gens]
    return disc, val, pop_mean, best


def main():
    prs = Presentation(SRC)
    global NOTES_BODY_TEMPLATE
    for sh in prs.slides[0].notes_slide.shapes:
        if sh.is_placeholder and sh.has_text_frame and sh.text_frame.text.strip():
            NOTES_BODY_TEMPLATE = sh._element
            break
    disc, val, pop_mean, best = run_numbers()
    base, ft = val["foundation (MACE-MP-0 small)"], val["fine-tuned"]
    summ = disc["summary"]
    champ_ref = disc["champion"]["ref_formation"] * 1000

    # ---------------- slide 1: authors + timing in notes
    s1 = prs.slides[0]
    set_lines(shape_by_text(s1, "Joshua Jang"), ["Joshua Jang  ·  Yuvraj Soni  ·  Fall 2026"])
    set_notes(s1, "[0:00-0:15] Today: MACE, a machine-learned model of atomic forces, and why it is the kind of tool an AI agent needs "
                  "to do real materials discovery. Three minutes of slides, then a seven-minute live demo where an agent fine-tunes MACE "
                  "on our own DFT data, searches for the most stable copper-gold arrangement, and checks its answer against DFT.")

    # ---------------- slide 5: evolution slide around our run
    s5 = prs.slides[4]
    set_lines(shape_by_text(s5, "Generate, score"), ["Generate, score, select, repeat — then verify the winner with the slow simulator (DFT)"])
    set_lines(shape_by_text(s5, "Mutate"), ["Mutate", "crossover + bit flips"])
    set_lines(shape_by_text(s5, "Score"), ["Score", "fine-tuned MACE energy"])
    set_lines(shape_by_text(s5, "GENOME"), ["GENOME: 8 SITES, 2×1×1 FCC CELL  (shown: 00010001 = L1₂ Cu₃Au, the ground state)"])
    strip = sorted([sh for sh in s5.shapes if sh.shape_type == 1 and abs(sh.width / 914400 - 0.15) < 0.01 and abs(sh.height / 914400 - 0.38) < 0.01], key=lambda sh: sh.left)
    bits = "00010001"
    for k, sh in enumerate(strip):
        if k < 8:
            sh.left = Inches(0.6 + k * 0.26)
            sh.width = Inches(0.22)
            sh.fill.solid()
            sh.fill.fore_color.rgb = rgb(AU if bits[k] == "1" else CU)
        else:
            remove(sh)
    legend = shape_by_text(s5, "■ Cu")
    set_lines(legend, ["■ Cu    ■ Au     bit string read left to right, site by site"])
    chart = next(sh for sh in s5.shapes if sh.has_chart).chart
    cd = CategoryChartData()
    cd.categories = [str(i) for i in range(len(best))]
    cd.add_series("population mean (MACE)", [round(v, 1) for v in pop_mean])
    cd.add_series("GA best (MACE)", [round(v, 1) for v in best])
    cd.add_series("DFT energy of the GA's pick", [round(champ_ref, 1)] * len(best))
    chart.replace_data(cd)
    chart.value_axis.minimum_scale = -60
    chart.value_axis.maximum_scale = 10
    chart.value_axis.major_unit = 10
    if chart.has_title:
        runs = [r for p in chart.chart_title.text_frame.paragraphs for r in p.runs]
        if runs:
            runs[0].text = f"GA best {best[-1]:.0f} (MACE)  ·  DFT says {champ_ref:.0f} meV/atom for the same structure"
            for r in runs[1:]:
                r._r.getparent().remove(r._r)
    set_lines(shape_by_text(s5, "Real run"), [f"Real run from today's demo: 8-site Cu/Au cell, MACE-MP-0 small fine-tuned on 135 DFT labels, "
                                              f"{summ['ga_unique_evaluations']} network calls at ~{summ['ga_ms_per_evaluation']:.0f} ms each; "
                                              f"the ~{abs(best[-1] - champ_ref):.0f} meV/atom gap is the network's offset — DFT confirms the ranking"])
    set_notes(s5, "[2:00-2:40] An evolutionary workflow: a population of candidate arrangements, each an 8-bit string saying which lattice sites "
                  "are copper and which gold. Mutate and cross them, score every child with the fine-tuned network, keep the fittest, repeat. "
                  f"The chart is a real run: {summ['ga_unique_evaluations']} network calls in about two seconds. The GA lands on L1-two Cu3Au; "
                  "the dashed line is what the slow simulator says about that same structure. The network sits a few meV low across the board, "
                  "but the ranking is right, and that is what the verify step confirms. You will see exactly this live in a minute.")

    # ---------------- slide 6: the demo slide
    s6 = prs.slides[5]
    set_lines(shape_by_text(s6, "An LLM agent in VS Code"), ["An agent runs the whole loop from a README — we only watch the dashboard"])
    set_lines(shape_by_text(s6, "Forces"), ["Learn"])
    set_lines(shape_by_text(s6, "Stretch a water"), ["Fine-tune MACE on 135 DFT labels for Cu–Au. About 3 minutes on this laptop."])
    set_lines(shape_by_text(s6, "Blind evolution"), ["Test"])
    set_lines(shape_by_text(s6, "A genetic algorithm scores"), ["51 structures it never saw: how far is the network from DFT, before vs after?"])
    set_lines(shape_by_text(s6, "Agent evolution"), ["Search + verify"])
    set_lines(shape_by_text(s6, "Claude proposes"), ["A genetic algorithm with MACE as fitness; DFT confirms the winner with 3 calls, not 27."])
    text(s6, 0.8, 6.45, 11.5, 0.4, "Watch for: the error curve falling · dots hugging the line · the population converging · three green checks",
         14, False, PINK)
    set_notes(s6, "[3:00] Switch to the dashboard. Paste the README prompt to the agent. Narrate from the presenter view: block 0 (DFT labels, "
                  "done overnight on the cluster), block 1 learn, block 2 test, block 3 search, block 4 verify. Keep the audience on the dashboard; "
                  "the terminal is the agent's.")

    # ---------------- slide 7: takeaways
    s7 = prs.slides[6]
    set_lines(shape_by_text(s7, "Evolve with MACE"), ["Fine-tune on DFT, evolve with MACE, verify winners with DFT — then the lab."])
    set_lines(shape_by_text(s7, "MACE turns an LLM"), ["A 40 ms answer instead of a 10-minute one lets an agent search instead of guess."])

    # ---------------- new main-flow slide: where the agent adds value (inserted after slide 5)
    blank = prs.slide_layouts[6]
    s = prs.slides.add_slide(blank)
    header(s, "PART 2 · THE AGENT", "Where the agent earns its keep",
           "The surrogate will sometimes be wrong. Verify is a decision point, not a checkbox — the LLM decides what the slow simulator is spent on", 8)
    # decision flow (left)
    box(s, 0.6, 1.95, 5.6, 4.7, CARD)
    flow = [(0.9, 2.15, 5.0, 0.7, WHITE, "MACE proposes", "best arrangement at each mixing ratio · milliseconds", DARK),
            (0.9, 3.1, 5.0, 0.7, WHITE, "DFT verifies the shortlist", "3 slow calls instead of 27", DARK)]
    for x, y, w, h, fill, t, b, col in flow:
        box(s, x, y, w, h, fill, line="E5E7EB")
        text(s, x + 0.2, y + 0.07, w - 0.4, 0.3, t, 13.5, True, col)
        text(s, x + 0.2, y + 0.36, w - 0.4, 0.3, b, 10.5, False, GREY)
    line(s, 3.4, 2.85, 3.4, 3.1, LIGHT, 1.5, arrow=True)
    line(s, 3.4, 3.8, 3.4, 4.05, LIGHT, 1.5, arrow=True)
    outcomes = [(0.9, 4.05, 1.55, 1.3, "2E7D4F", "confirmed", "report it as a discovery; the hull is trusted here"),
                (2.63, 4.05, 1.55, 1.3, MAROON, "rejected", "add DFT's answer to training, retrain, search again"),
                (4.35, 4.05, 1.55, 1.3, "8A6D1F", "unknown", "no label yet or the model is unsure: request DFT before deciding")]
    for x, y, w, h, fill, t, b in outcomes:
        box(s, x, y, w, h, fill)
        text(s, x + 0.12, y + 0.1, w - 0.24, 0.3, t, 12.5, True, WHITE)
        text(s, x + 0.12, y + 0.45, w - 0.24, 0.85, b, 10, False, PINK if fill == MAROON else "F3F4F6")
    text(s, 0.9, 5.5, 5.0, 1.0, ["Two of these happened in building this demo: the script flagged two arrangements with no DFT label (six calculations went to the cluster), "
                               "and on stand-in data DFT rejected a 50:50 pick the network had over-stabilised by 9 meV/atom."], 10.5, False, GREY, italic=True)
    # what the LLM decides (right)
    text(s, 6.5, 1.95, 6.2, 0.3, "WHAT THE LLM DECIDES", 11, True, ROSE)
    cards_ = [("1", "Which structures deserve a slow call",
               "Novel environments, committee disagreement, hull vertices with thin margins. The budget is the scarce resource; spending it well is the job."),
              ("2", "What to propose next",
               "Chemically sensible moves — swap a layer, respect the lattice — instead of random bit flips. The mutation operator becomes a reasoning step."),
              ("3", "When to stop, and what to report",
               "The hull is stable across two rounds; rejected picks have been retrained; the answer is explained in plain words with the evidence attached.")]
    y = 2.3
    for num, t, b in cards_:
        box(s, 6.5, y, 6.2, 1.25, WHITE, line="E5E7EB")
        circle(s, 6.68, y + 0.32, 0.6, MAROON, num, 15)
        text(s, 7.45, y + 0.1, 5.1, 0.35, t, 14, True, DARK)
        text(s, 7.45, y + 0.46, 5.1, 0.75, b, 11, False, GREY)
        y += 1.4
    box(s, 6.5, 6.5, 6.2, 0.45, MAROON)
    text(s, 6.65, 6.53, 5.9, 0.4, "In today's demo these decisions are scripted; the agent runs and explains them. Making them adaptively is the frontier.",
         10.5, False, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    set_notes(s, "[2:40-3:00] The surrogate will sometimes be wrong, and that is where an agent matters. After the search, the network's best pick at "
                 "each composition goes to DFT. Confirmed: report it. Rejected: that DFT result becomes training data, retrain, search again. "
                 "Unknown or uncertain: ask DFT first. Two of these happened while we built the demo. The LLM's job is to spend the expensive "
                 "budget well, to propose chemically sensible moves instead of random ones, and to know when to stop. In the demo these decisions "
                 "are scripted; the agent runs and explains them.")
    # move it to position 6 (after the evolution slide)
    sld_ids = prs.slides._sldIdLst
    new_el = list(sld_ids)[-1]
    sld_ids.remove(new_el)
    sld_ids.insert(5, new_el)

    # ---------------- backup slides
    n = 12

    # B1 DFT
    s = prs.slides.add_slide(blank)
    header(s, "BACKUP · THE SIMULATOR", "DFT in one slide", "The slow, trusted physics calculation that produced our labels", n); n += 1
    card(s, 0.6, 1.95, 3.95, 3.75, "What it is",
         ["Electrons obey the Schrödinger equation — unsolvable directly beyond a few electrons.",
          "Density functional theory (Kohn, Nobel 1998) rewrites it in terms of the electron density — three coordinates instead of 3N — solved iteratively (the SCF loop).",
          "One approximation: the exchange–correlation functional. We use PBE, the standard for solids."], icon="ψ", body_size=11.5)
    card(s, 4.68, 1.95, 3.95, 3.75, "Why it is trusted",
         ["First principles: inputs are atomic numbers and positions; nothing fitted to experiment for the system at hand.",
          "Lattice constants within 1–2 %, formation energies to tens of meV/atom, forces for free (Hellmann–Feynman).",
          "Ours: Cu 3.65 Å, Au 4.18 Å (experiment 3.61 / 4.08)."], icon="✓", body_size=11.5)
    card(s, 8.76, 1.95, 3.95, 3.75, "Why not inside the loop",
         ["Cost grows as N³ in the electrons: minutes for 8 atoms, hours for 100, impossible for a million-step simulation.",
          "Known systematic errors — PBE underbinds Cu–Au by ~30 %. The learned model inherits them.",
          "Zero kelvin: temperature needs many more evaluations."], fill=MAROON, title_color=WHITE, body_color=PINK, icon="N³", body_size=11.5)
    box(s, 0.6, 5.9, 12.1, 1.0, WHITE, line="E5E7EB")
    text(s, 0.85, 5.97, 11.6, 0.3, "Our calculations", 12.5, True, MAROON)
    text(s, 0.85, 6.27, 11.6, 0.6, "VASP 6.3.2 · PBE · PAW (Cu, Au: 11 valence e⁻) · ENCUT 400 eV · Γ-centred 8×8×8 / 4×8×8 k-points · Methfessel–Paxton 0.1 eV · "
         "186 single points (energy + forces at fixed geometry), minutes each on 12 cores, SLURM array on TAMU HPRC Grace.", 11, False, GREY)
    footer(s, "Hohenberg & Kohn 1964; Kohn & Sham 1965; Perdew, Burke & Ernzerhof 1996")
    set_notes(s, "Backup: DFT for a non-specialist. What it is, why we trust it, and the three reasons it cannot sit inside a search loop.")

    # B2 learned potentials
    s = prs.slides.add_slide(blank)
    header(s, "BACKUP · LEARNED POTENTIALS", "From DFT to a learned potential", "Treat the energy as a regression target — with three physics constraints built in", n); n += 1
    card(s, 0.6, 1.95, 3.95, 2.75, "Locality",
         ["Total energy is a sum of atomic energies, each depending only on neighbours within a cut-off (MACE: 6 Å).",
          "This is why a model trained on 4- and 8-atom cells works on any size, and why cost is linear in N."], icon="Σ")
    card(s, 4.68, 1.95, 3.95, 2.75, "Symmetry",
         ["Translate, rotate or relabel identical atoms: the energy must not change; forces must rotate along.",
          "Built into the architecture, not learned from data."], icon="⟳")
    card(s, 8.76, 1.95, 3.95, 2.75, "Forces by differentiation",
         ["F = −∂E/∂r, computed exactly by automatic differentiation.",
          "Energy-conserving by construction, and every DFT structure gives 3N force labels — most of the training signal."], icon="∇")
    text(s, 0.6, 4.95, 12.1, 0.3, "LINEAGE", 11, True, ROSE)
    steps = [("2007", "Behler–Parrinello", "atomic energies + neural nets"), ("2010", "GAP", "Gaussian processes"),
             ("2019", "ACE", "complete many-body basis"), ("2022", "NequIP", "equivariant graph nets"), ("2022", "MACE", "ACE products inside an equivariant GNN")]
    x = 0.6
    for i, (yr, name, what) in enumerate(steps):
        fill = MAROON if name == "MACE" else CARD
        box(s, x, 5.3, 2.26, 1.25, fill)
        text(s, x + 0.15, 5.38, 2.0, 0.3, yr, 11, True, PINK if fill == MAROON else ROSE)
        text(s, x + 0.15, 5.65, 2.0, 0.35, name, 15, True, WHITE if fill == MAROON else DARK)
        text(s, x + 0.15, 6.0, 2.0, 0.5, what, 11, False, PINK if fill == MAROON else GREY)
        if i < len(steps) - 1:
            line(s, x + 2.26, 5.92, x + 2.46, 5.92, LIGHT, 1.5, arrow=True)
        x += 2.46
    footer(s, "Behler & Parrinello 2007 · Bartók et al. 2010 · Drautz 2019 · Batzner et al. 2022 · Batatia et al. NeurIPS 2022")
    set_notes(s, "Backup: the three design principles every machine-learned potential shares, and where MACE sits in the lineage.")

    # B3 MACE as a graph
    s = prs.slides.add_slide(blank)
    header(s, "BACKUP · THE MODEL", "MACE: a graph of atoms passing messages", "Two layers, many-body messages, exact rotational symmetry", n); n += 1
    box(s, 0.6, 1.95, 5.2, 4.6, CARD)
    cx, cy = 3.2, 4.2
    import math
    nb = [(1.1, 0.0), (0.55, 0.95), (-0.55, 0.95), (-1.1, 0.0), (-0.55, -0.95), (0.55, -0.95), (1.6, 0.9), (-1.5, 1.1)]
    ring = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 1.9), Inches(cy - 1.9), Inches(3.8), Inches(3.8))
    ring.fill.background(); ring.line.color.rgb = rgb(ROSE); ring.line.width = Pt(1.25); ring.line.dash_style = 4
    for dx, dy in nb:
        line(s, cx, cy, cx + dx, cy + dy, LIGHT, 1.25)
    for dx, dy in nb:
        circle(s, cx + dx - 0.16, cy + dy - 0.16, 0.32, ROSE if abs(dx) < 1.3 else LIGHT)
    circle(s, cx - 0.22, cy - 0.22, 0.44, MAROON, "i", 13)
    text(s, 0.85, 2.05, 4.7, 0.35, "atom i and its neighbourhood (cut-off 6 Å)", 12, True, DARK)
    text(s, 0.85, 6.05, 4.7, 0.45, "nodes = atoms · edges = pairs within the cut-off · each layer passes messages along edges", 11, False, GREY)
    rows = [("1", "Embed", "Each atom starts with a learned vector for its element."),
            ("2", "Collect neighbours", "For every neighbour: distance (radial basis) × direction (spherical harmonics) × its features. Summed: a two-body description Aᵢ."),
            ("3", "Multiply", "Products of Aᵢ with itself encode angles and dihedrals — many-body information in one message (body order 4). This is the ACE idea."),
            ("4", "Update, read out, repeat", "Update the atom's features; after 2 layers a readout gives Eᵢ. Total E = Σ Eᵢ; forces = −∂E/∂r."),
            ]
    y = 1.95
    for num, t, b in rows:
        box(s, 6.05, y, 6.65, 1.05, WHITE, line="E5E7EB")
        circle(s, 6.2, y + 0.25, 0.55, MAROON, num, 14)
        text(s, 6.9, y + 0.1, 5.7, 0.35, t, 14, True, DARK)
        text(s, 6.9, y + 0.43, 5.7, 0.6, b, 11, False, GREY)
        y += 1.15
    text(s, 6.05, 6.6, 6.65, 0.35, "2 layers × 6 Å = 12 Å receptive field.  MACE-MP-0: 1.5 M structures, 89 elements, PBE.", 11, True, MAROON)
    footer(s, "Batatia, Kovács, Simm, Ortner & Csányi, NeurIPS 2022; Batatia et al., arXiv:2401.00096 (MACE-MP-0)")
    set_notes(s, "Backup: the mental model. Graph, messages, products for many-body information, two layers, readout, differentiate for forces.")

    # B4 equivariance
    s = prs.slides.add_slide(blank)
    header(s, "BACKUP · SYMMETRY", "Equivariance for absolute beginners", "Rotate the input and the output rotates with it — exactly, not approximately", n); n += 1
    for k, (x0, title_, sub, rot) in enumerate([(0.6, "Invariant: the energy", "Rotate the molecule → the same number comes out", 0),
                                                (6.65, "Equivariant: the forces", "Rotate the molecule → the force arrows rotate with it", 35)]):
        box(s, x0, 1.95, 6.05, 3.55, CARD)
        text(s, x0 + 0.25, 2.1, 5.6, 0.4, title_, 16, True, DARK)
        text(s, x0 + 0.25, 2.5, 5.6, 0.4, sub, 12, False, GREY)
        for j, ang in enumerate([0, rot if rot else 0]):
            ox = x0 + 1.6 + j * 2.9
            oy = 4.15
            pts = [(0, -0.55), (0.6, 0.4), (-0.6, 0.4)]
            a = math.radians(ang if j else 0)
            P = [(ox + px * math.cos(a) - py * math.sin(a), oy + px * math.sin(a) + py * math.cos(a)) for px, py in pts]
            for (ax_, ay_), (bx_, by_) in [(P[0], P[1]), (P[1], P[2]), (P[2], P[0])]:
                line(s, ax_, ay_, bx_, by_, LIGHT, 1.25)
            for (px, py), col in zip(P, [MAROON, ROSE, ROSE]):
                circle(s, px - 0.17, py - 0.17, 0.34, col)
            if k == 1:
                for (px, py) in P:
                    vx, vy = px - ox, py - oy
                    nrm = math.hypot(vx, vy) or 1
                    line(s, px, py, px + 0.55 * vx / nrm, py + 0.55 * vy / nrm, AU, 2.25, arrow=True)
            else:
                text(s, ox - 0.9, oy + 0.75, 1.8, 0.35, "E = −3.72 eV", 12, True, MAROON, PP_ALIGN.CENTER, "Cambria Math")
        text(s, x0 + 2.75, 4.0, 0.6, 0.4, "→", 20, True, GREY, PP_ALIGN.CENTER)
        text(s, x0 + 0.25, 5.0, 5.6, 0.4, "same energy" if k == 0 else "arrows turned by the same angle — nothing else changed", 11, False, GREY, PP_ALIGN.CENTER, italic=True)
    box(s, 0.6, 5.75, 12.1, 0.95, MAROON)
    text(s, 0.85, 5.85, 11.6, 0.75, ["Why it matters: forces are vectors. A network that must learn rotation from data wastes data and still gets it slightly wrong.",
                                    "MACE builds the symmetry in — its features carry rotation labels (spherical harmonics), so symmetry is exact and training data goes further."],
         12.5, False, WHITE)
    set_notes(s, "Backup: invariance (scalars stay the same) vs equivariance (vectors transform along). MACE's features are equivariant by construction.")

    # B5 math
    s = prs.slides.add_slide(blank)
    header(s, "BACKUP · THE MATH", "The math behind MACE, gently", "Four equations, each with a plain-language reading", n); n += 1
    eqs = [("E = Σᵢ Eᵢ", "The total energy is a sum of per-atom energies. Each Eᵢ depends only on atoms within the cut-off."),
           ("Aᵢ = Σⱼ R(rᵢⱼ) · Y(r̂ᵢⱼ) ⊗ hⱼ", "Collect the neighbours: a radial function of the distance, spherical harmonics of the direction, and the neighbour's features. Summing over j makes it a two-body descriptor."),
           ("Bᵢ = Aᵢ ⊗ Aᵢ ⊗ Aᵢ  (symmetrised)", "Multiply the collection by itself. Products of pair terms encode angles and dihedrals — many-body geometry — without ever enumerating triples. Body order 4 per layer; this is the Atomic Cluster Expansion."),
           ("hᵢ ← W·Bᵢ ,   Eᵢ = readout(hᵢ) ,   F = −∂E/∂r", "Update the atom's features with learned weights, repeat for the second layer, read out the energy, and differentiate for the forces. Every step is an equivariant tensor product, so rotations stay exact.")]
    y = 1.95
    for i, (eq, why) in enumerate(eqs):
        box(s, 0.6, y, 12.1, 1.08, CARD if i < 3 else WHITE, line=None if i < 3 else "E5E7EB")
        circle(s, 0.8, y + 0.27, 0.55, MAROON, str(i + 1), 14)
        text(s, 1.55, y + 0.12, 4.8, 0.85, eq, 17, True, MAROON, font="Cambria Math", anchor=MSO_ANCHOR.MIDDLE)
        text(s, 6.45, y + 0.1, 6.1, 0.9, why, 11.5, False, GREY, anchor=MSO_ANCHOR.MIDDLE)
        y += 1.18
    text(s, 0.6, 6.7, 12.1, 0.3, "Training: minimise  100·(ΔE/atom)² + 10·(ΔF)²  over 135 DFT structures, 20 epochs, starting from MACE-MP-0's weights.", 11.5, True, DARK)
    footer(s, "Notation simplified: channel indices, learnable radial weights and the irreducible-representation bookkeeping are omitted")
    set_notes(s, "Backup: the four equations. Sum of atomic energies; two-body collection; products for many-body terms (ACE); update, read out, differentiate.")

    # B6 results
    s = prs.slides.add_slide(blank)
    header(s, "BACKUP · RESULTS", "Fine-tuning on our DFT labels: what changed", "Held out: 51 structures of 17 orderings the model never saw", n); n += 1
    tbl = s.shapes.add_table(3, 4, Inches(0.6), Inches(1.95), Inches(7.4), Inches(1.8)).table
    hdrs = ["", "pretrained MACE-MP-0", "fine-tuned (20 epochs)", "fine-tuned (30 epochs)"]
    vals = [["formation-energy error (MAE)", f"{base['formation_energy_mae_meV_per_atom']:.1f} meV/atom", f"{ft['formation_energy_mae_meV_per_atom']:.1f}", "~4"],
            ["force error (RMSE)", f"{base['force_rmse_meV_per_A']:.0f} meV/Å", f"{ft['force_rmse_meV_per_A']:.0f}", "~23"]]
    for c, h in enumerate(hdrs):
        cell = tbl.cell(0, c); cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = rgb(MAROON)
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = rgb(WHITE); r.font.name = "Arial"
    for r_, row in enumerate(vals, start=1):
        for c, v in enumerate(row):
            cell = tbl.cell(r_, c); cell.text = v
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(CARD if r_ % 2 else WHITE)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(12.5); run.font.name = "Arial"; run.font.color.rgb = rgb(DARK); run.font.bold = (c == 0)
    text(s, 0.6, 3.95, 7.4, 1.6, ["Raw MACE is already decent on energies — it was trained on PBE too. Fine-tuning's big wins: forces 4× better, and the few-meV ranking the convex hull needs.",
                                 "The fine-tuned energies sit ~9 meV/atom low across the board; that offset cancels in rankings, which is why the hull matches DFT at all 5 vertices."], 12.5, False, GREY)
    kp = [("186", "DFT single points"), ("27", "distinct orderings; 10 trained, 17 held out"), ("3 of 27", "expensive calls in the verify step — all confirmed"), ("~3.5 min", "the whole loop on a laptop CPU")]
    for i, (big, small) in enumerate(kp):
        yk = 1.95 + i * 1.18
        box(s, 8.3, yk, 4.4, 1.05, WHITE, line="E5E7EB")
        text(s, 8.5, yk + 0.08, 4.0, 0.55, big, 26, True, MAROON)
        text(s, 8.5, yk + 0.62, 4.0, 0.4, small, 11, False, GREY)
    text(s, 0.6, 5.75, 7.4, 0.9, "Training recipe: start from MACE-MP-0 small; per-element reference energies fitted by least squares; loss weights energy 100 : forces 10; Adam, lr 0.01, EMA; batch 16; 20 epochs ≈ 3 min on CPU.", 11, False, LIGHT)
    footer(s, "Numbers from the run in this repository (results/validation.json, results/discovery.json)")
    set_notes(s, "Backup: the quantitative result. Use when asked 'why fine-tune at all' or 'how good is it'.")

    # B7 Cu-Au
    s = prs.slides.add_slide(blank)
    header(s, "BACKUP · THE SYSTEM", "Cu–Au: a real system with a textbook answer", "Copper and gold mix in all proportions and order into intermetallic compounds", n); n += 1
    card(s, 0.6, 1.95, 5.3, 4.65, "Why this system",
         ["Cu₃Au — L1₂ (Au on cube corners, Cu on faces); orders below ~390 °C.",
          "CuAu — L1₀ (alternating Cu and Au layers); ~410 °C.",
          "CuAu₃ — L1₂, weaker.",
          "",
          "The Cu₃Au order–disorder transition is the classic example in physical metallurgy, and Cu–Au was the original testbed for computational ground-state searches.",
          "",
          f"Our PBE formation energies: Cu₃Au −46.7, CuAu −48.6, CuAu₃ −28.2 meV/atom (experiment is more negative — PBE underbinds).",
          "",
          "Formation energy = E(mixture) − x·E(Au) − (1−x)·E(Cu), per atom. Negative = favourable. The convex hull = the stable compounds."], body_size=12)
    s.shapes.add_picture(str(ROOT / "docs" / "img" / "hull.png"), Inches(6.1), Inches(1.95), width=Inches(6.6))
    text(s, 6.1, 6.55, 6.6, 0.4, "Every distinct ordering: network (pink) vs DFT (circles). Hull vertices agree; the star is Cu₃Au L1₂.", 10.5, False, GREY, italic=True)
    set_notes(s, "Backup: the system is real and the answer is known, which is why the audience can judge the result.")

    # B8 data
    s = prs.slides.add_slide(blank)
    header(s, "BACKUP · THE DATA", "The 186 DFT calculations", "An active-learning snapshot: DFT on under 40 % of the ordering space; the network ranks the rest", n); n += 1
    tbl = s.shapes.add_table(5, 3, Inches(0.6), Inches(1.95), Inches(12.1), Inches(2.3)).table
    rows = [["group", "count", "why it is there"],
            ["4-atom cells · 7-point volume scan for each of 5 orderings", "35", "energy vs lattice constant per composition"],
            ["4-atom cells · rattled (σ 0.05 / 0.10 Å) at three volumes", "60", "non-zero forces: the local shape of the energy surface"],
            ["8-atom cells · 10 orderings (Cu₈, Au₈ + 8 random mixed), ideal + 3 rattled", "40", "ordering patterns a 4-atom cell cannot hold"],
            ["8-atom cells · the other 17 orderings, ideal + 2 rattled", "51", "never trained on: validation in step 2, answer key in step 3"]]
    for r_, row in enumerate(rows):
        for c, v in enumerate(row):
            cell = tbl.cell(r_, c); cell.text = v
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(MAROON if r_ == 0 else (CARD if r_ % 2 else WHITE))
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(12); run.font.name = "Arial"
                    run.font.bold = (r_ == 0 or c == 1)
                    run.font.color.rgb = rgb(WHITE if r_ == 0 else DARK)
    tbl.columns[0].width = Inches(6.6); tbl.columns[1].width = Inches(1.0); tbl.columns[2].width = Inches(4.5)
    pic_h = 2.35
    pic_w = pic_h * 2210 / 715
    s.shapes.add_picture(str(ROOT / "docs" / "img" / "structures.png"), Inches((13.333 - pic_w) / 2), Inches(4.5), height=Inches(pic_h))
    footer(s, "Geometries are fixed (no relaxation) so the network and DFT are compared on identical structures; all 27 orderings have DFT labels")
    set_notes(s, "Backup: what was computed and why each group exists.")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    prs.save(out)
    print(f"wrote {out} with {len(prs.slides)} slides")


if __name__ == "__main__":
    main()
