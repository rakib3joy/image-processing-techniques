import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(40, 22), facecolor='#0d1117')
ax  = fig.add_axes([0, 0, 1, 1])
W, H = 40, 22
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.axis('off'); ax.set_facecolor('#0d1117')

# ── palette ──────────────────────────────────────────────────────────────────
BG     = '#0d1117'
DARK2  = '#161b22'
DARK3  = '#1e252f'
BORDER = '#30363d'
CYAN   = '#2dd4dc'
BLUE   = '#4f8ef7'
PURPLE = '#b07cf8'
GREEN  = '#4ade80'
YELLOW = '#fbbf24'
WHITE  = '#e6edf3'
GRAY   = '#8b949e'
DGRAY  = '#4a5568'

# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, title, subs=None,
        ec=CYAN, fc=DARK2, lw=2.0, r=0.30,
        tfs=11, sfs=8.5, tc=WHITE, sc=GRAY):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)
    lines = [title] + (subs if subs else [])
    sizes = [tfs]   + [sfs]*(len(lines)-1)
    cols  = [tc]    + [sc]*(len(lines)-1)
    total_h = 0.42 * (len(lines)-1)
    for i, (l, fs, c) in enumerate(zip(lines, sizes, cols)):
        yt = cy + total_h/2 - i*0.42
        wt = 'bold' if i == 0 else 'normal'
        ax.text(cx, yt, l, ha='center', va='center',
                fontsize=fs, color=c, fontweight=wt,
                fontfamily='DejaVu Sans', zorder=4)

def big_box(ax, cx, cy, w, h, title, items,
            ec=CYAN, fc=DARK2, lw=2.0, r=0.35,
            tfs=12, ifs=9, tc=WHITE, ic=GRAY, div=True):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)
    # title
    ax.text(cx, cy+h/2-0.52, title, ha='center', va='center',
            fontsize=tfs, color=tc, fontweight='bold',
            fontfamily='DejaVu Sans', zorder=4)
    if div:
        ax.plot([cx-w/2+0.3, cx+w/2-0.3], [cy+h/2-0.92, cy+h/2-0.92],
                color=ec, lw=0.8, alpha=0.5, zorder=4)
    # items
    n = len(items)
    start_y = cy + h/2 - 1.25
    gap = (h - 1.35) / max(n, 1)
    for i, it in enumerate(items):
        ax.text(cx, start_y - i*gap, it, ha='center', va='center',
                fontsize=ifs, color=ic,
                fontfamily='DejaVu Sans', zorder=4)

def arrow(ax, x1, y1, x2, y2, col=CYAN, lw=2.0, rad=0.0, ms=16):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                mutation_scale=ms,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=5)

def seg_bg(ax, x, y, w, h, fc='#141a24', ec=BORDER, label='', lfs=8.5):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=0.45",
                       facecolor=fc, edgecolor=ec, linewidth=1.2,
                       alpha=0.55, zorder=1)
    ax.add_patch(p)
    if label:
        ax.text(x+w/2, y+h-0.28, label, ha='center', va='top',
                fontsize=lfs, color=ec, fontstyle='italic',
                fontfamily='DejaVu Sans', zorder=2)

# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
ax.text(W/2, 21.15,
        'Breast Ultrasound Tumor Segmentation & Classification',
        ha='center', va='center', fontsize=22, fontweight='bold',
        color=WHITE, fontfamily='DejaVu Sans', zorder=6)
ax.text(W/2, 20.5,
        'Multi-Task U-Net   ·   BiFPN Multi-Task U-Net   ·   BUSI Dataset',
        ha='center', va='center', fontsize=12, color=CYAN,
        fontfamily='DejaVu Sans', zorder=6)
ax.plot([1.0, W-1.0], [20.1, 20.1], color=BORDER, lw=1.0, zorder=2)

# ════════════════════════════════════════════════════════════════════════════
# STAGE HEADER BAR
# ════════════════════════════════════════════════════════════════════════════
stage_info = [
    (3.2,  'Dataset'),
    (8.1,  'Preprocessing'),
    (13.2, 'Models'),
    (18.5, 'Shared Arch.'),
    (23.8, 'Outputs'),
    (28.8, 'Training'),
    (33.8, 'Evaluation'),
    (38.2, 'Inference'),
]
for sx, sl in stage_info:
    ax.text(sx, 19.55, sl, ha='center', va='center',
            fontsize=9, color=DGRAY, fontfamily='DejaVu Sans', zorder=6)
ax.plot([1.0, W-1.0], [19.25, 19.25], color=BORDER, lw=0.8, ls='--', zorder=2)

# ════════════════════════════════════════════════════════════════════════════
# COLUMN X-CENTRES
# ════════════════════════════════════════════════════════════════════════════
CX = [3.2, 8.1, 13.2, 18.5, 23.8, 28.8, 33.8, 38.2]
MY = 11.5   # main pipeline y

# ════════════════════════════════════════════════════════════════════════════
# COL 0 – BUSI DATASET
# ════════════════════════════════════════════════════════════════════════════
seg_bg(ax, 1.0, MY-1.5, 4.4, 3.0, fc='#0d1a2e', ec='#2d5a9e')
box(ax, CX[0], MY, 3.8, 2.5,
    'BUSI Dataset',
    subs=['780 ultrasound images', '3 diagnostic classes',
          'Benign · Malignant · Normal'],
    ec=BLUE, fc='#0c1928', lw=2.2, r=0.35,
    tfs=12, sfs=9, tc=WHITE, sc=GRAY)

# ════════════════════════════════════════════════════════════════════════════
# COL 1 – PREPROCESSING (vertical stack of 4)
# ════════════════════════════════════════════════════════════════════════════
PBW = 4.2
PBH = 1.45
PY  = [MY+3.5, MY+1.7, MY-0.1, MY-1.9]
PRE_LABELS = [
    ('Image Loading',     ['Read images + masks from disk']),
    ('Preprocessing',     ['Resize to 256×256 pixels', 'Normalize pixel values [0, 1]']),
    ('Data Augmentation', ['Flip · Rotate · Brightness', 'Zoom · Gaussian Noise']),
    ('Train / Val Split', ['80% Training · 20% Validation']),
]
seg_bg(ax, CX[1]-PBW/2-0.3, MY-2.9, PBW+0.6, 7.6,
       fc='#0d1a2e', ec='#2d5a9e', label='Data Pipeline', lfs=9)

for i, (cy, (lbl, subs)) in enumerate(zip(PY, PRE_LABELS)):
    box(ax, CX[1], cy, PBW, PBH, lbl, subs=subs,
        ec=BLUE, fc='#0e1e35', lw=1.8, r=0.28,
        tfs=10.5, sfs=8.5)
    if i < 3:
        arrow(ax, CX[1], cy - PBH/2, CX[1], PY[i+1] + PBH/2,
              col=BLUE, lw=1.6)

# connect dataset to first pre box
arrow(ax, CX[0]+1.9, MY, CX[1]-PBW/2-0.15, PY[0], col=BLUE, lw=1.8)

# ════════════════════════════════════════════════════════════════════════════
# COL 2 – MODEL VARIANTS
# ════════════════════════════════════════════════════════════════════════════
MBW = 4.2
MBH = 1.6
M1Y = MY + 2.0
M2Y = MY - 1.4
seg_bg(ax, CX[2]-MBW/2-0.3, MY-2.9, MBW+0.6, 6.8,
       fc='#16102b', ec='#6230c8', label='Model Variants', lfs=9)

box(ax, CX[2], M1Y, MBW, MBH,
    'Multi-Task U-Net',
    subs=['Standard encoder-decoder', 'Symmetric skip connections'],
    ec=PURPLE, fc='#190f32', lw=2.0, r=0.30,
    tfs=11, sfs=8.5, tc=PURPLE, sc=GRAY)

box(ax, CX[2], M2Y, MBW, MBH,
    'BiFPN Multi-Task U-Net',
    subs=['Bi-directional feature pyramid', 'Weighted multi-scale fusion'],
    ec=PURPLE, fc='#190f32', lw=2.0, r=0.30,
    tfs=11, sfs=8.5, tc=PURPLE, sc=GRAY)

# separator label
ax.text(CX[2], MY+0.3, 'OR', ha='center', va='center',
        fontsize=10, color=DGRAY, fontweight='bold',
        fontfamily='DejaVu Sans', zorder=6)

# connect preprocessing → models
arrow(ax, CX[1]+PBW/2+0.15, PY[-1], CX[2]-MBW/2-0.15, M1Y,
      col=PURPLE, lw=1.8)
arrow(ax, CX[1]+PBW/2+0.15, PY[-1], CX[2]-MBW/2-0.15, M2Y,
      col=PURPLE, lw=1.8)

# connect models → shared arch
arrow(ax, CX[2]+MBW/2+0.15, M1Y, CX[3]-2.25, MY+1.5, col=PURPLE, lw=1.8)
arrow(ax, CX[2]+MBW/2+0.15, M2Y, CX[3]-2.25, MY-1.5, col=PURPLE, lw=1.8)

# ════════════════════════════════════════════════════════════════════════════
# COL 3 – SHARED ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
ABW = 4.2
ABH = 1.7
EY  = MY + 2.0
DY  = MY + 0.0
HY  = MY - 2.2
seg_bg(ax, CX[3]-ABW/2-0.3, MY-3.6, ABW+0.6, 7.6,
       fc='#0a1e12', ec='#1a5c30', label='Shared Architecture', lfs=9)

box(ax, CX[3], EY, ABW, ABH,
    'Shared Encoder',
    subs=['Contracting path', 'Max-pooling down-sampling'],
    ec=GREEN, fc='#091a10', lw=2.0, r=0.30,
    tfs=11, sfs=8.5, tc=GREEN, sc=GRAY)

box(ax, CX[3], DY, ABW, ABH,
    'Segmentation Decoder',
    subs=['UpSampling2D  ·  skip connections', 'Conv(1) sigmoid output'],
    ec=GREEN, fc='#091a10', lw=2.0, r=0.30,
    tfs=11, sfs=8.5, tc=GREEN, sc=GRAY)

box(ax, CX[3], HY, ABW, ABH,
    'Classification Head',
    subs=['Global average pooling', 'Fully-connected softmax'],
    ec=GREEN, fc='#091a10', lw=2.0, r=0.30,
    tfs=11, sfs=8.5, tc=GREEN, sc=GRAY)

arrow(ax, CX[3], EY-ABH/2, CX[3], DY+ABH/2, col=GREEN, lw=1.8)
arrow(ax, CX[3], DY-ABH/2, CX[3], HY+ABH/2, col=GREEN, lw=1.6)

# skip connection arc (encoder → decoder)
ax.annotate('',
    xy=(CX[3]+ABW/2-0.15, DY+ABH/2-0.2),
    xytext=(CX[3]+ABW/2-0.15, EY-ABH/2+0.2),
    arrowprops=dict(arrowstyle='->', color='#1a7a38', lw=1.2,
                    mutation_scale=12,
                    connectionstyle='arc3,rad=-0.7'),
    zorder=5)
ax.text(CX[3]+ABW/2+1.0, MY+1.05, 'skip\nconn.',
        ha='center', va='center', fontsize=8, color='#1a7a38',
        fontstyle='italic', fontfamily='DejaVu Sans', zorder=6)

# ════════════════════════════════════════════════════════════════════════════
# COL 4 – OUTPUTS
# ════════════════════════════════════════════════════════════════════════════
OBW = 4.2
OBH = 1.7
OSY = DY
OCY = HY
seg_bg(ax, CX[4]-OBW/2-0.3, MY-3.6, OBW+0.6, 7.6,
       fc='#1e1805', ec='#7a5a00', label='Model Outputs', lfs=9)

box(ax, CX[4], OSY, OBW, OBH,
    'Tumor Segmentation Mask',
    subs=['Binary pixel-level prediction', '256×256 output mask'],
    ec=YELLOW, fc='#201800', lw=2.2, r=0.30,
    tfs=11, sfs=8.5, tc=YELLOW, sc=GRAY)

box(ax, CX[4], OCY, OBW, OBH,
    'Tumor Classification',
    subs=['3-class softmax output', 'Benign · Malignant · Normal'],
    ec=YELLOW, fc='#201800', lw=2.2, r=0.30,
    tfs=11, sfs=8.5, tc=YELLOW, sc=GRAY)

arrow(ax, CX[3]+ABW/2+0.15, DY, CX[4]-OBW/2-0.15, OSY, col=YELLOW, lw=1.8)
arrow(ax, CX[3]+ABW/2+0.15, HY, CX[4]-OBW/2-0.15, OCY, col=YELLOW, lw=1.8)

# ════════════════════════════════════════════════════════════════════════════
# COL 5 – TRAINING
# ════════════════════════════════════════════════════════════════════════════
TBW = 4.5
TBH = 6.8
TX  = CX[5]
TY  = MY - 0.3
seg_bg(ax, TX-TBW/2-0.3, TY-TBH/2-0.3, TBW+0.6, TBH+0.6,
       fc='#1a1204', ec='#7a5a00', label='Training', lfs=9)

big_box(ax, TX, TY, TBW, TBH,
        'Training',
        ['Dice Loss  (segmentation)',
         'Cross-Entropy Loss  (classification)',
         '━━━━━━━━━━━━━━━━━',
         'Adam Optimizer  ·  lr = 1e-3',
         'Batch size 16  ·  Max 50 epochs',
         '━━━━━━━━━━━━━━━━━',
         'Early Stopping  (patience 10)',
         'Model Checkpointing'],
        ec=YELLOW, fc='#1c1503', lw=2.2, r=0.35,
        tfs=13, ifs=9, tc=WHITE, ic=GRAY)

# losses arrow label
arrow(ax, CX[4]+OBW/2+0.2, OSY, TX-TBW/2-0.15, TY+1.5, col=YELLOW, lw=1.8)
arrow(ax, CX[4]+OBW/2+0.2, OCY, TX-TBW/2-0.15, TY-1.5, col=YELLOW, lw=1.8)

# ════════════════════════════════════════════════════════════════════════════
# COL 6 – EVALUATION
# ════════════════════════════════════════════════════════════════════════════
EBW = 4.5
EBH = 6.8
EX  = CX[6]
EY2 = MY - 0.3
seg_bg(ax, EX-EBW/2-0.3, EY2-EBH/2-0.3, EBW+0.6, EBH+0.6,
       fc='#0b1424', ec='#1c4a8a', label='Evaluation', lfs=9)

big_box(ax, EX, EY2, EBW, EBH,
        'Evaluation Metrics',
        ['Dice Score',
         'Intersection over Union (IoU)',
         '━━━━━━━━━━━━━━━━━',
         'Pixel Accuracy',
         'Precision  ·  Recall  ·  F1',
         '━━━━━━━━━━━━━━━━━',
         'Per-class Classification Report',
         'Confusion Matrix'],
        ec=BLUE, fc='#0a1628', lw=2.2, r=0.35,
        tfs=13, ifs=9, tc=WHITE, ic=GRAY)

arrow(ax, TX+TBW/2+0.2, TY, EX-EBW/2-0.15, EY2, col=BLUE, lw=2.0)

# ════════════════════════════════════════════════════════════════════════════
# COL 7 – INFERENCE
# ════════════════════════════════════════════════════════════════════════════
IBW = 3.8
IBH = 1.7
IX  = CX[7]
IY0 = MY + 2.5
IY1 = MY + 0.4
IY2 = MY - 1.7
seg_bg(ax, IX-IBW/2-0.3, MY-2.9, IBW+0.6, 7.0,
       fc='#061818', ec='#0a5555', label='Inference', lfs=9)

box(ax, IX, IY0, IBW, IBH,
    'Input Image',
    subs=['Ultrasound scan', 'Any resolution → resized'],
    ec=CYAN, fc='#071c1c', lw=2.0, r=0.30,
    tfs=11, sfs=8.5, tc=WHITE, sc=GRAY)

box(ax, IX, IY1, IBW, IBH,
    'Predicted Mask',
    subs=['Tumor segmentation', 'Binary overlay'],
    ec=CYAN, fc='#071c1c', lw=2.0, r=0.30,
    tfs=11, sfs=8.5, tc=CYAN, sc=GRAY)

box(ax, IX, IY2, IBW, IBH,
    'Predicted Class',
    subs=['Benign · Malignant · Normal', 'Softmax confidence score'],
    ec=CYAN, fc='#071c1c', lw=2.0, r=0.30,
    tfs=11, sfs=8.5, tc=CYAN, sc=GRAY)

arrow(ax, IX, IY0-IBH/2, IX, IY1+IBH/2, col=CYAN, lw=1.8)
arrow(ax, IX, IY1-IBH/2, IX, IY2+IBH/2, col=CYAN, lw=1.8)
arrow(ax, EX+EBW/2+0.2, EY2, IX-IBW/2-0.15, IY0, col=CYAN, lw=2.0)

# ════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════
leg = [
    (BLUE,   'Data Pipeline'),
    (PURPLE, 'Model Variants'),
    (GREEN,  'Shared Architecture'),
    (YELLOW, 'Outputs & Training'),
    (BLUE,   'Evaluation Metrics'),
    (CYAN,   'Inference'),
]
lx0, ly0 = 1.2, 3.8
ax.text(lx0, ly0+0.6, 'Color Legend', ha='left', va='center',
        fontsize=10, color=GRAY, fontweight='bold',
        fontfamily='DejaVu Sans', zorder=6)
for i, (col, lbl) in enumerate(leg):
    ci = i % 3; ri = i // 3
    lx = lx0 + ci * 6.0
    ly = ly0 - ri * 0.75
    p = FancyBboxPatch((lx, ly-0.22), 0.55, 0.44,
                       boxstyle="round,pad=0,rounding_size=0.08",
                       facecolor='#1a1f27', edgecolor=col, linewidth=2.0, zorder=3)
    ax.add_patch(p)
    ax.text(lx+0.75, ly, lbl, ha='left', va='center',
            fontsize=9, color=GRAY, fontfamily='DejaVu Sans', zorder=4)

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
ax.plot([0.8, W-0.8], [1.8, 1.8], color=BORDER, lw=0.9, zorder=2)
ax.text(W/2, 1.1,
        'BUSI Dataset  ·  TensorFlow / Keras  ·  Multi-Task U-Net  ·  BiFPN Multi-Task U-Net  ·  '
        'Dice + Cross-Entropy Loss  ·  Adam Optimizer',
        ha='center', va='center', fontsize=9, color=DGRAY,
        fontfamily='DejaVu Sans', zorder=6)

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out = '/Users/joy/Documents/personal/image-processing-techniques/architecture.png'
fig.savefig(out, dpi=200, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print(f'Saved → {out}')
