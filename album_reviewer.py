#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
album_reviewer.py
=================

כלי מקומי לעיבוד קובצי PDF של אלבומי תמונות.
מקבל PDF שבו כל עמוד הוא כפולה, ומייצר PDF חדש עבור תיקוני לקוח:
  * קו אנכי לבן במרכז הכפולה (50% מרוחב העמוד), לכל הגובה.
  * זיהוי התמונות הנפרדות בכל כפולה.
  * מספור התמונות בסדר קריאה (שמאל→ימין, למעלה→למטה), מתאפס בכל כפולה.
  * כל מספר בתוך עיגול לבן שקוף-קלות עם מסגרת שחורה, בפינה השמאלית-העליונה של התמונה.

הכלי אינו משנה את מידות העמוד, יחס התמונה, הרזולוציה, סדר העמודים או איכות התמונות,
ושומר את הקובץ המקורי ללא שינוי (יוצר קובץ חדש: <שם>_numbered.pdf).

שימוש:
    python3 album_reviewer.py input/my_album.pdf
    python3 album_reviewer.py input/my_album.pdf --review
    python3 album_reviewer.py input/my_album.pdf --pair-pages

ראה README.md להסבר מלא.
"""

import argparse
import json
import logging
import os
import sys
import unicodedata
from datetime import datetime

# ---------------------------------------------------------------------------
# תלויות חיצוניות – מוצגת הודעה ברורה אם חסרות
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "שגיאה: לא נמצאה ספריית PyMuPDF.\n"
        "התקיני אותה עם:  pip install PyMuPDF\n"
        f"(פירוט: {exc})\n"
    )
    sys.exit(2)

try:
    import numpy as np
    import cv2  # OpenCV
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "שגיאה: לא נמצאו הספריות OpenCV / numpy.\n"
        "התקיני אותן עם:  pip install opencv-python-headless numpy\n"
        f"(פירוט: {exc})\n"
    )
    sys.exit(2)

try:
    from PIL import Image  # Pillow – לשימוש ב-previews
except Exception:  # pragma: no cover - previews ידרשו Pillow
    Image = None


# ---------------------------------------------------------------------------
# ברירות מחדל וקבועים
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_INPUT = os.path.join(APP_DIR, "input")
DIR_OUTPUT = os.path.join(APP_DIR, "output")
DIR_PREVIEWS = os.path.join(APP_DIR, "previews")
DIR_DEBUG = os.path.join(APP_DIR, "debug")
LOG_PATH = os.path.join(DIR_DEBUG, "process.log")

DEFAULTS = {
    "line_width": None,       # בנקודות PDF; None => יחסי לרוחב העמוד
    "label_size": 1.0,        # מקדם גודל לעיגול/טקסט (1.0 = ברירת מחדל)
    "min_image_area": 0.015,  # 1.5% משטח העמוד
    "dpi": 200,               # רזולוציית רינדור לזיהוי חזותי (fallback)
    "include_background": False,
    "pair_pages": False,
}

logger = logging.getLogger("album_reviewer")


# ---------------------------------------------------------------------------
# עזרי מערכת: תיקיות, לוג, שמות קבצים
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in (DIR_INPUT, DIR_OUTPUT, DIR_PREVIEWS, DIR_DEBUG):
        os.makedirs(d, exist_ok=True)


def setup_logging(verbose=True):
    ensure_dirs()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)


def unique_output_path(path):
    """אם קובץ הפלט כבר קיים – מוסיף מספר סידורי במקום לדרוס."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 2
    while True:
        candidate = f"{root}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def safe_stem(filename):
    """שם בסיס לקובץ – תומך בעברית ורווחים (ללא שינוי התווים)."""
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    return stem


# ---------------------------------------------------------------------------
# גיאומטריה של מלבנים (עובדים בפיקסלים של תמונת רינדור)
# ---------------------------------------------------------------------------
def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def _iou(a, b):
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _contains(a, b, tol=0.85):
    """האם b מוכל (ברובו) בתוך a."""
    ab = _area(b)
    return ab > 0 and _inter(a, b) / ab >= tol


def merge_duplicate_boxes(boxes, iou_thr=0.55, contain_tol=0.85):
    """הסרת מלבנים כפולים / כמעט חופפים. שומר את הגדול מבין החופפים."""
    boxes = sorted(boxes, key=_area, reverse=True)
    kept = []
    for b in boxes:
        if any(_contains(k, b, contain_tol) or _iou(k, b) > iou_thr for k in kept):
            continue
        kept.append(b)
    return kept


# ---------------------------------------------------------------------------
# זיהוי חזותי (fallback) עם OpenCV
# ---------------------------------------------------------------------------
def _flat_white_mask(gray, white_thr=234, std_thr=10.0, win=17):
    """מסכת 'לבן שטוח' – פיקסלים בהירים עם שונות מקומית נמוכה (רקע/gutter של דף)."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (win, win))
    mean_sq = cv2.blur(g * g, (win, win))
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))
    white = (gray >= white_thr) & (std <= std_thr)
    return white.astype(np.uint8)


def _border_connected(mask):
    """שומר רק רכיבים של המסכה שנוגעים בגבול התמונה (רקע/gutters של הדף,
    בניגוד ל'חורים לבנים' פנימיים בתוך תמונה כמו שמלה לבנה)."""
    m = (mask > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(m, connectivity=8)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    out = np.zeros_like(m)
    if border:
        out = np.isin(labels, list(border)).astype(np.uint8)
    return out


def _find_seam(gray_sub, min_gutter, max_gutter, bright_thr=233, edge_frac_thr=0.03):
    """מאתר 'תפר' פנימי (gutter דק ובהיר וללא קצוות) בתוך מלבן.
    מחזיר (axis, pos, width) עבור הפיצול הטוב ביותר, או None.
    axis='v' => פיצול אנכי בעמודה pos; axis='h' => פיצול אופקי בשורה pos."""
    h, w = gray_sub.shape
    edges = cv2.Canny(gray_sub, 30, 100)

    def scan(profile_bright, profile_edge, length):
        # עמודות/שורות בהירות מאוד (אחוזון 5 גבוה) ודלות קצוות
        bright = np.convolve(profile_bright, np.ones(5) / 5, mode="same")
        cand = (bright > bright_thr) & (profile_edge < edge_frac_thr)
        runs = []
        i = 0
        while i < length:
            if cand[i]:
                j = i
                while j < length and cand[j]:
                    j += 1
                runs.append((i, j - 1, j - i))  # start, end, width
                i = j
            else:
                i += 1
        # רק תפרים פנימיים ברוחב סביר של gutter
        margin = max(min_gutter, int(0.03 * length))
        valid = [
            r for r in runs
            if r[0] > margin and r[1] < length - margin
            and min_gutter <= r[2] <= max_gutter
        ]
        if not valid:
            return None
        # התפר הרחב ביותר; שובר שוויון לפי קרבה למרכז
        valid.sort(key=lambda r: (r[2], -abs((r[0] + r[1]) / 2 - length / 2)), reverse=True)
        r = valid[0]
        return (r[0], r[2])  # pos(start), width

    # אנכי: אחוזון 5 של בהירות לכל עמודה; שבר-קצוות לכל עמודה
    col_bright = np.percentile(gray_sub, 5, axis=0)
    col_edge = edges.mean(axis=0) / 255.0
    v = scan(col_bright, col_edge, w)

    # אופקי
    row_bright = np.percentile(gray_sub, 5, axis=1)
    row_edge = edges.mean(axis=1) / 255.0
    hh = scan(row_bright, row_edge, h)

    # מעדיפים את התפר הרחב יותר מבין השניים
    best = None
    if v is not None:
        best = ("v", v[0], v[1])
    if hh is not None and (best is None or hh[1] > best[2]):
        best = ("h", hh[0], hh[1])
    return best


def _split_by_seams(gray, box, min_area_px, page_w, page_h, depth=0):
    """פיצול רקורסיבי של מלבן לפי תפרים בהירים פנימיים (תמונות צמודות ללא gutter רחב)."""
    x0, y0, x1, y1 = box
    if depth > 6:
        return [box]
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return [box]
    min_gutter = max(2, int(0.002 * min(page_w, page_h)))
    max_gutter = int(0.04 * min(page_w, page_h))
    seam = _find_seam(sub, min_gutter, max_gutter)
    if seam is None:
        return [box]
    axis, pos, wid = seam
    if axis == "v":
        left = [x0, y0, x0 + pos, y1]
        right = [x0 + pos + wid, y0, x1, y1]
        parts = [left, right]
    else:
        top = [x0, y0, x1, y0 + pos]
        bottom = [x0, y0 + pos + wid, x1, y1]
        parts = [top, bottom]
    # פיצול תקף רק אם שני החלקים מספיק גדולים
    if all(_area(p) >= min_area_px for p in parts):
        out = []
        for p in parts:
            out.extend(_split_by_seams(gray, p, min_area_px, page_w, page_h, depth + 1))
        return out
    return [box]


def detect_regions_visual(page, dpi, min_area_frac):
    """נתיב fallback: רינדור העמוד ל-raster וזיהוי אזורי תמונות עם OpenCV.
    מחזיר (boxes_pdf, meta) כאשר boxes_pdf ביחידות נקודות PDF."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    H, W = img.shape[:2]
    page_area = float(H * W)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    white = _flat_white_mask(gray)
    paper = _border_connected(white)
    content = ((1 - paper) * 255).astype(np.uint8)

    k = max(3, int(min(H, W) * 0.006) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    content = cv2.morphologyEx(content, cv2.MORPH_OPEN, kernel, iterations=1)
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(content, connectivity=8)
    min_area_px = min_area_frac * page_area
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w * h < min_area_px:
            continue
        if area < 0.35 * w * h:  # בלוב דליל – כנראה לא תמונה מלאה
            continue
        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    boxes = merge_duplicate_boxes(boxes)

    # פיצול תמונות צמודות (ללא gutter רחב) לפי תפרים בהירים
    split = []
    for b in boxes:
        split.extend(_split_by_seams(gray, b, min_area_px, W, H))
    boxes = merge_duplicate_boxes(split)
    boxes = [b for b in boxes if _area(b) >= min_area_px]

    scale = 72.0 / dpi  # פיקסל -> נקודת PDF
    boxes_pdf = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for b in boxes]
    meta = {"width_px": W, "height_px": H, "page_area_px": page_area}
    return boxes_pdf, meta


def detect_regions_objects(page, min_area_frac):
    """נתיב ראשון: זיהוי אובייקטי תמונה נפרדים ב-PDF (get_image_info).
    מזהה מופעים נפרדים (placements) ולא רק קובצי תמונה ייחודיים,
    ומסיר מלבנים כמעט זהים. מחזיר רשימת מלבנים ביחידות PDF, או [] אם לא רלוונטי."""
    rect = page.rect
    page_area = rect.width * rect.height
    infos = page.get_image_info(xrefs=True)
    boxes = []
    for info in infos:
        bb = info.get("bbox")
        if not bb:
            continue
        b = [bb[0], bb[1], bb[2], bb[3]]
        if _area(b) < min_area_frac * page_area:
            continue
        boxes.append(b)
    return merge_duplicate_boxes(boxes)


def classify_page(page):
    """קובע האם העמוד משוטח (תמונת raster אחת מלאה) או מכיל אובייקטים נפרדים."""
    rect = page.rect
    page_area = rect.width * rect.height
    infos = page.get_image_info(xrefs=True)
    big = [i for i in infos if i.get("bbox") and
           _area(list(i["bbox"])) >= 0.80 * page_area]
    if len(infos) <= 1 or (len(big) >= 1 and len(infos) <= len(big) + 0):
        return "flattened" if len(infos) <= 1 else "objects"
    return "objects"


# ---------------------------------------------------------------------------
# סדר קריאה ומספור
# ---------------------------------------------------------------------------
def reading_order(boxes, page_height=None):
    """מסדר מלבנים בסדר קריאה חזותי: קיבוץ לשורות (מלמעלה למטה),
    ובכל שורה משמאל לימין.

    כלל: תמונות שמרכזן האנכי נמצא בטווח של ~12% מגובה העמוד נחשבות
    לאותה שורה, ובתוכה ממוינות משמאל לימין. סף יחסי לגובה העמוד
    (ולא לגובה התמונות) נותן סדר עקבי גם כשמעורבבות תמונות בגדלים שונים."""
    if not boxes:
        return []
    items = list(boxes)
    if page_height is None:
        page_height = max(b[3] for b in items) - min(b[1] for b in items)
    row_tol = 0.12 * page_height  # סף שיוך לשורה, יחסי לגובה העמוד

    items.sort(key=lambda b: b[1])  # לפי y עליון
    rows = []
    for b in items:
        cy = (b[1] + b[3]) / 2
        placed = False
        for row in rows:
            if abs(cy - row["cy"]) <= row_tol:
                row["boxes"].append(b)
                ys = [(x[1] + x[3]) / 2 for x in row["boxes"]]
                row["cy"] = sum(ys) / len(ys)
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "boxes": [b]})

    rows.sort(key=lambda r: r["cy"])
    ordered = []
    for row in rows:
        row["boxes"].sort(key=lambda b: b[0])  # משמאל לימין
        ordered.extend(row["boxes"])
    return ordered


def handle_background(boxes, page_area, include_background):
    """מטפל בתמונת רקע מלאה: אם מלבן מכסה כמעט את כל העמוד ויש מלבנים
    קטנים בתוכו – מסמן אותו כרקע ומחליט אם לכלול אותו במספור."""
    if not boxes:
        return boxes
    bg = [b for b in boxes if _area(b) >= 0.85 * page_area]
    small = [b for b in boxes if _area(b) < 0.85 * page_area]
    if bg and small and not include_background:
        return small
    return boxes


# ---------------------------------------------------------------------------
# ציור על העמוד: קו מרכזי + עיגולי מספור
# ---------------------------------------------------------------------------
def draw_center_line(page, line_width):
    """קו אנכי לבן בדיוק ב-50% מרוחב העמוד, לכל הגובה."""
    rect = page.rect
    cx = rect.x0 + rect.width / 2.0
    w = line_width if line_width else max(1.5, rect.width * 0.0035)
    shape = page.new_shape()
    shape.draw_line((cx, rect.y0), (cx, rect.y1))
    shape.finish(color=(1, 1, 1), width=w)
    shape.commit()
    return cx, w


def draw_number_label(page, box, number, page_rect, label_scale):
    """מצייר עיגול לבן שקוף-קלות עם מסגרת שחורה ומספר שחור בפינה
    השמאלית-העליונה של התמונה, בתוך גבולות התמונה."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0

    # גודל יחסי לעמוד (קריא גם בטלפון), מוגבל ע"י גודל התמונה
    base = min(page_rect.width, page_rect.height) * 0.028 * label_scale
    radius = max(7.0, base)
    radius = min(radius, bw * 0.42, bh * 0.42)  # לא לחרוג מהתמונה
    border = max(0.6, radius * 0.09)

    margin = radius * 0.55  # שוליים קטנים פנימה
    cx = x0 + margin + radius
    cy = y0 + margin + radius
    # מוודאים שהעיגול כולו בתוך גבולות התמונה
    cx = min(max(cx, x0 + radius + border), x1 - radius - border)
    cy = min(max(cy, y0 + radius + border), y1 - radius - border)
    center = fitz.Point(cx, cy)

    shape = page.new_shape()
    shape.draw_circle(center, radius)
    shape.finish(color=(0, 0, 0), fill=(1, 1, 1), width=border,
                 fill_opacity=0.82, stroke_opacity=1.0)
    shape.commit()

    # טקסט ממורכז בתוך העיגול
    text = str(number)
    fontname = "helv"
    fontsize = radius * (1.15 if len(text) == 1 else 0.95)
    tw = fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
    tx = cx - tw / 2.0
    ty = cy + fontsize * 0.35  # baseline מתחת למרכז
    page.insert_text(fitz.Point(tx, ty), text, fontname=fontname,
                     fontsize=fontsize, color=(0, 0, 0),
                     render_mode=0)


# ---------------------------------------------------------------------------
# בניית עמוד כפולה (עבור --pair-pages)
# ---------------------------------------------------------------------------
def build_spreads_doc(src_doc, pair_pages):
    """מחזיר (doc_לעבודה, is_temp).
    ב-pair_pages: מחבר כל שני עמודים עוקבים לכפולה אחת (ללא איבוד איכות,
    באמצעות show_pdf_page הווקטורי). אחרת: מחזיר עותק של המקור."""
    if not pair_pages:
        work = fitz.open()
        work.insert_pdf(src_doc)
        return work, True

    work = fitz.open()
    n = src_doc.page_count
    i = 0
    while i < n:
        left = src_doc[i]
        lr = left.rect
        if i + 1 < n:
            right = src_doc[i + 1]
            rr = right.rect
            width = lr.width + rr.width
            height = max(lr.height, rr.height)
            new_page = work.new_page(width=width, height=height)
            new_page.show_pdf_page(fitz.Rect(0, 0, lr.width, lr.height), src_doc, i)
            new_page.show_pdf_page(
                fitz.Rect(lr.width, 0, lr.width + rr.width, rr.height), src_doc, i + 1)
            i += 2
        else:
            # עמוד אחרון בודד – משאירים כפי שהוא
            new_page = work.new_page(width=lr.width, height=lr.height)
            new_page.show_pdf_page(fitz.Rect(0, 0, lr.width, lr.height), src_doc, i)
            i += 1
    return work, True


# ---------------------------------------------------------------------------
# ליבת העיבוד לכל כפולה
# ---------------------------------------------------------------------------
def detect_spread_boxes(page, args):
    """מזהה את מלבני התמונות בכפולה (יחידות PDF) לפי שני הנתיבים,
    ומחזיר (boxes_ordered, info)."""
    rect = page.rect
    page_area = rect.width * rect.height
    kind = classify_page(page)

    boxes = []
    used = "objects"
    if kind == "objects":
        boxes = detect_regions_objects(page, args.min_image_area)
    if kind == "flattened" or len(boxes) <= 1:
        # נתיב fallback עם OpenCV
        vboxes, _ = detect_regions_visual(page, args.dpi, args.min_image_area)
        if len(vboxes) >= len(boxes):
            boxes = vboxes
            used = "visual"

    boxes = merge_duplicate_boxes(boxes)
    boxes = handle_background(boxes, page_area, args.include_background)
    boxes = [b for b in boxes if _area(b) >= args.min_image_area * page_area]
    ordered = reading_order(boxes, page_height=rect.height)

    info = {"kind": kind, "detector": used, "count": len(ordered)}
    return ordered, info


def process_page(page, args):
    """מעבד כפולה אחת: קו מרכזי + מספור. מחזיר info לתיעוד."""
    rect = page.rect
    boxes, info = detect_spread_boxes(page, args)
    draw_center_line(page, args.line_width)
    for idx, b in enumerate(boxes, start=1):
        draw_number_label(page, b, idx, rect, args.label_size)
    info["boxes"] = boxes
    return info


# ---------------------------------------------------------------------------
# REVIEW: יצירת תמונות preview
# ---------------------------------------------------------------------------
def render_preview(page, boxes, out_path, dpi=140):
    """שומר תמונת preview של הכפולה עם המלבנים והמספרים שזוהו."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    scale = dpi / 72.0
    # קו מרכזי
    cx = int(pix.width / 2)
    cv2.line(img, (cx, 0), (cx, pix.height), (0, 120, 255), max(1, int(pix.width * 0.002)))
    for idx, b in enumerate(boxes, start=1):
        x0, y0, x1, y1 = [int(v * scale) for v in b]
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), max(2, int(pix.width * 0.003)))
        r = max(14, int(min(pix.width, pix.height) * 0.028))
        c = (x0 + r + 6, y0 + r + 6)
        cv2.circle(img, c, r, (255, 255, 255), -1)
        cv2.circle(img, c, r, (0, 0, 0), 2)
        t = str(idx)
        fs = r / 16.0
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.putText(img, t, (c[0] - tw // 2, c[1] + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
    if Image is not None:
        Image.fromarray(img).save(out_path)
    else:  # pragma: no cover
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# MANUAL: טעינת מלבנים ידניים מקובץ JSON
# ---------------------------------------------------------------------------
def load_manual(json_path):
    """טוען מיקומי תמונות ידניים.
    פורמט: { "pages": { "0": [[x0,y0,x1,y1], ...], "3": [...] } }
    הקואורדינטות ביחידות נקודות PDF."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pages = data.get("pages", data)
    manual = {}
    for k, v in pages.items():
        manual[int(k)] = [[float(c) for c in box] for box in v]
    return manual


# ---------------------------------------------------------------------------
# פתיחת מסמך עם טיפול בשגיאות
# ---------------------------------------------------------------------------
def open_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"הקובץ לא נמצא: {path}")
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"לא ניתן לפתוח את הקובץ (ייתכן שהוא פגום): {exc}")
    if doc.needs_pass or doc.is_encrypted:
        # ננסה סיסמה ריקה; אם נכשל – שגיאה ברורה
        if not doc.authenticate(""):
            raise RuntimeError(
                "הקובץ מוגן בסיסמה. הסירי את ההגנה ונסי שוב, "
                "או ספקי גרסה לא-מוגנת.")
    if doc.page_count == 0:
        raise RuntimeError("הקובץ אינו מכיל עמודים.")
    return doc


# ---------------------------------------------------------------------------
# תהליך ראשי
# ---------------------------------------------------------------------------
def run(args):
    ensure_dirs()
    input_path = args.input
    stem = safe_stem(input_path)

    logger.info("=" * 60)
    logger.info("מתחילים עיבוד: %s", input_path)
    logger.info("מצב: %s | pair_pages=%s | include_background=%s | min_image_area=%.3f | dpi=%d",
                "REVIEW" if args.review else ("MANUAL" if args.manual else "AUTO"),
                args.pair_pages, args.include_background, args.min_image_area, args.dpi)

    src = open_pdf(input_path)
    logger.info("נפתח בהצלחה. עמודים במקור: %d", src.page_count)

    # דוח מבנה קצר
    kinds = [classify_page(src[i]) for i in range(min(src.page_count, src.page_count))]
    flat = sum(1 for k in kinds if k == "flattened")
    logger.info("מבנה הקובץ: %d/%d עמודים משוטחים (תמונת raster מלאה), השאר עם אובייקטים נפרדים.",
                flat, len(kinds))

    manual = load_manual(args.manual) if args.manual else None

    # בונים מסמך כפולות לעבודה (כולל --pair-pages)
    work, _ = build_spreads_doc(src, args.pair_pages)
    n_spreads = work.page_count
    logger.info("מספר כפולות לעיבוד: %d", n_spreads)

    results = []

    if args.review:
        # מצב REVIEW – יוצרים previews בלבד
        preview_paths = []
        for i in range(n_spreads):
            page = work[i]
            if manual is not None and i in manual:
                boxes = reading_order(manual[i], page_height=page.rect.height)
                info = {"kind": "manual", "detector": "manual", "count": len(boxes), "boxes": boxes}
            else:
                boxes, info = detect_spread_boxes(page, args)
            out = os.path.join(DIR_PREVIEWS, f"{stem}_spread_{i + 1:03d}.png")
            out = unique_output_path(out)
            render_preview(page, boxes, out)
            preview_paths.append(out)
            results.append({"spread": i + 1, **info})
            logger.info("כפולה %d: זוהו %d תמונות (%s) -> %s",
                        i + 1, info["count"], info["detector"], os.path.basename(out))
        work.close()
        src.close()
        logger.info("סיום REVIEW. נוצרו %d קובצי preview בתיקייה: %s",
                    len(preview_paths), DIR_PREVIEWS)
        return {"mode": "review", "previews": preview_paths, "results": results}

    # מצב AUTO / MANUAL – מייצרים PDF סופי
    for i in range(n_spreads):
        page = work[i]
        rect = page.rect
        if manual is not None and i in manual:
            boxes = reading_order(manual[i], page_height=rect.height)
            info = {"kind": "manual", "detector": "manual", "count": len(boxes)}
        else:
            boxes, info = detect_spread_boxes(page, args)
        draw_center_line(page, args.line_width)
        for idx, b in enumerate(boxes, start=1):
            draw_number_label(page, b, idx, rect, args.label_size)
        results.append({"spread": i + 1, **info})
        logger.info("כפולה %d: מידות %.0fx%.0f | זוהו %d תמונות (%s)",
                    i + 1, rect.width, rect.height, info["count"], info["detector"])

    out_path = os.path.join(DIR_OUTPUT, f"{stem}_numbered.pdf")
    out_path = unique_output_path(out_path)
    try:
        work.save(out_path, garbage=3, deflate=True)
    except Exception as exc:
        raise RuntimeError(f"שמירת הקובץ נכשלה: {exc}")
    finally:
        work.close()
        src.close()

    logger.info("נשמר קובץ הפלט: %s", out_path)
    return {"mode": "auto", "output": out_path, "results": results,
            "n_spreads": n_spreads, "source_pages": src.page_count if not src.is_closed else None}


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="album_reviewer.py",
        description="עיבוד PDF של אלבום: קו מרכזי לבן + מספור תמונות בכל כפולה.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="נתיב לקובץ ה-PDF (למשל input/my_album.pdf)")
    p.add_argument("--review", action="store_true",
                   help="מצב REVIEW: יצירת תמונות preview לכל כפולה במקום PDF סופי")
    p.add_argument("--manual", metavar="JSON",
                   help="מצב MANUAL: קובץ JSON עם מלבני תמונות ידניים")
    p.add_argument("--line-width", type=float, default=DEFAULTS["line_width"],
                   help="עובי הקו המרכזי בנקודות PDF (ברירת מחדל: יחסי לרוחב העמוד)")
    p.add_argument("--label-size", type=float, default=DEFAULTS["label_size"],
                   help="מקדם גודל לעיגול ולמספר (1.0 = ברירת מחדל)")
    p.add_argument("--min-image-area", type=float, default=DEFAULTS["min_image_area"],
                   help="שטח מינימלי של תמונה כשבר משטח העמוד (0.015 = 1.5%%)")
    p.add_argument("--dpi", type=int, default=DEFAULTS["dpi"],
                   help="רזולוציית רינדור לזיהוי החזותי (fallback)")
    p.add_argument("--include-background", action="store_true",
                   help="לכלול תמונת רקע מלאה במספור")
    p.add_argument("--pair-pages", action="store_true",
                   help="חבר כל שני עמודים עוקבים לכפולה אחת לפני העיבוד")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    setup_logging(verbose=True)
    try:
        result = run(args)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("שגיאה: %s", exc)
        print(f"\n❌ שגיאה: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        logger.exception("שגיאה בלתי צפויה")
        print(f"\n❌ שגיאה בלתי צפויה: {exc}", file=sys.stderr)
        return 1

    print()
    if result["mode"] == "review":
        print("✅ מצב REVIEW הושלם.")
        print(f"נוצרו {len(result['previews'])} קובצי preview בתיקייה: {DIR_PREVIEWS}")
        for pth in result["previews"][:12]:
            print("   •", pth)
    else:
        print("✅ הקובץ נוצר בהצלחה!")
        print(f"נתיב הפלט: {result['output']}")
    print(f"לוג מפורט: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
