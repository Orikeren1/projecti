#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
album_gui.py
============

ממשק גרפי פשוט (Tkinter) לכלי עיבוד אלבומי התמונות, למשתמשת ב-Mac.
מאפשר:
  * לבחור קובץ PDF.
  * לבחור תיקיית פלט.
  * לסמן האם כל עמוד הוא כפולה או שיש לחבר זוגות עמודים (--pair-pages).
  * לבחור האם לכלול תמונת רקע במספור.
  * לבחור מצב AUTO או REVIEW.
  * ללחוץ על "צור PDF ללקוח".
  * לראות הודעת הצלחה וקישור למיקום הקובץ.

הפעלה:  python3 album_gui.py   (או דאבל-קליק על run_app.command)
"""

import os
import sys
import threading
import traceback
import subprocess

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# מייבאים את מנוע העיבוד
import album_reviewer as ar


class AlbumGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("מספור אלבום תמונות – כלי תיקונים")
        self.geometry("640x560")
        self.minsize(600, 520)
        self.configure(padx=18, pady=18)

        self.pdf_path = tk.StringVar()
        self.out_dir = tk.StringVar(value=ar.DIR_OUTPUT)
        self.pair_pages = tk.BooleanVar(value=False)
        self.include_bg = tk.BooleanVar(value=False)
        self.mode = tk.StringVar(value="AUTO")

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        try:
            self.tk.call("tk", "scaling", 1.3)
        except Exception:
            pass

        title = tk.Label(self, text="מספור אלבום תמונות ליצירת PDF ללקוח",
                         font=("Helvetica", 17, "bold"))
        title.pack(pady=(0, 4))
        sub = tk.Label(self, text="קו מרכזי לבן + מספור אוטומטי של התמונות בכל כפולה",
                       font=("Helvetica", 11), fg="#555")
        sub.pack(pady=(0, 14))

        # --- בחירת קובץ PDF ---
        frm_file = tk.LabelFrame(self, text="  1. קובץ ה-PDF של האלבום  ",
                                 font=("Helvetica", 11, "bold"), padx=10, pady=10)
        frm_file.pack(fill="x", pady=6)
        tk.Entry(frm_file, textvariable=self.pdf_path).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(frm_file, text="בחרי קובץ…", command=self.choose_pdf).pack(side="left")

        # --- תיקיית פלט ---
        frm_out = tk.LabelFrame(self, text="  2. תיקיית הפלט  ",
                                font=("Helvetica", 11, "bold"), padx=10, pady=10)
        frm_out.pack(fill="x", pady=6)
        tk.Entry(frm_out, textvariable=self.out_dir).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(frm_out, text="בחרי תיקייה…", command=self.choose_out).pack(side="left")

        # --- אפשרויות ---
        frm_opt = tk.LabelFrame(self, text="  3. אפשרויות  ",
                                font=("Helvetica", 11, "bold"), padx=10, pady=10)
        frm_opt.pack(fill="x", pady=6)

        tk.Radiobutton(frm_opt, text="כל עמוד הוא כפולה שלמה",
                       variable=self.pair_pages, value=False).pack(anchor="w")
        tk.Radiobutton(frm_opt, text="כל עמוד הוא דף בודד – חברי כל שני עמודים לכפולה",
                       variable=self.pair_pages, value=True).pack(anchor="w")

        tk.Checkbutton(frm_opt, text="כלול תמונת רקע מלאה במספור",
                       variable=self.include_bg).pack(anchor="w", pady=(6, 0))

        # --- מצב עבודה ---
        frm_mode = tk.LabelFrame(self, text="  4. מצב עבודה  ",
                                 font=("Helvetica", 11, "bold"), padx=10, pady=10)
        frm_mode.pack(fill="x", pady=6)
        tk.Radiobutton(frm_mode, text="AUTO – יצירת ה-PDF הסופי מיד",
                       variable=self.mode, value="AUTO").pack(anchor="w")
        tk.Radiobutton(frm_mode,
                       text="REVIEW – יצירת תמונות תצוגה מקדימה לבדיקה לפני ה-PDF",
                       variable=self.mode, value="REVIEW").pack(anchor="w")

        # --- כפתור ראשי + פס התקדמות ---
        self.btn = tk.Button(self, text="צור PDF ללקוח", font=("Helvetica", 14, "bold"),
                             bg="#2e7d32", fg="white", activebackground="#1b5e20",
                             height=2, command=self.on_run)
        self.btn.pack(fill="x", pady=(14, 6))

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.status = tk.Label(self, text="", font=("Helvetica", 10), fg="#333")
        self.status.pack(pady=(2, 0))

    # ------------------------------------------------------------------
    def choose_pdf(self):
        path = filedialog.askopenfilename(
            title="בחרי קובץ PDF של אלבום",
            filetypes=[("PDF files", "*.pdf"), ("כל הקבצים", "*.*")])
        if path:
            self.pdf_path.set(path)

    def choose_out(self):
        path = filedialog.askdirectory(title="בחרי תיקיית פלט")
        if path:
            self.out_dir.set(path)

    # ------------------------------------------------------------------
    def on_run(self):
        pdf = self.pdf_path.get().strip()
        if not pdf:
            messagebox.showwarning("חסר קובץ", "אנא בחרי קובץ PDF.")
            return
        if not os.path.exists(pdf):
            messagebox.showerror("שגיאה", "הקובץ שנבחר לא נמצא.")
            return

        self.btn.config(state="disabled")
        self.progress.pack(fill="x", pady=(6, 4))
        self.progress.start(12)
        self.status.config(text="מעבד… זה עשוי לקחת רגע.", fg="#333")

        t = threading.Thread(target=self._run_worker, args=(pdf,), daemon=True)
        t.start()

    def _run_worker(self, pdf):
        try:
            # מפנים את תיקיות הפלט של המנוע לפי בחירת המשתמשת
            out_dir = self.out_dir.get().strip() or ar.DIR_OUTPUT
            os.makedirs(out_dir, exist_ok=True)
            ar.DIR_OUTPUT = out_dir
            ar.ensure_dirs()
            ar.setup_logging(verbose=False)

            class Args:
                pass
            args = Args()
            args.input = pdf
            args.review = (self.mode.get() == "REVIEW")
            args.manual = None
            args.line_width = ar.DEFAULTS["line_width"]
            args.label_size = ar.DEFAULTS["label_size"]
            args.min_image_area = ar.DEFAULTS["min_image_area"]
            args.dpi = ar.DEFAULTS["dpi"]
            args.include_background = self.include_bg.get()
            args.pair_pages = self.pair_pages.get()

            result = ar.run(args)
            self.after(0, self._on_success, result)
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(0, self._on_error, str(exc), tb)

    # ------------------------------------------------------------------
    def _on_success(self, result):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn.config(state="normal")

        if result["mode"] == "review":
            folder = ar.DIR_PREVIEWS
            n = len(result["previews"])
            self.status.config(text=f"נוצרו {n} תמונות תצוגה מקדימה.", fg="#2e7d32")
            if messagebox.askyesno(
                    "הצלחה",
                    f"נוצרו {n} תמונות תצוגה מקדימה בתיקייה:\n{folder}\n\n"
                    "לפתוח את התיקייה?"):
                self._open_path(folder)
        else:
            out = result["output"]
            self.status.config(text="ה-PDF נוצר בהצלחה!", fg="#2e7d32")
            if messagebox.askyesno(
                    "הצלחה",
                    f"ה-PDF ללקוח נוצר בהצלחה!\n\n{out}\n\nלפתוח את מיקום הקובץ?"):
                self._open_path(os.path.dirname(out))

    def _on_error(self, msg, tb):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn.config(state="normal")
        self.status.config(text="אירעה שגיאה.", fg="#c62828")
        messagebox.showerror("שגיאה", f"{msg}\n\nפרטים נוספים נכתבו ללוג:\n{ar.LOG_PATH}")

    # ------------------------------------------------------------------
    @staticmethod
    def _open_path(path):
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", path])
            elif os.name == "nt":
                os.startfile(path)  # noqa
            else:
                subprocess.run(["xdg-open", path])
        except Exception:
            pass


def main():
    app = AlbumGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
