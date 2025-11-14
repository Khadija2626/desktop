import tkinter as tk
from tkinter import messagebox, ttk
import random, time, csv, os, datetime

# ------------------ DATA ------------------
words = [
    "apple","banana","keyboard","python","developer","coffee","monitor",
    "function","variable","iteration","random","speed","accuracy",
    "practice","challenge"
]

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Practice makes progress every single day.",
    "Typing speed improves with focused practice.",
    "Write clean code and keep learning new things.",
    "Small goals compound into big achievements."
]

LEADERBOARD = "leaderboard.csv"
GOALS = "daily_goals.csv"

# ------------------ FILE SETUP ------------------
def ensure_files():
    if not os.path.exists(LEADERBOARD):
        with open(LEADERBOARD, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name","wpm","accuracy","date"])

    if not os.path.exists(GOALS):
        with open(GOALS, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date","goal","best"])

ensure_files()

# ------------------ MAIN APP ------------------
class TypingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Practice App")
        self.root.geometry("900x550")
        self.theme = "dark"

        self.start_time = 0
        self.prompt_text = ""

        self.build_ui()
        self.apply_theme()

    # ------------------ UI ------------------
    def build_ui(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        self.title_lbl = tk.Label(self.frame, text="Typing Practice App", font=("Arial", 26, "bold"))
        self.title_lbl.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(self.frame)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="10 Words Test", width=20, command=self.start_words_test).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Sentence Test", width=20, command=self.start_sentence_test).grid(row=0, column=1, padx=10)
        tk.Button(btn_frame, text="Leaderboard", width=20, command=self.show_leaderboard).grid(row=1, column=0, padx=10, pady=5)
        tk.Button(btn_frame, text="Daily Goal", width=20, command=self.set_daily_goal).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(btn_frame, text="Toggle Theme", width=20, command=self.toggle_theme).grid(row=2, column=0, columnspan=2, pady=10)

        # Text prompt
        self.prompt_lbl = tk.Label(self.frame, text="", wraplength=800, font=("Arial", 16))
        self.prompt_lbl.pack(pady=15)

        # Input box
        self.input_box = tk.Text(self.frame, height=5, width=80, font=("Arial", 15))
        self.input_box.pack(pady=10)
        self.input_box.bind("<Key>", self.start_timer)

        # Submit button
        self.submit_btn = tk.Button(self.frame, text="Submit", width=20, command=self.calculate_results)
        self.submit_btn.pack()

        # Result label
        self.result_lbl = tk.Label(self.frame, text="", font=("Arial", 14))
        self.result_lbl.pack(pady=10)

    # ------------------ THEME ------------------
    def apply_theme(self):
        if self.theme == "dark":
            bg = "#1e1e1e"
            fg = "white"
            btn_bg = "#333"
        else:
            bg = "white"
            fg = "black"
            btn_bg = "#e0e0e0"

        self.root.configure(bg=bg)
        self.frame.configure(bg=bg)
        self.title_lbl.configure(bg=bg, fg=fg)
        self.prompt_lbl.configure(bg=bg, fg=fg)
        self.result_lbl.configure(bg=bg, fg=fg)
        self.input_box.configure(bg="#2e2e2e" if self.theme=="dark" else "white", fg="white" if self.theme=="dark" else "black")

        for widget in self.frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.configure(bg=btn_bg, fg=fg)

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.apply_theme()

    # ------------------ TEST GENERATION ------------------
    def start_words_test(self):
        self.prompt_text = " ".join(random.choice(words) for _ in range(10))
        self.show_prompt()

    def start_sentence_test(self):
        self.prompt_text = random.choice(sentences)
        self.show_prompt()

    def show_prompt(self):
        self.prompt_lbl.config(text=self.prompt_text)
        self.input_box.delete("1.0", tk.END)
        self.result_lbl.config(text="")
        self.start_time = 0

    def start_timer(self, event):
        if self.start_time == 0:
            self.start_time = time.time()

    # ------------------ RESULTS ------------------
    def calculate_results(self):
        if self.start_time == 0:
            return

        typed = self.input_box.get("1.0", tk.END).strip()
        total_time = time.time() - self.start_time
        minutes = total_time / 60

        correct_chars = sum(1 for i in range(min(len(typed), len(self.prompt_text))) if typed[i] == self.prompt_text[i])

        accuracy = (correct_chars / len(self.prompt_text)) * 100
        wpm = (correct_chars / 5) / minutes if minutes > 0 else 0

        self.result_lbl.config(
            text=f"Time: {total_time:.2f}s   |   WPM: {wpm:.2f}   |   Accuracy: {accuracy:.2f}%"
        )

        self.ask_leaderboard_save(wpm, accuracy)
        self.update_daily_goal(wpm)

    # ------------------ LEADERBOARD ------------------
    def ask_leaderboard_save(self, wpm, acc):
        name = tk.simpledialog.askstring("Leaderboard", "Enter your name (or leave blank):")
        if name:
            with open(LEADERBOARD, "a", newline="") as f:
                csv.writer(f).writerow([name, f"{wpm:.2f}", f"{acc:.2f}", datetime.date.today()])

    def show_leaderboard(self):
        win = tk.Toplevel(self.root)
        win.title("Leaderboard")
        win.geometry("500x400")

        table = ttk.Treeview(win, columns=("name","wpm","acc","date"), show="headings")
        table.pack(fill="both", expand=True)

        for col in ("name","wpm","acc","date"):
            table.heading(col, text=col.capitalize())

        rows = []
        with open(LEADERBOARD) as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                rows.append(r)

        rows.sort(key=lambda x: float(x[1]), reverse=True)

        for r in rows[:20]:
            table.insert("", tk.END, values=r)

    # ------------------ DAILY GOALS ------------------
    def set_daily_goal(self):
        goal = tk.simpledialog.askfloat("Daily Goal", "Enter target WPM for today:")
        if goal:
            today = datetime.date.today().isoformat()
            with open(GOALS, "a", newline="") as f:
                csv.writer(f).writerow([today, goal, 0])
            messagebox.showinfo("Goal Set", f"Today's goal: {goal} WPM")

    def update_daily_goal(self, wpm):
        today = datetime.date.today().isoformat()
        rows = []

        with open(GOALS) as f:
            rows = list(csv.reader(f))

        updated = False
        for r in rows:
            if r[0] == today:
                if wpm > float(r[2]):
                    r[2] = wpm
                updated = True

        if updated:
            with open(GOALS, "w", newline="") as f:
                csv.writer(f).writerows(rows)

            messagebox.showinfo("Goal Update", f"Best WPM today: {wpm:.2f}")

# ------------------ RUN APP ------------------
root = tk.Tk()
TypingApp(root)
root.mainloop()
