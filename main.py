"""Smart Delivery Route Optimizer — native Tkinter + Matplotlib ACO dashboard."""
from __future__ import annotations

import csv
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


@dataclass(frozen=True)
class ACOConfig:
    checkpoints: int = 20
    ants: int = 30
    iterations: int = 150
    alpha: float = 1.0
    beta: float = 3.0
    evaporation: float = 0.15
    seed: Optional[int] = 42


class DeliveryRoutingProblem:
    """Synthetic 100 × 100 delivery network and closed-tour calculations."""

    def __init__(self, checkpoint_count: int, seed: Optional[int] = None) -> None:
        self.checkpoint_count = checkpoint_count
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.coordinates = self.rng.uniform(0, 100, size=(checkpoint_count, 2))
        deltas = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        self.distance_matrix = np.sqrt(np.sum(deltas * deltas, axis=2))
        np.fill_diagonal(self.distance_matrix, 0.0)

    def tour_distance(self, tour: list[int]) -> float:
        if len(tour) != self.checkpoint_count or len(set(tour)) != self.checkpoint_count:
            raise ValueError("A tour must visit every checkpoint exactly once.")
        closed_tour = tour + [tour[0]]
        return float(sum(self.distance_matrix[a, b] for a, b in zip(closed_tour, closed_tour[1:])))


class AntColonyOptimizer:
    """Ant Colony Optimization for TSP; each ``step`` completes one colony iteration."""

    def __init__(self, problem: DeliveryRoutingProblem, config: ACOConfig) -> None:
        self.problem, self.config = problem, config
        self.rng = np.random.default_rng(config.seed)
        n = problem.checkpoint_count
        self.pheromone = np.ones((n, n), dtype=float)
        self.visibility = np.zeros((n, n), dtype=float)
        nonzero = problem.distance_matrix > 0
        self.visibility[nonzero] = 1.0 / problem.distance_matrix[nonzero]
        self.best_tour: Optional[list[int]] = None
        self.best_distance = math.inf
        self.best_iteration = 0
        self.iteration = 0
        self.best_history: list[float] = []
        self.average_history: list[float] = []
        # Checkpoint 0 is the visual depot and fixed tour origin. The closed
        # tour distance remains cyclically equivalent to any rotation.
        self.initial_tour = [0] + self.rng.permutation(np.arange(1, n)).tolist()
        self.initial_distance = problem.tour_distance(self.initial_tour)
        self.best_history.append(self.initial_distance)
        self.average_history.append(self.initial_distance)

    @property
    def evaluations(self) -> int:
        return self.iteration * self.config.ants

    def _construct_tour(self) -> list[int]:
        n = self.problem.checkpoint_count
        # Keep a stable delivery depot while the remaining route is still
        # probabilistically constructed by the ACO transition rule.
        start = 0
        tour, unvisited = [start], set(range(n)) - {start}
        while unvisited:
            current = tour[-1]
            candidates = np.fromiter(unvisited, dtype=int)
            weights = (self.pheromone[current, candidates] ** self.config.alpha) * (
                self.visibility[current, candidates] ** self.config.beta
            )
            total = float(weights.sum())
            if not math.isfinite(total) or total <= 0:
                next_city = int(self.rng.choice(candidates))
            else:
                next_city = int(self.rng.choice(candidates, p=weights / total))
            tour.append(next_city)
            unvisited.remove(next_city)
        return tour

    def step(self) -> None:
        if self.iteration >= self.config.iterations:
            return
        tours = [self._construct_tour() for _ in range(self.config.ants)]
        distances = np.asarray([self.problem.tour_distance(tour) for tour in tours])
        winner = int(np.argmin(distances))
        if float(distances[winner]) < self.best_distance:
            self.best_distance = float(distances[winner])
            self.best_tour = tours[winner].copy()
            self.best_iteration = self.iteration + 1
        self.pheromone *= 1.0 - self.config.evaporation
        for tour, distance in zip(tours, distances):
            deposit = 1.0 / float(distance)
            closed = tour + [tour[0]]
            for a, b in zip(closed, closed[1:]):
                self.pheromone[a, b] += deposit
                self.pheromone[b, a] += deposit
        np.fill_diagonal(self.pheromone, 0.0)
        self.iteration += 1
        self.best_history.append(self.best_distance)
        self.average_history.append(float(distances.mean()))


class MetricCard(tk.Frame):
    def __init__(self, parent: tk.Misc, title: str, accent: str) -> None:
        super().__init__(parent, bg="#15233a", highlightbackground="#27415f", highlightthickness=1)
        tk.Label(self, text=title.upper(), bg="#15233a", fg="#9eb1c9", font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=13, pady=(10, 2))
        self.value = tk.Label(self, text="—", bg="#15233a", fg=accent, font=("Segoe UI", 17, "bold"))
        self.value.pack(anchor="w", padx=13, pady=(0, 10))

    def set(self, value: str) -> None:
        self.value.configure(text=value)


class OptimizerApp(tk.Tk):
    BG, PANEL, PANEL_2, TEXT, MUTED, BLUE, CYAN, GREEN, AMBER, CORAL = (
        "#0b1220", "#101b2d", "#15233a", "#edf5ff", "#9eb1c9", "#3b9eff", "#28d7d0", "#56d789", "#f4ba52", "#ff6f6f"
    )

    def __init__(self) -> None:
        super().__init__()
        self.title("Smart Delivery Route Optimizer")
        self.geometry("1400x900")
        self.minsize(1100, 720)
        self.configure(bg=self.BG)
        self.problem: Optional[DeliveryRoutingProblem] = None
        self.optimizer: Optional[AntColonyOptimizer] = None
        self.state_name = "Ready"
        self.callback_id: Optional[str] = None
        self._make_vars()
        self._style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._update_ui()

    def _make_vars(self) -> None:
        defaults = ACOConfig()
        self.checkpoints_var = tk.StringVar(value=str(defaults.checkpoints)); self.ants_var = tk.StringVar(value=str(defaults.ants))
        self.iterations_var = tk.StringVar(value=str(defaults.iterations)); self.alpha_var = tk.StringVar(value=str(defaults.alpha))
        self.beta_var = tk.StringVar(value=str(defaults.beta)); self.evaporation_var = tk.StringVar(value=str(defaults.evaporation))
        self.seed_var = tk.StringVar(value=str(defaults.seed)); self.delay_var = tk.IntVar(value=30)
        self.status_var = tk.StringVar(value="READY")

    def _style(self) -> None:
        style = ttk.Style(self); style.theme_use("clam")
        style.configure("TFrame", background=self.BG); style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=self.PANEL, foreground=self.CYAN, font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", fieldbackground="#0c1728", foreground=self.TEXT, insertcolor=self.TEXT, bordercolor="#27415f", padding=5)
        style.configure("Primary.TButton", background=self.BLUE, foreground="#ffffff", font=("Segoe UI", 9, "bold"), borderwidth=0, padding=8)
        style.map("Primary.TButton", background=[("active", "#66b5ff"), ("disabled", "#294365")])
        style.configure("Secondary.TButton", background="#233956", foreground=self.TEXT, font=("Segoe UI", 9), borderwidth=0, padding=8)
        style.map("Secondary.TButton", background=[("active", "#314f76"), ("disabled", "#1a2a3e")])
        style.configure("Horizontal.TScale", background=self.PANEL, troughcolor="#203550")

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg="#0d192b", height=72); header.pack(fill="x"); header.pack_propagate(False)
        icon = tk.Canvas(header, width=38, height=38, bg="#0d192b", highlightthickness=0); icon.pack(side="left", padx=(20, 9), pady=16)
        icon.create_oval(5, 12, 28, 28, fill=self.BLUE, outline=""); icon.create_line(26, 10, 34, 4, fill=self.CYAN, width=2); icon.create_line(26, 29, 34, 34, fill=self.CYAN, width=2)
        title_box = tk.Frame(header, bg="#0d192b"); title_box.pack(side="left", pady=11)
        tk.Label(title_box, text="Smart Delivery Route Optimizer", bg="#0d192b", fg=self.TEXT, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(title_box, text="ANT COLONY OPTIMIZATION  |  INTELLIGENT LOGISTICS", bg="#0d192b", fg=self.MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(header, textvariable=self.status_var, bg="#173354", fg="#bfe0ff", font=("Segoe UI", 9, "bold"), padx=12, pady=5).pack(side="right", padx=(8, 18), pady=20)
        self.header_iter = tk.Label(header, text="ITERATION 0 / 150", bg="#0d192b", fg=self.MUTED, font=("Segoe UI", 9)); self.header_iter.pack(side="right", padx=10)

        body = tk.Frame(self, bg=self.BG); body.pack(fill="both", expand=True, padx=12, pady=12)
        sidebar = tk.Frame(body, bg=self.PANEL, width=268, highlightbackground="#203854", highlightthickness=1); sidebar.pack(side="left", fill="y"); sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)
        work = tk.Frame(body, bg=self.BG); work.pack(side="left", fill="both", expand=True, padx=(12, 0))
        cards = tk.Frame(work, bg=self.BG); cards.pack(fill="x", pady=(0, 10))
        self.cards = [MetricCard(cards, "Best Route Distance", self.CYAN), MetricCard(cards, "Initial Route Distance", self.TEXT), MetricCard(cards, "Distance Improvement", self.GREEN), MetricCard(cards, "Iterations Completed", self.AMBER)]
        for card in self.cards: card.pack(side="left", fill="x", expand=True, padx=(0, 8))
        charts = tk.Frame(work, bg=self.BG); charts.pack(fill="both", expand=True)
        self.figure = Figure(facecolor=self.PANEL, dpi=100); self.conv_ax = self.figure.add_subplot(121); self.map_ax = self.figure.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.figure, master=charts); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        report_box = tk.Frame(work, bg=self.PANEL, highlightbackground="#203854", highlightthickness=1); report_box.pack(fill="both", pady=(10, 0), ipady=3)
        tk.Label(report_box, text="EXECUTION REPORT", bg=self.PANEL, fg=self.CYAN, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        self.report = tk.Text(report_box, height=9, bg="#0b1627", fg="#cbd8e7", insertbackground=self.TEXT, relief="flat", wrap="word", font=("Consolas", 9), padx=10, pady=7)
        scroll = ttk.Scrollbar(report_box, command=self.report.yview); self.report.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y", padx=4, pady=4); self.report.pack(fill="both", expand=True, padx=(8, 0), pady=(0, 8)); self.report.configure(state="disabled")
        self._draw_charts()

    def _build_sidebar(self, side: tk.Frame) -> None:
        tk.Label(side, text="SIMULATION CONTROLS", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=16, pady=(15, 2))
        tk.Label(side, text="Configure and run your delivery experiment", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8)).pack(anchor="w", padx=16, pady=(0, 12))
        self._section(side, "PROBLEM CONFIGURATION")
        self._field(side, "Checkpoints (5–60)", self.checkpoints_var); self._field(side, "Random seed (optional)", self.seed_var)
        self._hint(side, "Synthetic coordinates on a 100 × 100 grid")
        self._section(side, "ACO PARAMETERS")
        self._field(side, "Number of ants (5–100)", self.ants_var); self._field(side, "Maximum iterations (10–1000)", self.iterations_var)
        self._field(side, "Pheromone importance α", self.alpha_var); self._hint(side, "Alpha: influence of successful routes")
        self._field(side, "Visibility importance β", self.beta_var); self._hint(side, "Beta: influence of distance visibility")
        self._field(side, "Evaporation rate", self.evaporation_var); self._hint(side, "Previous pheromone information fades")
        self._section(side, "SIMULATION")
        action_row = tk.Frame(side, bg=self.PANEL); action_row.pack(fill="x", padx=16, pady=(0, 6))
        self.run_btn = ttk.Button(action_row, text="Run Simulation", style="Primary.TButton", command=self.start); self.run_btn.pack(side="left", fill="x", expand=True)
        self.pause_btn = ttk.Button(action_row, text="Pause", style="Secondary.TButton", command=self.pause); self.pause_btn.pack(side="left", padx=(6, 0))
        action_row2 = tk.Frame(side, bg=self.PANEL); action_row2.pack(fill="x", padx=16, pady=(0, 6))
        self.resume_btn = ttk.Button(action_row2, text="Resume", style="Secondary.TButton", command=self.resume); self.resume_btn.pack(side="left", fill="x", expand=True)
        self.reset_btn = ttk.Button(action_row2, text="Reset", style="Secondary.TButton", command=self.reset); self.reset_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.again_btn = ttk.Button(side, text="Run Again", style="Secondary.TButton", command=self.run_again); self.again_btn.pack(fill="x", padx=16)
        self._section(side, "ANIMATION SPEED")
        speed = tk.Frame(side, bg=self.PANEL); speed.pack(fill="x", padx=16)
        ttk.Scale(speed, from_=0, to=200, variable=self.delay_var, orient="horizontal", command=lambda _: self.speed_label.configure(text=f"{self.delay_var.get()} ms")).pack(side="left", fill="x", expand=True)
        self.speed_label = tk.Label(speed, text="30 ms", bg=self.PANEL, fg=self.CYAN, font=("Segoe UI", 8, "bold"), width=7); self.speed_label.pack(side="right", padx=(6, 0))
        self._section(side, "DATA EXPORT")
        self.export_btn = ttk.Button(side, text="Export History CSV", style="Secondary.TButton", command=self.export_csv); self.export_btn.pack(fill="x", padx=16)
        self._hint(side, "Exports recorded iteration history")

    def _section(self, parent: tk.Misc, text: str) -> None:
        tk.Label(parent, text=text, bg=self.PANEL, fg=self.CYAN, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(13, 5))

    def _field(self, parent: tk.Misc, label: str, var: tk.StringVar) -> None:
        row = tk.Frame(parent, bg=self.PANEL); row.pack(fill="x", padx=16, pady=2)
        tk.Label(row, text=label, bg=self.PANEL, fg="#c8d7e8", font=("Segoe UI", 8)).pack(side="left")
        ttk.Entry(row, textvariable=var, width=8, justify="center").pack(side="right")

    def _hint(self, parent: tk.Misc, text: str) -> None:
        tk.Label(parent, text=text, bg=self.PANEL, fg="#7188a3", font=("Segoe UI", 7)).pack(anchor="w", padx=16, pady=(0, 2))

    def _config(self) -> ACOConfig:
        try:
            checkpoints, ants, iterations = int(self.checkpoints_var.get()), int(self.ants_var.get()), int(self.iterations_var.get())
            alpha, beta, evaporation = float(self.alpha_var.get()), float(self.beta_var.get()), float(self.evaporation_var.get())
            seed_text = self.seed_var.get().strip(); seed = int(seed_text) if seed_text else None
        except ValueError as exc:
            raise ValueError("Checkpoints, ants, iterations, and seed must be integers; ACO values must be numbers.") from exc
        if not 5 <= checkpoints <= 60: raise ValueError("Checkpoints must be between 5 and 60.")
        if not 5 <= ants <= 100: raise ValueError("Number of ants must be between 5 and 100.")
        if not 10 <= iterations <= 1000: raise ValueError("Maximum iterations must be between 10 and 1000.")
        if not 0.1 <= alpha <= 5.0 or not 0.1 <= beta <= 5.0: raise ValueError("Alpha and beta must be between 0.1 and 5.0.")
        if not 0.01 <= evaporation <= 0.99: raise ValueError("Evaporation must be between 0.01 and 0.99.")
        return ACOConfig(checkpoints, ants, iterations, alpha, beta, evaporation, seed)

    def _initialize(self) -> bool:
        try: config = self._config()
        except ValueError as error:
            messagebox.showerror("Invalid configuration", str(error), parent=self); return False
        self.problem = DeliveryRoutingProblem(config.checkpoints, config.seed); self.optimizer = AntColonyOptimizer(self.problem, config)
        return True

    def start(self) -> None:
        if self.state_name in ("Running", "Paused"): return
        if not self._initialize(): return
        self.state_name = "Running"; self._update_ui(); self._run_step()

    def _run_step(self) -> None:
        if self.state_name != "Running" or not self.optimizer: return
        self.callback_id = None; self.optimizer.step(); self._update_ui()
        if self.optimizer.iteration >= self.optimizer.config.iterations:
            self.state_name = "Completed"; self._update_ui(); return
        self.callback_id = self.after(self.delay_var.get(), self._run_step)

    def pause(self) -> None:
        if self.state_name == "Running":
            self.state_name = "Paused"; self._cancel_callback(); self._update_ui()

    def resume(self) -> None:
        if self.state_name == "Paused":
            self.state_name = "Running"; self._update_ui(); self._run_step()

    def reset(self) -> None:
        self._cancel_callback(); self.problem = None; self.optimizer = None; self.state_name = "Ready"; self._update_ui()

    def run_again(self) -> None:
        self.reset(); self.start()

    def _cancel_callback(self) -> None:
        if self.callback_id:
            try: self.after_cancel(self.callback_id)
            except tk.TclError: pass
            self.callback_id = None

    def _update_ui(self) -> None:
        running, paused, completed = self.state_name == "Running", self.state_name == "Paused", self.state_name == "Completed"
        self.status_var.set(self.state_name.upper()); self.run_btn.configure(state="normal" if not running and not paused else "disabled")
        self.pause_btn.configure(state="normal" if running else "disabled"); self.resume_btn.configure(state="normal" if paused else "disabled")
        self.again_btn.configure(state="normal" if completed else "disabled"); self.export_btn.configure(state="normal" if self.optimizer else "disabled")
        if self.optimizer:
            opt = self.optimizer; self.header_iter.configure(text=f"ITERATION {opt.iteration} / {opt.config.iterations}")
            best = opt.best_distance if math.isfinite(opt.best_distance) else opt.initial_distance
            improve = (opt.initial_distance - best) / opt.initial_distance * 100
            self.cards[0].set(f"{best:.2f}"); self.cards[1].set(f"{opt.initial_distance:.2f}"); self.cards[2].set(f"{improve:.1f}%"); self.cards[3].set(f"{opt.iteration} / {opt.config.iterations}")
        else:
            self.header_iter.configure(text=f"ITERATION 0 / {self.iterations_var.get()}")
            for card in self.cards: card.set("—")
        self._draw_charts(); self._write_report()

    def _draw_charts(self) -> None:
        for ax in (self.conv_ax, self.map_ax):
            ax.clear(); ax.set_facecolor("#0c1728"); ax.tick_params(colors="#9eb1c9", labelsize=8)
            for spine in ax.spines.values(): spine.set_color("#29425f")
        self.conv_ax.set_title("Route Convergence", color=self.TEXT, fontweight="bold", fontsize=10, pad=10)
        self.conv_ax.set_xlabel("Iteration", color=self.MUTED, fontsize=8); self.conv_ax.set_ylabel("Tour distance", color=self.MUTED, fontsize=8); self.conv_ax.grid(color="#243b57", alpha=.55, linewidth=.6)
        self.map_ax.set_title("Delivery Network Map", color=self.TEXT, fontweight="bold", fontsize=10, pad=10)
        self.map_ax.set_xlabel("X coordinate", color=self.MUTED, fontsize=8); self.map_ax.set_ylabel("Y coordinate", color=self.MUTED, fontsize=8); self.map_ax.grid(color="#243b57", alpha=.45, linewidth=.6)
        if self.optimizer and self.problem:
            opt, prob = self.optimizer, self.problem; xs = np.arange(len(opt.best_history))
            self.conv_ax.plot(xs, opt.best_history, color=self.CYAN, lw=2, label="Best route")
            self.conv_ax.plot(xs, opt.average_history, color=self.AMBER, lw=1.6, alpha=.9, label="Colony average")
            self.conv_ax.legend(facecolor="#14243a", edgecolor="#2c4867", labelcolor=self.TEXT, fontsize=8)
            coords = prob.coordinates; pher = opt.pheromone; upper = pher[np.triu_indices_from(pher, 1)]; threshold = np.percentile(upper, 88) if len(upper) else 0
            for a in range(prob.checkpoint_count):
                for b in range(a + 1, prob.checkpoint_count):
                    if pher[a, b] >= threshold and threshold > 0:
                        strength = min(1.0, pher[a, b] / max(threshold, 1e-9))
                        self.map_ax.plot(coords[[a, b], 0], coords[[a, b], 1], color="#ff9f43", alpha=.16 + .34 * strength, lw=.6 + strength)
            self.map_ax.scatter(coords[:, 0], coords[:, 1], s=30, c=self.CORAL, edgecolors="#ffc0c0", linewidths=.4, label="Checkpoints", zorder=3)
            self.map_ax.scatter(coords[0, 0], coords[0, 1], s=130, c="#f4ba52", marker="*", edgecolors="#fff1b5", linewidths=.7, label="Depot", zorder=5)
            if opt.best_tour:
                route = opt.best_tour + [opt.best_tour[0]]; pts = coords[route]
                self.map_ax.plot(pts[:, 0], pts[:, 1], color=self.BLUE, lw=2.2, label="Best route", zorder=2)
            self.map_ax.legend(facecolor="#14243a", edgecolor="#2c4867", labelcolor=self.TEXT, fontsize=7, loc="best")
            self.map_ax.set_xlim(-5, 105); self.map_ax.set_ylim(-5, 105)
        else:
            self.conv_ax.text(.5, .5, "Run a simulation to view convergence", color=self.MUTED, ha="center", va="center", transform=self.conv_ax.transAxes, fontsize=9)
            self.map_ax.text(.5, .5, "Delivery network will appear here", color=self.MUTED, ha="center", va="center", transform=self.map_ax.transAxes, fontsize=9)
        self.figure.tight_layout(pad=2); self.canvas.draw_idle()

    def _write_report(self) -> None:
        if not self.optimizer:
            content = "READY\n\nConfigure the synthetic delivery network and press Run Simulation. The optimizer will construct probabilistic ant tours and visualize its best discovered closed route."
        else:
            o, c = self.optimizer, self.optimizer.config; best = o.best_distance if math.isfinite(o.best_distance) else o.initial_distance
            improvement = (o.initial_distance - best) / o.initial_distance * 100
            tour = " → ".join(map(str, (o.best_tour or o.initial_tour) + [(o.best_tour or o.initial_tour)[0]]))
            verdict = "Optimization is paused; progress is preserved." if self.state_name == "Paused" else ("Optimization complete. This is the best solution discovered, not a guarantee of the global optimum." if self.state_name == "Completed" else "Optimization is running; pheromone trails reinforce shorter candidate tours.")
            content = (f"CONFIGURATION AND METRICS\nCheckpoints: {c.checkpoints} | Ants: {c.ants} | Max iterations: {c.iterations} | α: {c.alpha} | β: {c.beta} | Evaporation: {c.evaporation}\nTotal tour evaluations: {o.evaluations}\n\nCONVERGENCE SUMMARY\nInitial route distance: {o.initial_distance:.2f} | Best route distance: {best:.2f} | Best reported at iteration: {o.best_iteration or 0} | Improvement: {improvement:.2f}%\n\nBEST DELIVERY TOUR\nDepot/start checkpoint: 0\nCheckpoint sequence: {tour}\nTotal closed-tour distance: {best:.2f}\n\nOPTIMIZATION VERDICT\n{verdict}\n\nVISUALIZATION GUIDE\nThe cyan curve is the best distance found so far; amber is the colony average. Blue links show the best closed delivery route; orange links indicate strong pheromone trails.")
        self.report.configure(state="normal"); self.report.delete("1.0", "end"); self.report.insert("1.0", content); self.report.configure(state="disabled")

    def export_csv(self) -> None:
        if not self.optimizer: return
        path = filedialog.asksaveasfilename(parent=self, title="Export iteration history", defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="aco_history.csv")
        if not path: return
        try:
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file); writer.writerow(["iteration", "best_tour_distance", "colony_avg_distance"])
                writer.writerows(zip(range(len(self.optimizer.best_history)), self.optimizer.best_history, self.optimizer.average_history))
        except OSError as error: messagebox.showerror("Export failed", str(error), parent=self); return
        messagebox.showinfo("History exported", f"Saved {len(self.optimizer.best_history)} history entries to:\n{Path(path).name}", parent=self)

    def _close(self) -> None:
        self._cancel_callback(); self.destroy()


if __name__ == "__main__":
    OptimizerApp().mainloop()

