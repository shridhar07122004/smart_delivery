# Smart Delivery Route Optimizer

A native desktop dashboard that applies **Ant Colony Optimization (ACO)** to a synthetic delivery-routing Travelling Salesperson Problem. It generates checkpoints on an abstract 100 × 100 coordinate grid, searches for a short closed route, and shows live convergence and pheromone information.

The repository also includes a static project page in `public/index.html` that can be deployed to Vercel. `vercel.json` selects the static `Other` preset and serves only `public`, preventing the root-level desktop `main.py` from being treated as a serverless function. The interactive optimizer itself is a native Tkinter desktop application and runs locally; Vercel hosts the project page, not the desktop GUI.

## Objectives

- Visit every delivery checkpoint exactly once and return to the depot.
- Preserve a transparent, configurable ACO workflow suitable for project demonstrations.
- Provide live visual feedback, an execution report, and exportable history.

## Features

- Premium dark Tkinter/ttk dashboard with responsive charts.
- Configurable checkpoints, ants, iterations, alpha, beta, evaporation, seed, and animation delay.
- Start, pause, resume, reset, and run-again controls.
- Live best-route and colony-average convergence chart.
- Delivery map with depot, best closed tour, and strong pheromone trails.
- Accurate metrics, report, validation messages, reproducible seeds, and CSV export.

## How ACO works

Each ant constructs a complete candidate route. For a move from checkpoint `i` to unvisited checkpoint `j`, selection probability is proportional to:

`pheromone(i,j)^alpha × visibility(i,j)^beta`

where visibility is inverse Euclidean distance. After every colony iteration, pheromone evaporates and each ant deposits pheromone inversely proportional to its tour length. The application records the best route found so far; it does not claim a guaranteed global optimum.

- **Alpha (α):** influence of previously reinforced routes.
- **Beta (β):** influence of short-distance visibility.
- **Evaporation:** proportion of older pheromone information that fades each iteration.

## Install and launch (Windows)

```powershell
cd "C:\Users\Shridhar kalasgonda\Downloads\internship\smart_delivery_optimizer"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` for that terminal session, then activate again.

## Use

1. Set configuration values and select **Run Simulation**.
2. Use **Pause** and **Resume** without losing optimization state.
3. **Reset** clears the current engine; **Run Again** starts a fresh reproducible run using the displayed inputs.
4. Choose **Export History CSV** to save `iteration`, `best_tour_distance`, and `colony_avg_distance`, including iteration zero.

The cyan chart curve is the best discovered tour distance, while amber is the current colony average. The map’s blue loop is the best delivery tour and orange links show strong pheromone edges.

## Structure

```text
smart_delivery_optimizer/
├── main.py              # ACO engine and Tkinter dashboard
├── test_optimizer.py    # Engine invariants and reproducibility tests
├── requirements.txt
└── README.md
```

## Example use case

A dispatcher wants a demonstration route over 20 synthetic delivery locations. With seed 42, they can repeatedly compare alpha, beta, evaporation, and colony size while watching the route improve and exporting the iteration history for a report.

## Limitations and future work

Coordinates are synthetic and distances are abstract grid units—not real-world kilometres unless calibrated against geographic data. Future work could import road-network distances, permit fixed depot selection, add route constraints, or compare parameter experiments.
