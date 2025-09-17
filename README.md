# marketing-markov-model
This project analyzes user journeys across marketing channels with Markov Chain models, providing insights into transition probabilities, drop-offs, channel ranking, removal effects, and Monte Carlo simulations. Includes Python implementation, Jupyter notebook, and a business report (Markdown + PDF).

# Markov Chain – Channel Journey Analysis

This project analyzes marketing channel paths with an absorbing Markov Chain:
- Transition/Q/R/N/B matrices
- Channel removal effects
- Monte Carlo simulations
- Business insights + recommendations

## Repo Structure
- `src/markov_chain.py` — core functions (build matrices, simulate, removal effects)
- `src/viz.py` - Visualization functions (Heatmap, barplot, sankey diagram)
- `notebooks/markov_chain_demo.ipynb` — detailed analysis and visualizations
- `reports/markovchain_report.md` — report (Markdown, GitHub-friendly)
- `reports/markovchain_report.pdf` — styled PDF export
- `requirements.txt` — Python dependencies

## Quickstart
```bash
# 1) create env (optional)
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) run notebook
jupyter lab  # or: jupyter notebook


