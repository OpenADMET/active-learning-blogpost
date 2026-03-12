STRATEGIES = ["EI", "UCB", "Random", "Exploitation", "Exploration", "Diversity"]
STRATEGY_COLORS = {
    "EI": "#2d6a4f",
    "UCB": "#1d3557",
    "Random": "#e76f51",
    "Exploitation": "#9b2226",
    "Exploration": "#7b2d8b",
    "Diversity": "#0077b6",
}
STRATEGY_QUERY_KEYS = {
    "EI": "expected-improvement",
    "UCB": "upper-confidence-bound",
    "Random": None,
    "Exploitation": "exploitation",
    "Exploration": "max-uncertainty-reduction",
    "Diversity": None,
}
