"""Use a non-interactive backend before any test imports pyplot."""

import matplotlib

matplotlib.use("Agg")
