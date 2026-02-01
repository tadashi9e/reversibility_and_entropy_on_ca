# Reversibility and Entropy in a Thermodynamic System Based on Langton's Ant

## langtons_ant_cupy.py

A Langton's ant simulation implemented on a GPU (using CuPy),
leveraging a lattice gas method.

Displays cell colors and plots historical entropies.

## langtons_ant_cupy_1d.py

Essentially the same as above, but with 1D rules.

- When an ant is on a white cell, change the cell color to black
  and continue moving straight.
- When an ant is on a black cell, change the cell color to white,
  reverse the ant's direction, and continue moving straight.
