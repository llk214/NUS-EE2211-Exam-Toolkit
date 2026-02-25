# EE2211 Exam Toolkit

## Overview
Tkinter-based GUI toolkit for NUS EE2211 (Introduction to Machine Learning). All ML algorithms implemented from scratch using only NumPy — zero external ML libraries.

## Project Structure
```
EE2211_GUI.py              # Main entry point (runs gui.app.App)
EE2211_ALL.py              # Legacy single-file CLI version

gui/
  app.py                   # Main window, sidebar navigation, frame switching
  base_frame.py            # ModuleFrame base class (input pane, tabbed output)
  constants.py             # UI colors/fonts (dark sidebar theme)
  utils.py                 # parse_matrix(), parse_vector()
  widgets.py               # RoundedButton, MatrixGrid, CollapsibleSection, TreeVisualizer, ToggleSwitch
  widgets_cost.py          # Cost minimizer widgets (optional matplotlib/sympy)

  compute/                 # Pure computation — no UI dependencies
    classification.py      # Binary (sigmoid) & multiclass (softmax) logistic regression
    clustering.py          # K-Means, K-Means++, Fuzzy C-Means
    cost_minimizer.py      # Numerical gradient descent on arbitrary expressions
    gradient_descent.py    # Linear & softmax GD with iteration-by-iteration output
    neural.py              # MLP forward/backward propagation (configurable layers)
    regression.py          # OLS, Ridge, Polynomial regression + Pearson r
    tree.py                # Decision tree (Gini/Entropy/MSE), Random forest

  modules/                 # UI frames — one per ML module, each calls its compute/ counterpart
    classification.py
    clustering.py
    cost_minimizer.py
    gradient_descent.py
    neural.py
    regression.py
    tree.py

past year questions/       # Reference past papers with answers
test_past_paper.py         # Automated verification against past paper answers
```

## Dependencies
- **Required:** Python 3.7+, NumPy, Tkinter (built-in)
- **Optional:** matplotlib (cost function plotting), sympy (LaTeX rendering)
- No requirements.txt — install numpy with `pip install numpy`

## Running
```
python EE2211_GUI.py
```

## Architecture
- **Three-layer separation:** Compute (pure functions) → Modules (UI frames) → App (navigation)
- All compute functions return a standard dict: `{text, summary_text, weights, bias, iterations, tree, trees, predictions, metrics}`
- Input parsing: comma-separated rows → numpy arrays (e.g. `"1 2 3, 4 5 6"` → 2×3 matrix)
- Each UI module inherits `ModuleFrame` and overrides `run()` to call its compute function

## Compute Module Capabilities
| Module | Functions | Key Inputs |
|--------|-----------|------------|
| Regression | OLS, Ridge, Polynomial (any degree) | X, Y matrices; alpha; degree |
| Classification | Binary/multiclass logistic regression | X, y; learning rate, epochs, L2, batch type |
| Gradient Descent | Linear GD, Softmax GD | X, y, W0; learning rate, iterations |
| Clustering | K-Means, K-Means++, Fuzzy C-Means | X, initial centroids; max iter, tolerance |
| Decision Tree | Classification (Gini/Entropy), Regression (MSE), Random Forest | X, y; depth, min samples, custom thresholds |
| Neural Network | MLP with configurable layers, activations, backprop | X, Y; layer configs, loss type |
| Cost Minimizer | Numerical GD on arbitrary math expressions | Expression string, initial values, learning rate |

## Testing
```
python test_past_paper.py
```
Verifies 7 past paper questions (19 checks) against the compute modules. All should pass.

## Conventions
- Matrix input format: rows separated by commas, values by spaces (e.g. `"1 2, 3 4"`)
- Compute functions are stateless and side-effect-free
- UI uses Segoe UI font, dark sidebar (#2b2d42), red accent (#ef233c)
