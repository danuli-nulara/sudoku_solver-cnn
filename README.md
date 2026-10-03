# Neural Sudoku Solver

A convolutional neural network that solves Sudoku one cell at a time, the way a person would, combined with a small rule-checking backtracking search that catches its mistakes.

## How it works
1. **Data:** the Kaggle "Sudoku" dataset (1M puzzles with solutions). 990k puzzles for training, the last 10k held out for validation.
2. **Training signal:** for each puzzle, a random share (0 to 80%) of the blank cells is revealed from the solution, so the model sees boards at every stage of completion. The network predicts the correct digit for the still-blank cells; the loss is computed on those cells only.
3. **Model:** ResNet-style CNN, input is a 10-channel one-hot board (digits 0-9), 8 residual blocks with 128 channels, a 1x1 conv head giving 9 digit scores per cell. [N] epochs, Adam, mixed precision, trained on a Colab T4 GPU.
4. **Solving:** repeatedly fill the blank cell the model is most confident about. A digit is only accepted if it obeys Sudoku rules; if a cell has no legal digit left, the solver undoes its last move and tries the model's next best digit (model-guided backtracking).

## Results (held-out puzzles)
| Method | Fully solved |
|---|---|
| Network only (greedy, no rule checks) | 92.05% (2,000 puzzles) |
| Network + rule-checked backtracking | [your new number] ([N] puzzles) |

## Run it
```
pip install torch numpy gradio
python -c "from sudoku_solver import *; m = load_model('sudoku_net.pt'); print(solve_with_steps(m, [[0]*9]*9) is not None)"
```
The notebook `sudoku.ipynb` has the full training code; `app.py` launches the Gradio demo.

## Files
- `sudoku_solver.py`: model, rule checks, guided search
- `sudoku_net.pt`: trained weights (about 10 MB)
- `app.py`: Gradio demo (replay the solver's moves with a slider)
- `sudoku.ipynb`: data loading and training

## What I learned / limitations
- Per-cell accuracy was 99.75%, yet only 92% of puzzles were fully solved: one wrong early digit ruins the rest. This is why the rule checks and backtracking matter.
- Hard puzzles still need many undone guesses; the search is capped by a move budget.
