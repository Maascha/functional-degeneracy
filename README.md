# Functional Degeneracy in Neural Networks: Measurement and Pruning

Code for the paper *Functional Degeneracy in Neural Networks: Measurement and Pruning* 
(Maria Matveev, Pascal Esser, Ayush Bharadwaj, Lucius Bushnaq, Gitta Kutyniok).

## Requirements

Python 3.11 with

```
torch==2.5.1
torchvision==0.20.1
numpy==2.3.1
matplotlib==3.10.8
```

```bash
pip install -r requirements.txt
```

The teacher-student experiments can run on a CPU, the MNIST experiments need a GPU.

## Documentation

The behavioral recovery rank of a trained network $\theta^\ast$ is the smallest
number $k$ of leading eigendirections of the behavioral Hessian
$H_{\mathrm{beh}}(\theta^\ast)$ such that the projected checkpoint
$\hat\theta_k = \theta_{\mathrm{ref}} + \Pi_k(\theta^\ast - \theta_{\mathrm{ref}})$
still reaches a given performance level. The code computes this rank and compares
it with pruning baselines (structural pruning, magnitude pruning, Optimal Brain
Damage).

To run the experiments:

- `scripts/run_teacher_student.sh`: trains one-hidden-layer ReLU students on
  teacher networks over a range of student widths and seeds, computes the
  behavioral eigenpairs and the projection curves, and, for the main setting, the
  recovery rank at snapshots taken during training.
- `scripts/run_mnist.sh`: trains width-scaled MLPs on MNIST and computes the
  leading 12,000 behavioral eigenpairs and the projection curves.

To plot the results, use the plotting and analysis scripts in `src/`. Each one
documents its usage at the top:

- teacher-student: `plot_ts_rank.py` (rank vs. width with piecewise fit, heatmap),
  `plot_ts_breakpoint.py`, `plot_ts_diagnostics.py`, `plot_ts_tight_tolerances.py`,
  `plot_ts_temporal.py`, `prune_vs_recovery_rank.py`, `locality_diagnostic.py`;
- MNIST: `compare_widths.py` (curves, spectra, rank vs. width),
  `prune_vs_recovery_rank_mnist.py`.


## License

MIT, see `LICENSE`.

## Citation

If you use the code, please cite:

```bibtex
@misc{matveev2026functionaldegeneracyneuralnetworks,
      title={Functional Degeneracy in Neural Networks: Measurement and Pruning}, 
      author={Maria Matveev and Pascal Esser and Ayush Bharadwaj and Lucius Bushnaq and Gitta Kutyniok},
      year={2026},
      eprint={2608.30741},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.30741}, 
}
```
