# Code for L45 Project

Repository containing code for the L45 project: *Can k-NN-based graph inference methods discover the underlying causal structure?*

## Model Training / Evaluation

The model training can be executed via the bash script:

```bash
sh ./train.sh
```
This includes both the training as well as the testing of the model.

## Checkpoints and Figures

Pretrained checkpoints and figures are available for download: https://tinyurl.com/2esrfadv

To produce the figures from the trained checkpoints, use the plotting script:
```python
python plot_results.py
```

## License

MIT
