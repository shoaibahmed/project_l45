# Code the L45 Project

Repository containing code for the L45 project: *Can k-NN-based graph inference methods discover the underlying causal structure?*

## Execution

### Training / evaluation

The model training can be executed via the bash script:

```bash
sh ./train.sh
```
This includes both the training as well as the testing of the model.

### Results visualization

One the model is trained, all the analysis plots can be generated via the plot_results script.

```python
python plot_results.py
```

## License

MIT
