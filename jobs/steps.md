# Steps
1. Train a sentence and an answer model with `./train_barts/launch_tasks.sh sentence [output_name]` and `./train_barts/launch_tasks.sh answer [output_name]`
2. Check the rouge of the model checkpoints and select the best with `analyse_results/rouge.ipynb`. You can maybe delete a few unnecessary checkpoints.
3. Put the selected checkpoints in the `retrieve_and_decode/attempt.ipynb` script and run it.
4. Analyse the outputs of `retrieve_and_decode/attempt.ipynb` with `analyze_results/analyze_results_attempt.ipynb`