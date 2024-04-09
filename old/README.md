# Todo
[x] simple training for single training
[x] modify for batch and support padding
[x] save the best model based on bleu on dev
[x] add beam search decoding in `decoder` step of the model
[] way to load and evaluate an existing model
[x] initiate the decoder hidden states with last step of encoder hidden states
[] ignore padding in attention
[x] make decoder lstm bidirectional
[] write detail of the attention mechanism
[] visualize attention matrices


# script
[] train:
    1. source file
    2. target file
    3. glove embedding txt path
    4. embedding_vector.pt path
    5. lstm hyperparameters (hidden, embedding, num_layers, dropout, bidir)
    6. training hyperparams (lr, clip norm)
    7. checkpoint path; loads and continues training
    8. enable neptuna

[] evaluate: prints scores
    1. source file
    2. target file
    3. checkpoint path
    4. save output flag

    
    