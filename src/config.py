# source and target maximum lengths before sub-word tokenization
# and special tokens
src_max_seq = 40
tgt_max_seq = 14

src_vocab_size = 45000
tgt_vocab_size = 28000

lr = 9.350806868882102e-05  # from optuna
weight_decay = 0.006504135871871216  # from optuna
warmup_steps = 413  # from optuna
nth_hidden_layer = -1

batch_size = 32
model_name = "bert-base-multilingual-cased"
checkpoint_dir = "./checkpoints"
max_epoch = 7
random_seed = 111

valid_step_interval = 10
train_step_interval = 20

labels = [
    "Appeal_to_Values",
    "Loaded_Language",
    "Consequential_Oversimplification",
    "Causal_Oversimplification",
    "Questioning_the_Reputation",
    "Straw_Man",
    "Repetition",
    "Guilt_by_Association",
    "Appeal_to_Hypocrisy",
    "Conversation_Killer",
    "False_Dilemma-No_Choice",
    "Whataboutism",
    "Slogans",
    "Obfuscation-Vagueness-Confusion",
    "Name_Calling-Labeling",
    "Flag_Waving",
    "Doubt",
    "Appeal_to_Fear-Prejudice",
    "Exaggeration-Minimisation",
    "Red_Herring",
    "Appeal_to_Popularity",
    "Appeal_to_Authority",
    "Appeal_to_Time",
]

