"""
Train the tf-keras model
"""

from datetime import datetime
from itertools import repeat
from keras import backend as K
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import \
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import \
    Input, Dense, Dropout, Embedding, Reshape, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
from utils import status

import fasttext
import numpy as np
import pandas as pd
import psycopg2
import random
import re
import sys
import tensorflow as tf
import time

if 'snakemake' not in globals():
    import interactive
    print("## Warning: loading a custom snakemake object ##")
    snakemake = interactive.SnakeMakeClass()
else:
    try:
        open(snakemake.log[0], "w").write("") # reset log, maybe unnecessary
        sys.stdout = open(snakemake.log[0], "a")
        sys.stderr = open(snakemake.log[0], "a")
    except:
        pass

params = snakemake.params
config = snakemake.config
target_label = snakemake.wildcards["target_label"]
DSN = f"host=dbserver user={config['user']} dbname={config['database']}"

def build_model(input_shape, hp):

    """
    Generate a simple MLP or 1D-convoluation network model

    input_shape: int, the size of the input vector
    hp: dict of hyperparameters (from the snakemake.params now; Optuna later?)
    """

    inputs = Input(shape=input_shape)
    if hp['model_name'] == 'mlp':
        inputs_core = inputs
        core_layer = Dense
        kwargs = {
            "activation": hp["activation_function"],
            "kernel_regularizer": l2(hp["l2_penalty"]),
            "bias_regularizer": l2(hp["l2_penalty"])
        }

    elif hp['model_name'] == 'cnn':
        inputs_core = Reshape((input_shape, 1))(inputs)
        core_layer = Conv1D
        kwargs = {
            "kernel_size": hp["kernel_size"],
            "activation": hp["activation_function"],
            "kernel_regularizer": l2(hp["l2_penalty"]),
            "bias_regularizer": l2(hp["l2_penalty"])
        }

    else:
        raise NotImplementedError

    for idx_layer in range(hp["n_layers"]):
        if idx_layer == 0:
            hidden = core_layer(hp["n_nodes"],**kwargs)(inputs_core)
        else:
            hidden = core_layer(hp["n_nodes"], **kwargs)(hidden)
        # hidden = Dropout(hp["dropout_rate"])(hidden)

    if hp["n_layers"]:
        incoming = hidden
    else:
        incoming = inputs

    outputs = Dense(1, activation='sigmoid')(incoming)

    return Model(inputs=inputs, outputs=outputs)

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self, data, token_vectors, batch_size,
        do_shuffle=True, steps_per_epoch=None, balance_epoch=False
    ):

        """
        data: list of 2-element tuples
            - first element: label (0/1 coded)
            - second element: list of tokens to be embedded
        token_vectors: dict, embedding vectors for all tokens in data
        do_shuffle: boolean, shuffle data on epoch end?
        balance_epoch: boolean, conduct random undersampling of majority class on epoch end?
        remaining: self-explanatory
        """

        self.data = data
        self.token_vectors = token_vectors
        self.labels = np.asarray([row[0] for row in self.data])

        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        self.steps_per_epoch = steps_per_epoch

        self.positive_indices = np.where(self.labels == 1)[0]
        self.balance_epoch = balance_epoch
        if self.balance_epoch:
            if self.batch_size % 2:
                self.batch_size += 1
                print(f"Setting batch size to : {self.batch_size}")
        self.on_epoch_end()

    def __len__(self): # the number of batches per epoch

        num_steps = int(np.ceil(len(self.data) / self.batch_size))
        if self.balance_epoch:
            num_steps = int(np.floor(sum(self.labels)*2 / self.batch_size))

        if self.steps_per_epoch:
            return min(num_steps, self.steps_per_epoch)
        else:
            return num_steps

    def __getitem__(self, index): # generates one batch of data

        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.data))
        batch_of_data = [self.data[i] for i in self.indices[start:end]]

        return self.__extract_xy(batch_of_data)

    def __extract_xy(self, data):

        x, y = [], []
        for row in data:
            vectors = [self.token_vectors[t] for t in row[1]]
            y.extend(repeat(row[0], len(vectors)))
            x.extend(vectors)
        
        return np.stack(x), np.stack(y)

    def on_epoch_end(self): # updates indices after each epoch

        if self.balance_epoch:
            negative_indices = np.random.choice(
                np.where(self.labels == 0)[0].tolist(),
                size=self.batch_size // 2,
                replace=False
            )
            negative_indices = np.repeat(negative_indices, self.__len__())

            self.indices = np.concatenate(
                (negative_indices, self.positive_indices),
                axis=-1
            )
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.data))
            if self.do_shuffle == True:
                np.random.shuffle(self.indices)


if __name__== "__main__":

    status("Loading data")

    conn = psycopg2.connect(DSN)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                visit_key
                , CASE WHEN %s = ANY(labels) THEN 1 ELSE 0 END
                , tokens
            FROM {config['keras_table']}
            WHERE tokens IS NOT NULL;
        """, (target_label, ))
        data = [row for row in cur] # tokens stored as arrays
    conn.close()

    cases = [row for row in data if row[1] == 1]
    controls = [(x[0], x[1], x[2][:min(50, len(x[2]))]) for x in data if x[1] == 0] 
        # keep top-50 terms; if less than 50, keep all
    # controls = [row for row in train if row[1] == 0]
    
    if len(controls) > 2 * len(cases): # random 1:2 undersampling
        status("Undersampling controls")
        dev = cases + random.sample(controls, 2 * len(cases))
    else:
        dev = cases + controls
    
    # visit_keys = [row[0] for row in dev] 
    dev = [row[1:] for row in dev] # discard visit_keys
    unique_tokens = set(t for row in dev for t in row[1]) # set of flattened

    # Save visit_keys in database (for track-keeping)
    # status("Saving visit_keys in database")
    # q = f"""
    #     BEGIN;

    #     CREATE TABLE IF NOT EXISTS {config["visit_keys_table"]} (
    #         target_label TEXT
    #         , visit_keys TEXT[] -- array of text 
    #     );
    #     GRANT ALL PRIVILEGES ON {config["visit_keys_table"]} TO bth_user;

    #     DELETE FROM {config["visit_keys_table"]}
    #     WHERE target_label = %s;

    #     INSERT INTO {config["visit_keys_table"]} (target_label, visit_keys)
    #     VALUES (%s, %s);

    #     COMMIT;
    # """
    # conn = psycopg2.connect(DSN)
    # with conn.cursor() as cur:
    #     cur.execute(q, (target_label, target_label, visit_keys, ))
    # conn.close()

    print(f"Found {len(dev)} visits with data")
    n_outcome = np.sum([x[0] for x in dev])
    prop_outcome = np.round(np.mean([x[0] for x in dev]) * 100, 2)
    print(f"{n_outcome} ({prop_outcome}%) visits with label=1")
    print(f"Found {len(unique_tokens)} unique tokens")

    status("Loading fastText model")
    fasttext_model = fasttext.load_model(snakemake.input["embedding_model"])

    status("Creating dict of embedding vectors")
    token_vectors = {t: fasttext_model.get_word_vector(t) for t in unique_tokens}

    status("Building model")
    mlp = build_model(
        input_shape=fasttext_model.get_dimension(),
        hp=params
    )
    optimizer = eval(params["optimizer"])(lr=params["learning_rate"])
    mlp.compile(
        loss='binary_crossentropy', 
        optimizer=optimizer,
        metrics="AUC")
    mlp.summary()

    # Set up callbacks
    checkpoint = ModelCheckpoint(
        filepath=snakemake.output[0],
        monitor=params["metric_monitor"],
        mode=params["metric_mode"],
        verbose=params["verbose"],
        save_best_only=True,
        save_weights_only=False
    )
    callback_list = [checkpoint]

    if params["early_stopping"]:
        status("Setting up early stopping")
        earlystopping = EarlyStopping(
            monitor=params["metric_monitor"],
            mode=params["metric_mode"],
            min_delta=params["min_delta"],
            patience=params["patience"],
            verbose=True
        )
        callback_list.append(earlystopping)

    if params["reduce_lr_factor"]:
        status("Setting up LR reduction on plateau")
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor=params["metric_monitor"],
            mode=params["metric_mode"],
            min_delta=params["min_delta"],
            factor=params["reduce_lr_factor"],
            patience=params["patience"] // 2,
            verbose=True
        )
        callback_list.append(reduce_lr_on_plateau)

    status("Setting up generators")
    train, val = train_test_split(dev, train_size=0.8, random_state=42)
    
    train_generator = DataGenerator(
        data=train,
        token_vectors=token_vectors,
        batch_size=min(params["batch_size"], len(train)),
        balance_epoch=params["balance_epoch"]
    )

    val_generator = DataGenerator(
        data=val,
        token_vectors=token_vectors,
        batch_size=min(params["batch_size"], len(val)),
        balance_epoch=params["balance_epoch"]
    )

    status("Training model")
    train_hist = mlp.fit(
        x=train_generator,
        callbacks=callback_list,
        epochs=params["n_epochs"],
        validation_data=val_generator,
        verbose=params["verbose"],
        use_multiprocessing=False,
        workers=snakemake.threads
    )

    status("Evaluating performance")

    status("-- metrics")
    eval_metrics = mlp.evaluate(
        val_generator, 
        verbose=params["verbose"],
        return_dict=True
    )
    
    status("-- calibration")
    cali_generator = DataGenerator(
        data=val,
        token_vectors=token_vectors,
        batch_size=params["batch_size"],
        balance_epoch=False 
    )

    status("Fetching data for calibration evaluation")
    y_obs, y_pred = [], []  
    for X, y in cali_generator:
        y_obs.append(y)
        y_pred.append(mlp.predict(X).squeeze())

    pred_df = pd.DataFrame({
        "y_obs": np.concatenate(y_obs),
        "y_pred": np.concatenate(y_pred)
    })
    pred_df["q"] = pd.qcut( # quantile binning of predictions
        pred_df.y_pred, 
        params["calibration_n_bins"], 
        labels=False
    )

    status("Compute binned stats")
    df = pred_df.groupby("q").agg(
        mean_y_pred = ("y_pred", np.mean), # x axis of calibration curve
        mean_y_obs = ("y_obs", np.mean), # y axis of calibration curve
        n_outcomes = ("y_obs", sum), # for error bars in calibration plot
        n_tot = ("y_obs", len) # idem
    )

    status("Fit linear regression to calibration curve points")
    cali_reg = LinearRegression().fit(
        np.array(df.mean_y_pred).reshape(-1, 1), # one feature requires reshaping
        df.mean_y_obs
    )

    q = f"""
        BEGIN;

        CREATE TABLE IF NOT EXISTS {config["eval_table"]} (
            target_label TEXT
            , auroc FLOAT
            , intercept FLOAT
            , slope FLOAT
            , mean_y_pred FLOAT[] 
            , mean_y_obs FLOAT[]
            , n_outcomes INT[]
            , n_tot INT[]   
        );
        
        GRANT ALL PRIVILEGES ON {config["eval_table"]} TO bth_user;

        DELETE FROM {config["eval_table"]}
        WHERE target_label = %s;

        INSERT INTO {config["eval_table"]}
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);

        COMMIT;
    """

    q_args = (
        target_label, 
        target_label,
        eval_metrics["auc"],
        float(cali_reg.intercept_),
        float(cali_reg.coef_),
        df.mean_y_pred.tolist(),
        df.mean_y_obs.tolist(),
        df.n_outcomes.tolist(),
        df.n_tot.tolist()
    )

    status("Writing performance summary stats to database")
    # Sometimes scripts running in parallel try to GRANT PRIVILEGS simultaneously
    # causing all except the first-comer to fail. This is handled here. 
    attempts = 1
    max_attempts = 10
    random.seed(hash(target_label)) # ensure dt's differ across parallel jobs
    while True:
        try:
            psycopg2.connect(DSN).cursor().execute(q, q_args)
            status(f"Saved metrics after {attempts} attempt(s)")
            break
        except:
            if attempts < max_attempts:
                dt = random.uniform(1, 10) # delta time
                print(f"Couldn't save performance metrics in database; retrying in {dt} secs.")
                time.sleep(dt)
            else:
                raise Exception(f"Couldn't save training summary to database after {max_attempts} attempts")
        finally:
            attempts += 1

    status("Writing output monitored by snakemake")
    open(snakemake.output[1], "w").write(str(datetime.now()))

    status("Done")
