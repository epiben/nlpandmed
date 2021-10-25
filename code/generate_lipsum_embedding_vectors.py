# FIX?: Fit a multivariate Normal instead of several independent Normals

from tensorflow.keras.models import load_model
from utils import create_model_dict, make_exposure_predictions

import fasttext
import numpy as np
import pandas as pd
import re
import sqlalchemy

if __name__ == "__main__":

    config = snakemake.config
    params = snakemake.params

    keras_models = create_model_dict(snakemake.input["keras_models"])
    fasttext_model = fasttext.load_model(snakemake.input["embedding_model"])

    words = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
        Suspendisse vulputate fermentum nisl quis maximus. 
        Nam et tristique ligula, sit amet gravida arcu. 
        Vestibulum et semper lectus, et ullamcorper ipsum. 
        Nam rutrum pulvinar enim non vestibulum. 
        Praesent elementum dignissim sagittis. 
        Morbi vitae tincidunt elit, egestas consectetur nulla. 
        Duis fermentum faucibus est sit amet tincidunt. 
        Ut vitae augue accumsan, fermentum est eu, consequat dolor. 
        Suspendisse eget lacus ut velit aliquam bibendum sed vel nibh. 
        Integer id eros vitae velit varius aliquet eget eget ex. 
        Nulla lacinia nunc velit, vel ultrices est tempus in. 
        Vivamus mollis nec felis in euismod. Sed in lobortis nisi. 
        Nam mollis aliquam faucibus. 
        Phasellus vestibulum finibus ex, eget pretium ex dapibus a. 
        Ut efficitur maximus ipsum, at ullamcorper velit scelerisque ac. 
        Nulla facilisi. Fusce nec aliquam tellus. 
        Donec maximus et mauris sit amet interdum. 
        Interdum et malesuada fames ac ante ipsum primis in faucibus.
        Fusce aliquet nulla ut massa ornare molestie. Phasellus at enim massa. 
        Vivamus lorem justo, mattis nec risus eu, viverra tempus augue. 
        Praesent et dapibus nunc. Vestibulum malesuada imperdiet posuere. 
        Mauris ligula nisl, imperdiet sed dictum sit amet, viverra vitae diam. 
        Phasellus tempor convallis velit ut dignissim. 
        Pellentesque congue lacus vitae molestie auctor. 
        Integer odio nisi, blandit non iaculis sed, tempor ac odio. 
        Donec fringilla elit eget turpis tristique, nec cursus neque feugiat. 
        Cras vestibulum neque quam, ac ullamcorper lorem porttitor accumsan. 
        Donec rutrum interdum mi quis porttitor. 
        Mauris dapibus tincidunt enim, sed convallis massa condimentum vel. 
        Vivamus sagittis leo id auctor sagittis. 
        Sed nunc neque, semper id nisi eu, venenatis ornare nibh. 
        Nulla molestie vulputate aliquet. Sed a laoreet dolor, ut hendrerit dolor. 
        Phasellus imperdiet vestibulum dolor. 
        Phasellus vehicula nulla nulla, non imperdiet nisi placerat nec. 
        Pellentesque porttitor consectetur sapien ac interdum. 
        Suspendisse facilisis velit et dolor auctor pellentesque. 
        Fusce magna leo, auctor et dignissim et, tincidunt quis dolor. 
        Pellentesque vel tellus pellentesque, mattis ex eget, dignissim dui. 
        Sed at eros quis ipsum vehicula bibendum. 
        Etiam in eros euismod, sodales libero sed, malesuada eros. 
        Cras ac eleifend erat. In luctus elit a nulla sodales vulputate. 
        Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.
    """

    words = [
        re.sub(r"[^\w]", " ", w).lower() for w in words.split() if len(w) >= 3
    ]

    lipsum_matrix = np.stack([fasttext_model.get_word_vector(w) for w in words])

    mu = np.apply_along_axis(np.mean, 0, lipsum_matrix)
    sigma = np.apply_along_axis(np.std, 0, lipsum_matrix)
    vectors = np.stack([
        np.random.normal(m, s, params["n_vectors"]) for m,s in zip(mu, sigma)
    ])

    vectors = {str(i): vectors[:, i] for i in range(vectors.shape[1])}

    res = pd.concat(
        make_exposure_predictions(vectors, l, m) for l,m in keras_models.items()
    )

    schema, table = config["predictions_lipsum_table"].split(".")

    engine = sqlalchemy.create_engine(f"""
        postgresql://{config["user"]}@dbserver/{config["database"]}
    """)

    res.to_sql(
        table, 
        engine, 
        schema=schema, 
        if_exists="replace", 
        index=False
    )
