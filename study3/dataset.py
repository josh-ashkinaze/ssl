import pandas as pd
from study3.config import *

def load_study2(filename):

    fp = DATA_DIR / filename
    df = pd.read_csv(fp, sep="\t")

    complete_df = df[df.finished]