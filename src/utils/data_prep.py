import pandas as pd

# Collapse into uninterrupted speech
def collapse_speech(df_):
    df = df_.copy()
    df["first"] = (df["role" ].shift(1) != df["role"]).astype(int)
    df["uttID"] =  df["first"].cumsum()

    # Concatenate utterance text per uttID
    df["Speech"  ] = df["Speech"].astype(str)
    df["utt"     ] = df.groupby("uttID")["Speech"].transform(lambda x: ' '.join(x))
    df["uttWords"] = df.groupby("uttID")["Speech"].transform(lambda x: len(x))
    
    return df


def prep_speech(df):
    collapsed = collapse_speech(df)
    user_only = collapsed[(collapsed["first"] == 1) & (collapsed["role"] == "user")]["utt"].values
    user_rows = [str(row) for row in user_only if len(str(row).strip().split(" ")) > 3]
    return user_rows
