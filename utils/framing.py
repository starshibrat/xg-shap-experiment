import pandas as pd

def make_stratified_frame(
    dataframe: pd.DataFrame,
    column_name: str,
    sort_by: str="freq",
    ascending: bool=False
):
    gb = dataframe.groupby([column_name, "DIABETES"], dropna=False).size()
    desired = gb.to_frame("count")
    desired["prop"] = gb / gb.groupby(level=0, dropna=False).sum()

    desired["strat_freq"] = [f"{int(a):,}" + " (" + f"{b * 100:.2f}" + ")" for a, b in zip(desired["count"], desired["prop"])]

    desired = desired.unstack(level="DIABETES")
    desired["freq"] = desired["count"].sum(axis=1)
    respondent = desired["freq"].sum()

    desired = desired.sort_values(by=sort_by, ascending=ascending)

    desired["freq"] = desired["freq"].apply(lambda x: f"{x:,} ({(x * 100/respondent):.3f})")

    desired = desired.drop(columns=["count", "prop"])

    

    return desired[["freq", "strat_freq"]]