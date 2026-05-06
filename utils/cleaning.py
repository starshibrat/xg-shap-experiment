import polars as pl
import pandas as pd



def init(brfss_datasets_path=None):
    """
    Initializing the brfss raw annual survey datasets (2017 - 2023).
    The format used is .XPT then converted to .parquet for polars lazyframe.
    Make sure the path is correct before proceeding.
    """

    # Warning: change with your own directory
    if brfss_datasets_path is None:
        brfss_datasets_path = "/home/sakane/Documents/brfss-dataset/brfss/"

    brfss_2023_path = brfss_datasets_path + "LLCP2023XPT/LLCP2023.XPT"
    brfss_2021_path = brfss_datasets_path + "LLCP2021XPT/LLCP2021.XPT"
    brfss_2019_path = brfss_datasets_path + "LLCP2019XPT/LLCP2019.XPT"
    brfss_2017_path = brfss_datasets_path + "LLCP2017XPT/LLCP2017.XPT"

    ls = [brfss_2017_path, brfss_2019_path, brfss_2021_path,brfss_2023_path]

    for i in range(len(ls)):
        df = pd.read_sas(ls[i])
        df.to_parquet(f"../datasets/BRFSS/{2017 + i * 2}.parquet", index=False)



"""
The features being used is grouped by below list.
Reference: https://doi.org/10.1177/21501319251400546

features = [
    # Socio-demographic
    "SEXVAR",
    "_AGE_G",
    "_RACE",
    "MARITAL",
    "_EDUCAG",
    "_INCOMG1",
    "EMPLOY1",
    "_URBSTAT",

    # Health status indicator
    "_BMI5CAT",
    "_RFHLTH",
    "_PHYS14D",
    "_MENT14D",

    # ii. Disability
    "DEAF",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON"

    # iii. chronic condition
    "_DRDXAR2",
    "CVDINFR4",
    "CVDCRHD4",
    "CVDSTRK3",
    "_ASTHMS1",
    "CHCSCNC1",
    "CHCOCNC1",
    "CHCCOPD3",
    "ADDEPEV3",
    "CHCKDNY2",
    "DIABETE4",
    "_RFHYPE6",
    "_RFCHOL3",

    # Health behavior
    "_SMOKER3",
    "_RFDRHV9",
    "_TOTINDA",
    "_HCVU654",

    # Additional
    "VETERAN3",
]
"""

mp_2023 = {
    "diabetes": "DIABETE4",
    "sex": "SEXVAR",
    "income": "_INCOMG1",
    "arthritis": "_DRDXAR2",
    "skincancer": "CHCSCNC1",
    "otcancer": "CHCOCNC1",
    "emphysema": "CHCCOPD3",
    "depression": "ADDEPEV3",
    "kidney": "CHCKDNY2",
    "highbp": "_RFHYPE6",
    "highchol": "_RFCHOL3",
    "drink": "DRNKANY6",
    "insurance": "_HCVU653"
}

mp_2021 = {
    "diabetes": "DIABETE4",
    "sex": "SEXVAR",
    "income": "_INCOMG1",
    "arthritis": "_DRDXAR3",
    "skincancer": "CHCSCNCR",
    "otcancer": "CHCOCNCR",
    "emphysema": "CHCCOPD3",
    "depression": "ADDEPEV3",
    "kidney": "CHCKDNY2",
    "highbp": "_RFHYPE6",
    "highchol": "_RFCHOL3",
    "drink": "DRNKANY5",
    "insurance": "_HCVU652"
}

mp_2019 = {
    "diabetes": "DIABETE4",
    "sex": "SEXVAR",
    "income": "_INCOMG",
    "arthritis": "_DRDXAR2",
    "skincancer": "CHCSCNCR",
    "otcancer": "CHCOCNCR",
    "emphysema": "CHCCOPD2",
    "depression": "ADDEPEV3",
    "kidney": "CHCKDNY2",
    "highbp": "_RFHYPE5",
    "highchol": "_RFCHOL2",
    "drink": "DRNKANY5",
    "insurance": "_HCVU651"
}

mp_2017 = {
    "diabetes": "DIABETE3",
    "sex": "SEX",
    "income": "_INCOMG",
    "arthritis": "_DRDXAR1",
    "skincancer": "CHCSCNCR",
    "otcancer": "CHCOCNCR",
    "emphysema": "CHCCOPD1",
    "depression": "ADDEPEV2",
    "kidney": "CHCKIDNY",
    "highbp": "_RFHYPE5",
    "highchol": "_RFCHOL1",
    "drink": "DRNKANY5",
    "insurance": "_HCVU651"
}

mp_common = {
    "age": "_AGE_G",
    "race": "_RACEGR3",
    "marital": "MARITAL",
    "education": "_EDUCAG",
    "employment": "EMPLOY1",
    "bmi": "_BMI5CAT",
    "health": "_RFHLTH",
    "physical": "_PHYS14D",
    "mental": "_MENT14D",
    "deaf": "DEAF",
    "blind": "BLIND",
    "concentration": "DECIDE",
    "diffwalk": "DIFFWALK",
    "diffdres": "DIFFDRES",
    "diffalon": "DIFFALON",
    "heart": "_MICHD",
    "stroke": "CVDSTRK3",
    "asthma": "_CASTHM1",
    "smoker": "_SMOKER3",
    "exercise": "_TOTINDA",
    "veteran": "VETERAN3",
}

mp_common_missing = {
    'marital': [9, None],
    'education': [9],
    'employment': [9, None],
    'bmi': [None],
    'health': [9],
    'physical': [9],
    'mental': [9],
    'deaf': [7, 9, None],
    'blind': [7, 9, None],
    'concentration': [7, 9, None],
    'diffwalk': [7, 9, None],
    'diffdres': [7, 9, None],
    'diffalon': [7, 9, None],
    'heart': [None],
    'stroke': [7, 9, None],
    'asthma': [9],
    'smoker': [9, None],
    'exercise': [9, None],
    "diabetes": [9, None],
    "income": [9],
    "arthritis": [None],
    "skincancer": [7, 9, None],
    "otcancer": [7, 9, None],
    "emphysema": [7, 9, None],
    "depression": [7, 9, None],
    "kidney": [7, 9, None],
    "highbp": [9],
    "drink": [7, 9],
    "insurance": [9],
    "veteran": [7, 9, None],
}

mp_2017_missing = {
    "race": [9],
    "highchol": [None],
    "sex": [9],
}

mp_2019_missing = {
    "race": [9, None],
    "highchol": [9, None],
    "sex": [],
}

mp_2021_missing = {
    "race": [9],
    "highchol": [9, None],
    "sex": [],
}


mp_2023_missing = {
    "race": [9, None],
    "highchol": [9, None],
    "sex": [],
}

disability = [
    "deaf", "blind", "concentration", "diffwalk", "diffdres", "diffalon"
]

chronic_condition = [
    "heart", "stroke", "asthma",
    "emphysema", "kidney", "depression", 
]

cancer_condition = [
    "skincancer", "otcancer"
]

separate_chronic_condition = [
    "arthritis"
]

def get_map(dataset_name: int) -> dict:
    """
    Get dictionary of dataset's columns i.e. intended features to in table actual columns
    """

    match dataset_name:
        case 2023:
            return mp_2023
        case 2021:
            return mp_2021
        case 2019:
            return mp_2019

    return mp_2017

def get_missing_value(dataset_name: int, column_name: str):
    """
    Missing value takes different approach for the features. Sometimes uses 7, 9 and blank, sometimes just 9 or just blank. This function intended to get those missing value types
    """

    val = mp_common_missing.get(column_name)
    
    if val:
        return val
    
    match dataset_name:
        case 2023:
            val = mp_2023_missing.get(column_name)
        case 2021:
            val = mp_2021_missing.get(column_name)
        case 2019:
            val = mp_2019_missing.get(column_name)
        case 2017:
            val = mp_2017_missing.get(column_name)
    
    assert val is not None

    return val


def build_filter(column_name: str, op: str, value=None):
    """
    Filter used in polar's lazyframe.

    build_filter(pl.col('foo'), 'eq', 3) is in equivalent to
    pl.col('foo') == 3

    """
    assert op == "eq" or op == "in" or op == "isnull" or op == "notin" or op == "isnotnull"

    col = pl.col(column_name)
    match op:
        case "eq":
            return col == value
        case "in":
            assert isinstance(value, list)
            return col.is_in(value)
        case "notin":
            assert isinstance(value, list)
            return ~col.is_in(value)
        case "isnull":
            return col.is_null() | col.is_nan()
        case "isnotnull":
            return col.is_not_null() & col.is_not_nan()
            

        
def build_chain_filter(filters: list[pl.Expr], chain: str):
    assert chain == "|" or chain == "&"

    f = filters[0]

    for i in range(1, len(filters)):
        match chain:
            case "|":
                f |= filters[i]
            case "&":
                f &= filters[i]
                
    return f

def build_when(ft: pl.Expr, column_name: str, then_val=None, ot_val=None,alias=None, cast=pl.UInt8) -> pl.Expr:
    if ot_val is None:
        ot_val = pl.col(column_name)
    
    if alias is None:
        alias = column_name

    return pl.when(ft).then(then_val).otherwise(ot_val).alias(alias).cast(cast)

def get_value_from_universal_dict(dataset_name: int, key: str) -> str:
    val = mp_common.get(key)
    
    if val:
        return val
    
    match dataset_name:
        case 2023:
            val = mp_2023.get(key)
        case 2021:
            val = mp_2021.get(key)
        case 2019:
            val = mp_2019.get(key)
        case 2017:
            val = mp_2017.get(key)
    
    assert val is not None

    return val

def build_missing_dataset_filter(dataset_name: int, skip_columns: list[str]=[]) -> list[pl.Expr]:
    mp = get_map(dataset_name)
    used_columns = list(mp.keys()) + list(mp_common.keys())
    ls_when = []

    for c in used_columns:
        
        if c in skip_columns:
            continue

        og_col = get_value_from_universal_dict(dataset_name, c)

        missing_val = get_missing_value(dataset_name, c)
        ft = build_filter(og_col, "in", missing_val)
        when = build_when(ft, og_col)
        ls_when.append(when)
        
    return ls_when


def build_cast_column_filter(dataset_name: int, columns: list[str], cast=pl.UInt8) -> list[pl.Expr]:
    fls = []

    for c in columns:
        col = get_value_from_universal_dict(dataset_name, c)
        ft = pl.when(True).then(pl.col(col)).cast(cast)
        fls.append(ft)

    return fls

def build_diabetes_filter(lf: pl.LazyFrame, dataset_name: int) -> pl.LazyFrame:
    col = get_value_from_universal_dict(dataset_name, "diabetes")
    a = build_filter(col, "notin", [1, 2, 3, 4])
    b = build_filter(col, "isnull")
    ft = build_chain_filter([a, b], "|")

    lf = lf.remove(ft)

    ft2 = build_filter(col, "in", [1, 2])
    ft2 = build_when(ft2, col, 1, 0)

    lf = lf.with_columns(ft2)

    return lf

def build_income_filter(dataset_name: int) -> pl.Expr:
    col = get_value_from_universal_dict(dataset_name, "income")
    ft = build_filter(col, "in", [6, 7])
    ft = build_when(ft, col, 5)
    return ft

def build_rename_filter(dataset_name: int) -> dict:
    rename_mp = {}
    mp = get_map(dataset_name)


    for k, v in mp.items():
        rename_mp[v] = k.upper()
    
    for k, v in mp_common.items():
        rename_mp[v] = k.upper()
    
    rename_mp["COMORBIDITY_COUNT"] = "COMORBIDITY_COUNT"
    rename_mp["DISABILITY_COUNT"] = "DISABILITY_COUNT"
    rename_mp["CANCER_DIAGNOSIS"] = "CANCER_DIAGNOSIS"

    rename_mp = {k: v for k,v in rename_mp.items() if v.lower() not in disability}
    rename_mp = {k: v for k,v in rename_mp.items() if v.lower() not in chronic_condition}
    rename_mp = {k: v for k,v in rename_mp.items() if v.lower() not in cancer_condition}

    return rename_mp

def build_comorbidity_count_filter(lf: pl.LazyFrame, dataset_name: int) -> pl.LazyFrame:
    count_columns = []
    for d in chronic_condition:
        col_name = f"{d}_count"
        count_columns.append(col_name)

    fls = []
    for d in range(len(chronic_condition)):
        col = get_value_from_universal_dict(dataset_name, chronic_condition[d])
        ft = build_filter(col, "eq", 1)
        ft = build_when(ft, col, 1, 0, alias=count_columns[d])
        fls.append(ft)

    lf = lf.with_columns(
        fls,
    ).with_columns(
        COMORBIDITY_COUNT = pl.sum_horizontal(count_columns)
    )

    cm_cnt_col = "COMORBIDITY_COUNT"

    ct1_ft = build_filter(cm_cnt_col, "eq", 0)
    ct2_ft = build_filter(cm_cnt_col, "in", [1, 2])

    categorize_filter = pl.when(ct1_ft).then(0).when(ct2_ft).then(1).otherwise(2).alias(cm_cnt_col)

    return lf.with_columns(
        categorize_filter,
    )

def build_disability_count_filter(lf: pl.LazyFrame, dataset_name: int) -> pl.LazyFrame:
    count_columns = []
    for d in disability:
        col_name = f"{d}_count"
        count_columns.append(col_name)

    fls = []
    for d in range(len(disability)):
        col = get_value_from_universal_dict(dataset_name, disability[d])
        ft = build_filter(col, "eq", 1)
        ft = build_when(ft, col, 1, 0, alias=count_columns[d])
        fls.append(ft)

    lf = lf.with_columns(
        fls,
    ).with_columns(
        DISABILITY_COUNT = pl.sum_horizontal(count_columns)
    )

    cm_cnt_col = "DISABILITY_COUNT"

    ct1_ft = build_filter(cm_cnt_col, "eq", 0)
    ct2_ft = build_filter(cm_cnt_col, "in", [1, 2])

    categorize_filter = pl.when(ct1_ft).then(0).when(ct2_ft).then(1).otherwise(2).alias(cm_cnt_col)

    return lf.with_columns(
        categorize_filter,
    )

def build_cancer_diagnosa_filter(lf: pl.LazyFrame, dataset_name: int) -> pl.LazyFrame:
    skincancer_var = get_value_from_universal_dict(dataset_name, "skincancer")
    otcancer_var = get_value_from_universal_dict(dataset_name, "otcancer")


    sk_ext = build_filter(skincancer_var, "eq", 1)
    ot_ext = build_filter(otcancer_var, "eq", 1)

    sk_no = build_filter(skincancer_var, "eq", 2)
    ot_no = build_filter(otcancer_var, "eq", 2)

    sk_null = build_filter(skincancer_var, "isnull")
    ot_null = build_filter(otcancer_var, "isnull")

    exist_cond1 = build_chain_filter([sk_ext, ot_ext], "&")
    exist_cond2 = build_chain_filter([sk_ext, ot_no], "&")
    exist_cond3 = build_chain_filter([sk_ext, ot_null], "&")

    exist_cond4 = build_chain_filter([sk_no, ot_ext], "&")
    exist_cond5 = build_chain_filter([sk_null, ot_ext], "&")

    no_cond = build_chain_filter([sk_no, ot_no], "&")

    exist_cond = build_chain_filter([exist_cond1, exist_cond2, exist_cond3, exist_cond3, exist_cond4, exist_cond5], "|")

    ft = pl.when(exist_cond).then(1).when(no_cond).then(0).otherwise(None).alias("CANCER_DIAGNOSIS")

    return lf.with_columns(ft)

def build_marital_filter(lf: pl.LazyFrame, dataset_name: int) -> pl.LazyFrame:
    marital_col = get_value_from_universal_dict(dataset_name, "marital")

    ft1 = build_filter(marital_col, op="eq", value=1)
    ft2 = build_filter(marital_col, op="in", value=[2, 3, 4, 5, 6])

    ft = pl.when(ft1).then(1).when(ft2).then(0).otherwise(marital_col).alias(marital_col)

    return lf.with_columns(ft)

def build_employment_filter(lf: pl.LazyFrame, dataset_name: int) -> pl.LazyFrame:
    employment_col = get_value_from_universal_dict(dataset_name, "employment")

    ft1 = build_filter(employment_col, op="in", value=[1, 2])
    ft2 = build_filter(employment_col, op="in", value=[3, 4, 5, 6, 7, 8])

    ft = pl.when(ft1).then(1).when(ft2).then(0).otherwise(employment_col).alias(employment_col)

    return lf.with_columns(ft)

    
def clean_dataset(dataset_name: int):

    brfss_dataset_path = "../datasets/BRFSS/"
    brfss_path = brfss_dataset_path + f"{dataset_name}.parquet"

    lf = pl.scan_parquet(
        brfss_path,
        missing_columns='insert'
    )

    lf = build_diabetes_filter(lf, dataset_name)

    missing_filter = build_missing_dataset_filter(dataset_name, skip_columns=["age"])

    for w in missing_filter:
        lf = lf.with_columns(w)
    
    cast_col = build_cast_column_filter(dataset_name, ["age"])

    for w in cast_col:
        lf = lf.with_columns(w)

    if dataset_name == 2023 or dataset_name == 2021:
        income_filter = build_income_filter(dataset_name)
        lf = lf.with_columns(income_filter)
    
    lf = build_comorbidity_count_filter(lf, dataset_name)
    lf = build_disability_count_filter(lf, dataset_name)
    lf = build_cancer_diagnosa_filter(lf, dataset_name)
    lf = build_employment_filter(lf, dataset_name)
    lf = build_marital_filter(lf, dataset_name)
    
    rename_filter = build_rename_filter(dataset_name)

    lf = lf.rename(rename_filter)

    lf = lf.select(list(rename_filter.values()))

    return lf