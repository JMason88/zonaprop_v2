import pandas as pd
import numpy as np
import scipy.io as spi
import scipy.sparse as sps
import sys

def build_user_item_mtrx(train_file, test_file, user_file, item_file):
    """ Build user item matrix (for test and train datasets)
    (sparse matrix, Mui[u,i] = 1 if user u has purchase item i, 0 otherwise)

    arg : week_ID (str) validation week
    """

    print("Creating user_item matrix for LightFM")

    # For now, only consider the detail dataset
    train = pd.read_pickle(train_file)
    train = train.groupby(['idusuario', 'idaviso']).count().reset_index()

    test = pd.read_pickle(test_file)

    item = pd.read_pickle(item_file)
    item = item.groupby('idaviso').count().reset_index()

    ulist = pd.read_pickle(user_file)
    ulist = ulist.groupby('idusuario').count().reset_index()
    # Build a dict with the coupon index in cpltr
    d_ci_tr = {}
    for i in range(len(item)):
        item_idx = item["idaviso"].values[i]
        d_ci_tr[item_idx] = i

    # Build a dict with the user index in ulist
    d_ui = {}
    for i in range(len(ulist)):
        user = ulist["idusuario"].values[i]
        d_ui[user] = i

    # Build the user x item matrices using scipy lil_matrix
    Mui_tr = sps.lil_matrix((len(ulist), len(item)), dtype=np.int8)

    # Now fill Mui_tr with the info from cpdtr
    for i in range(len(train)):
        sys.stdout.write(
            "\rProcessing row " + str(i) + "/ " + str(train.shape[0]))
        sys.stdout.flush()
        user = train["idusuario"].values[i]
        item_idx = train["idaviso"].values[i]
        ui, ii = d_ui[user], d_ci_tr[item_idx]
        Mui_tr[ui, ii] = 1


    # Save the matrix in the COO format

    return Mui_tr
    #spi.mmwrite(
    #    "../Data/Validation/%s/user_item_train_mtrx_%s.mtx" %
    #    (week_ID, week_ID),
    #    Mui_tr)


def build_user_feature_matrix(week_ID):
    """ Build user feature matrix
    (feat = AGE, SEX_ID, these feat are then binarized)

    arg : week_ID (str) validation week

    """

    print
    "Creating user_feature matrix for LightFM"

    def age_function(age, age_low=0, age_up=100):
        """Binarize age in age slices"""

        if age_low <= age < age_up:
            return 1
        else:
            return 0

    def format_reg_date(row):
        """Format reg date to "year-month" """

        row = row.split(" ")
        row = row[0].split("-")
        reg_date = row[0]  # + row[1]
        return reg_date

    ulist = pd.read_csv(
        "../Data/Validation/%s/user_list_validation_%s.csv" %
        (week_ID, week_ID))

    # Format REG_DATE
    ulist["REG_DATE"] = ulist["REG_DATE"].apply(format_reg_date)

    # Segment the age
    ulist["0to30"] = ulist["AGE"].apply(age_function, age_low=0, age_up=30)
    ulist["30to50"] = ulist["AGE"].apply(
        age_function,
        age_low=30,
        age_up=50)
    ulist["50to100"] = ulist["AGE"].apply(
        age_function,
        age_low=50,
        age_up=100)

    list_age_bin = [col for col in ulist.columns.values if "to" in col]
    ulist = ulist[["USER_ID_hash",
                   "PREF_NAME",
                   "SEX_ID",
                   "REG_DATE"] + list_age_bin]

    ulist = ulist.T.to_dict().values()
    vec = DictVectorizer(sparse=True)
    ulist = vec.fit_transform(ulist)
    # ulist is in csr format, make sure the type is int
    ulist = sps.csr_matrix(ulist, dtype=np.int32)

    # Save the matrix. They are already in csr format
    spi.mmwrite(
        "../Data/Validation/%s/user_feat_mtrx_%s.mtx" %
        (week_ID, week_ID), ulist)


def build_item_feature_matrix(week_ID):
    """ Build item feature matrix

    arg : week_ID (str) validation week
    """

    print "Creating item_feature matrix for LightFM"

    def binarize_function(val, val_low=0, val_up=100):
        """Function to binarize a given column in slices
        """
        if val_low <= val < val_up:
            return 1
        else:
            return 0

    # Utility to convert a date to the day of the week
    #(indexed by i in [0,1,..6])
    def get_day_of_week(row):
        """Convert to unix time. Neglect time of the day
        """
        row = row.split(" ")
        row = row[0].split("-")
        y, m, d = int(row[0]), int(row[1]), int(row[2])
        return date(y, m, d).weekday()

    # Load coupon data
    cpltr = pd.read_csv(
        "../Data/Validation/%s/coupon_list_train_validation_%s.csv" %
        (week_ID, week_ID))
    cplte = pd.read_csv(
        "../Data/Validation/%s/coupon_list_test_validation_%s.csv" %
        (week_ID, week_ID))

    cplte["DISPFROM_day"] = cplte["DISPFROM"].apply(get_day_of_week)
    cpltr["DISPFROM_day"] = cpltr["DISPFROM"].apply(get_day_of_week)
    cplte["DISPEND_day"] = cplte["DISPEND"].apply(get_day_of_week)
    cpltr["DISPEND_day"] = cpltr["DISPEND"].apply(get_day_of_week)

    cpltr["PRICE_0to50"] = cpltr["PRICE_RATE"].apply(
        binarize_function,
        val_low=0,
        val_up=30)
    cpltr["PRICE_50to70"] = cpltr["PRICE_RATE"].apply(
        binarize_function,
        val_low=50,
        val_up=70)
    cpltr["PRICE_70to100"] = cpltr["PRICE_RATE"].apply(
        binarize_function,
        val_low=70,
        val_up=100)

    cplte["PRICE_0to50"] = cplte["PRICE_RATE"].apply(
        binarize_function,
        val_low=0,
        val_up=30)
    cplte["PRICE_50to70"] = cplte["PRICE_RATE"].apply(
        binarize_function,
        val_low=50,
        val_up=51)
    cplte["PRICE_70to100"] = cplte["PRICE_RATE"].apply(
        binarize_function,
        val_low=51,
        val_up=100)

    list_quant_name = [0, 20, 40, 60, 80, 100]
    quant_step = list_quant_name[1] - list_quant_name[0]

    list_prices = cpltr["CATALOG_PRICE"].values
    list_quant = [np.percentile(list_prices, quant)
                  for quant in list_quant_name]

    for index, (quant_name, quant) in enumerate(zip(list_quant_name, list_quant)):
        if index > 0:
            cpltr["CAT_%sto%s" % (
                quant_name - quant_step,
                quant_name)] = cpltr["CATALOG_PRICE"].apply(binarize_function,
                                                            val_low=list_quant[
                                                                index -
                                                                1],
                                                            val_up=quant)
            cplte["CAT_%sto%s" % (
                quant_name - quant_step,
                quant_name)] = cplte["CATALOG_PRICE"].apply(binarize_function,
                                                            val_low=list_quant[
                                                                index -
                                                                1],
                                                            val_up=quant)

    list_prices = cpltr["DISCOUNT_PRICE"].values
    list_quant = [np.percentile(list_prices, quant)
                  for quant in list_quant_name]
    for index, (quant_name, quant) in enumerate(zip(list_quant_name, list_quant)):
        if index > 0:
            cpltr["DIS_%sto%s" % (
                quant_name - quant_step,
                quant_name)] = cpltr["DISCOUNT_PRICE"].apply(binarize_function,
                                                             val_low=list_quant[
                                                                 index -
                                                                 1],
                                                             val_up=quant)
            cplte["DIS_%sto%s" % (
                quant_name - quant_step,
                quant_name)] = cplte["DISCOUNT_PRICE"].apply(binarize_function,
                                                             val_low=list_quant[
                                                                 index -
                                                                 1],
                                                             val_up=quant)

    list_col_bin = [col for col in cplte.columns.values if "to" in col]

    # List of features
    list_feat = [
        "GENRE_NAME", "large_area_name", "small_area_name", "VALIDPERIOD", "USABLE_DATE_MON", "USABLE_DATE_TUE",
        "USABLE_DATE_WED", "USABLE_DATE_THU", "USABLE_DATE_FRI",
        "USABLE_DATE_SAT", "USABLE_DATE_SUN", "USABLE_DATE_HOLIDAY",
        "USABLE_DATE_BEFORE_HOLIDAY"] + list_col_bin

    # NA imputation
    cplte = cplte.fillna(-1)
    cpltr = cpltr.fillna(-1)

    list_col_to_str = [
        "PRICE_RATE",
        "CATALOG_PRICE",
        "DISCOUNT_PRICE",
        "DISPFROM_day",
        "DISPEND_day",
        "DISPPERIOD",
        "VALIDPERIOD"]
    cpltr[list_col_to_str] = cpltr[list_col_to_str].astype(str)
    cplte[list_col_to_str] = cplte[list_col_to_str].astype(str)

    # Reduce dataset to features of interest
    cpltr = cpltr[list_feat]
    cplte = cplte[list_feat]

    list_us = [col for col in list_feat if "USABLE" in col]
    for col in list_us:
        cpltr.loc[cpltr[col] > 0, col] = 1
        cpltr.loc[cpltr[col] < 0, col] = 0
        cplte.loc[cpltr[col] > 0, col] = 1
        cplte.loc[cpltr[col] < 0, col] = 0

    # Binarize categorical features
    cpltr = cpltr.T.to_dict().values()
    vec = DictVectorizer(sparse=True)
    cpltr = vec.fit_transform(cpltr)
    cplte = vec.transform(cplte.T.to_dict().values())

    cplte = sps.csr_matrix(cplte, dtype=np.int32)
    cpltr = sps.csr_matrix(cpltr, dtype=np.int32)

    # Save the matrix. They are already in csr format
    spi.mmwrite(
        "../Data/Validation/%s/train_item_feat_mtrx_%s.mtx" %
        (week_ID, week_ID), cpltr)
    spi.mmwrite(
        "../Data/Validation/%s/test_item_feat_mtrx_%s.mtx" %
        (week_ID, week_ID), cplte)