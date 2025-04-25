"""This file provides functions to read the seperate datasets, store them in a list and integrate them into three seperate datasets."""


# necessary imports
import pandas as pd
import glob
import os
import csv
import re




def df_resample(df_org, timestamp, frequency, mode, fillid, value):
    """This function is used to resample the dataset to the same frequency.
    Parameters:
   - original dataset, 
   - the name of the timestamp column 
   - the desired frequency
   - the mode which can be glucose or vitals, since heart-rate data does not need to be fillforward PtID
   - fillid which is the name of the column which needs to be filled with fillforward method to due produced missing values.
   - value, is the column name which should be converted into integer values (either glucose or heartrate)
    Output: It returns the resampled dataframe with no missing values in the PtID column.
   """ 
    df = df_org.copy()
    # if mode is glucose first timestamps are sorted then rounded and resampled by desired frequency
    if mode == "glucose":
        df = df.sort_values(by=timestamp)
        df[timestamp] = df[timestamp].dt.round(frequency)
        # rounding can induce duplicates which need to be removed
        df = df.drop_duplicates(subset=[timestamp])
        df = df.set_index(timestamp)
        df = df.resample(frequency).asfreq()
        df = df.reset_index()
        # missing which occured due to resampling but are essential are feedforward filled 
        df[fillid] = df[fillid].fillna(method="ffill")
    # if mode is virals, the timestamps are only rounded to the desired frequency
    elif mode == "vitals":
        df[timestamp] = df[timestamp].dt.round(frequency)
        df = df.drop_duplicates(subset=[timestamp])
    # warning if wrong input is given
    else:
        print("Mode can be either glucose or vitals")
    # glucose or heartrate columns should be converted to numericals
    df[value] = pd.to_numeric(df[value], errors="coerce")
    return df


def fill_gaps_sampling(df, timestamp, subject_id, glucose, fillmin=15):
    """
    This function feedforward filled missing glucose values which only occured due to undersampling
    Parameters:
    - the original df
    - timestamp, is the name of the column with the timestamps
    - subjects id, is the name of the column with the subject ids
    - glucose, is the name of the column with the glucose measurements
    - fillmin is the frequency of the dataset, by default it is set to 15 minutes
    """ 
    test = df.sort_values(timestamp).copy()
    # first the true glucose measurements are identified which only have a time difference of the dataset's frequency
    # so that consecutive measurements are identified 
    test["time_diff"] = test[timestamp].diff()
    test["gap"] = test["time_diff"] > pd.Timedelta(minutes=fillmin)
    test["group_id"] = test["gap"].cumsum()

    # Resample within each group
    resampled_groups = []

    # for each group of consecutive measurements, the timestamp is undersampled to 5 minutes and occuring gaps are fillforward filled
    for _, group in test.groupby("group_id"):
        group_resampled = df_resample(group, timestamp = timestamp, frequency= "5min", mode="glucose", fillid = subject_id, value = glucose)
        group_resampled[glucose] = group_resampled[glucose].ffill()
        resampled_groups.append(group_resampled)

    # All groups are concatenated to have one dataframe
    final_df = pd.concat(resampled_groups)
    # Finally, the whole dataset is resampled in 5 minute intervals 
    final_df = df_resample(final_df, timestamp = timestamp, frequency= "5min", mode="glucose", fillid = subject_id, value = glucose)
    return final_df.reset_index()



def detect_best_separator(file_path, sample_size=10000):
    """ function finding the best seperator to read the data automatically
    Parameter: file_path
    Output: best found seperator
    """
    # possible encodings
    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin1"]
    # possible delimeters
    delimiters = [",", ";", "\t", "|"]

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                sample = f.read(sample_size)
                # If UTF-16, skip sniffing and just count delimiters
                if enc == "utf-16":
                    counts = {d: sample.count(d) for d in delimiters}
                    best_guess = max(counts, key=counts.get)
                    if counts[best_guess] > 0:
                        return best_guess
                    continue
                try:
                    # try with Sniffer function 
                    dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
                    return dialect.delimiter
                except csv.Error:
                    # fallback: use count-based detection
                    counts = {d: sample.count(d) for d in delimiters}
                    best_guess = max(counts, key=counts.get)
                    if counts[best_guess] > 0:
                        return best_guess
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not detect delimiter or unsupported encoding.")



def smart_read(file_path, skip = 0):
    """
    Reads a file into a pandas DataFrame based on its extension.
    Automatically detects delimiter for .csv and .txt files from function "detect_best_separator()".
    Parameter: file_path
    Output: pandas dataframe
    """
    
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext in [".csv", ".txt"]:
        sep = detect_best_separator(file_path)
        try:
            df = pd.read_csv(file_path, sep=sep, engine="python", on_bad_lines="skip", skiprows=skip)
        except:
            df = pd.read_csv(file_path, sep=sep, engine="python", on_bad_lines="skip", encoding="utf-16", skiprows=skip)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    

    return df



def detect_sample_rate(df, time_col=None, expected_rates=(5, 10, 15)):
    """
    This functions detects the sample rate of the glucose measurements.
    A frequency of 5, 10, and 15 minutes are allowed which is typical for CGM devices
    Parameter: original dataframe as df
    Output: Most common frequency
    """
    # Extract time column or index
    times = pd.to_datetime(df[time_col]) if time_col else pd.to_datetime(df.index)
    times = times.sort_values()

    # Compute time differences in minutes
    deltas = times.diff().dropna().dt.total_seconds() / 60  # in minutes

    # Round to nearest whole minute
    deltas_rounded = deltas.round().astype(int)

    # Find the most common interval
    most_common = deltas_rounded.value_counts().idxmax()

    # Match to known expected rates
    if most_common in expected_rates:
        return f"{most_common} min"
    else:
        return "Unknown"
    


def read_data(read_all = True):
    """
    This function reads the datasets individually and returns them as a list of dataframes
    Parameter: read_all can be True or False, decides which set of datasets should be read 
    Output: returns a list of dataframes
    """
    def df_granada():
        # Read the CGM measurements
        df_granada = smart_read("datasets for T1D/granada/T1DiabetesGranada/glucose_measurements.csv")
        # Rename column names for semantic equality
        df_granada = df_granada.rename(columns={"Measurement": "GlucoseCGM", "Patient_ID": "PtID"})

        # All datasets should keep the same format of date and time -> combine both and convert to datetime
        df_granada["ts"] = pd.to_datetime(df_granada["Measurement_date"] + " " + df_granada["Measurement_time"])
        # Resample to 5 minute intervals to have unifrom sample rate and enable missing values count, also forward fill gaps due to undersampling
        # This is done for each subject seperately
        df_granada = df_granada.groupby("PtID", group_keys=False).apply(lambda x: fill_gaps_sampling(x, "ts", "PtID", "GlucoseCGM"))

        # Read demographics and merge the date with the CGM measurements
        df_granada_info = smart_read("datasets for T1D/granada/T1DiabetesGranada/Patient_info.csv")
        df_granada_info["Initial_measurement_date_date"] = pd.to_datetime(df_granada_info["Initial_measurement_date"])
        df_granada_info = df_granada_info[["Sex", "Birth_year", "Patient_ID"]]
        df_granada_info = df_granada_info.rename(columns={"Patient_ID": "PtID"})
        df_granada = pd.merge(df_granada, df_granada_info, on="PtID", how="inner")
        # Count the age based on the birth year and the datetime of measurement
        df_granada["Age"] = df_granada["ts"].dt.year - df_granada["Birth_year"]
        # Remove Birthyear
        df_granada = df_granada.drop(["Birth_year"], axis=1)

        # Rename the patinet id so that it can be identified to which initial dataset the subjects belong to
        df_granada["PtID"] = df_granada["PtID"].astype(str) + "_T1DGranada"
        # Add a "Database" column with the name of the Dataset
        df_granada["Database"] = "T1DGranada"
        return df_granada

    def df_diatrend():
        # Path to the single CGM datasets of each subject
        file_paths_diatrend = glob.glob("datasets for T1D/Diatrend/DiaTrendAll/*.xlsx") 

        # Initialize an empty list to store DataFrames
        df_list_diatrend = []

        # Loop through each file, reading it into a DataFrame and assigning a PtID
        for idx, file in enumerate(file_paths_diatrend, start=1):
            df = smart_read(file)  
            # extract the PtID from the filename
            name = os.path.splitext(os.path.basename(file))[0] 
            df["PtID"] = str(name)
            df_list_diatrend.append(df)

        # Concatenate all DataFrames into one
        df_diatrend = pd.concat(df_list_diatrend, ignore_index=True)

        # Rename column names for semantic equality
        df_diatrend = df_diatrend.rename(columns={"mg/dl": "GlucoseCGM"})
        # Resample to 5 minute intervals in order to count the missing values and generate time series data
        df_diatrend["ts"] = pd.to_datetime(df_diatrend["date"], format="%Y-%m-%d %H:%M:%S")
        # This is done for each subject seperately
        df_diatrend = df_diatrend.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # include age, gender, and race 
        df_diatrend_info = smart_read("datasets for T1D/Diatrend/SubjectDemographics_3-15-23.xlsx") 
        df_diatrend_info["Sex"] = df_diatrend_info["Gender"].replace({"Male": "M", "Female": "F"})
        df_diatrend_info["PtID"] = "Subject" + df_diatrend_info["Subject"].astype(str)
        df_diatrend_info = df_diatrend_info[["Sex", "Age","PtID"]]
        df_diatrend = pd.merge(df_diatrend, df_diatrend_info, on="PtID", how="inner")
        # add the dataset name so that we can identify from which database after integrating
        df_diatrend["PtID"] = df_diatrend["PtID"].astype(str) + "_DiaTrend"
        df_diatrend["Database"] = "DiaTrend"

        return df_diatrend

    def df_city():
        # read the CGM data and rename column 
        df_city = smart_read("datasets for T1D/CITYPublicDataset/Data Tables/DeviceCGM.txt")
        # rename columns for semantic equality
        df_city = df_city.rename(columns={"Value": "GlucoseCGM"})

        # differentiate between CGM glucose and finger prick glucose
        df_city["mGLC"] = df_city["GlucoseCGM"].where(df_city["RecordType"] == "Calibration")
        df_city["GlucoseCGM"] = df_city["GlucoseCGM"].where(df_city["RecordType"] == "CGM")

        #  convert timestamp to datetime
        df_city["ts"] = pd.to_datetime(df_city["DeviceDtTm"])
        # resample to 5 minutes, this is done for each subject seperately
        df_city = df_city.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # read sex data, 
        df_city_screen = smart_read("datasets for T1D/CITYPublicDataset/Data Tables/DiabScreening.txt")
        df_city_screen = df_city_screen[["PtID", "Sex"]]

        # read age data
        df_city_age = smart_read("datasets for T1D/CITYPublicDataset/Data Tables/PtRoster.txt")
        df_city_age = df_city_age[["PtID", "AgeAsOfEnrollDt"]]

        # merge age and sex data
        df_city_info = pd.merge(df_city_screen, df_city_age, on=["PtID"], how="inner")

        # merge demorgaphics with cgm data
        df_city = pd.merge(df_city, df_city_info, on=["PtID"], how="left")
        # rename columns
        df_city = df_city.rename(columns={"AgeAtEnrollment" : "Age"})
        # add the database name to the PtID 
        df_city["PtID"] = df_city["PtID"].astype(str) + "_CITY"
        # add a Database column holding the database name 
        df_city["Database"] = "CITY"

        return df_city

    def df_dclp():
        # This dataset has two different CGM records which are read seperately and then concatenated based on the Patient ID 
        df_dclp = smart_read("datasets for T1D/DCLP3/Data Files/DexcomClarityCGM_a.txt")
        df_dclp = df_dclp[["PtID", "RecID", "DataDtTm", "CGM", "DataDtTm_adj"]]

        df_dclp_other = smart_read("datasets for T1D/DCLP3/Data Files/OtherCGM_a.txt")
        df_dclp_other = df_dclp_other[["PtID", "RecID", "DataDtTm", "CGM", "DataDtTm_adjusted"]]

        # rename column name so that both databases have an uniform schema
        df_dclp_other.rename(columns= {"DataDtTm_adjusted": "DataDtTm_adj"})
        # merge both dataframes
        df_DCLP = pd.concat([df_dclp, df_dclp_other])

        # convert timestamps to datetime
        df_DCLP["ts"] = pd.to_datetime(df_DCLP["DataDtTm"])
        # resample to 5 minutes. This is done for each subject seperately
        df_DCLP = df_DCLP.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "CGM"))

        # read demographics and merge with CGM data
        df_dclp_screen = smart_read("datasets for T1D/DCLP3/Data Files/DiabScreening_a.txt")
        df_dclp_screen = df_dclp_screen[["PtID", "AgeAtEnrollment",	"Gender"]]
        df_DCLP = pd.merge(df_DCLP, df_dclp_screen, on=["PtID"], how="left")

        # rename column names for semantic equality
        df_DCLP = df_DCLP.rename(columns={"CGM": "GlucoseCGM", "Gender" : "Sex", "AgeAtEnrollment" : "Age", "HbA1cTestRes" : "Hba1c"})
        # add the database name to the PtID
        df_DCLP["PtID"] = df_DCLP["PtID"].astype(str) + "_DLCP3"
        # add a Database column holding the database name
        df_DCLP["Database"] = "DLCP3"

        return df_DCLP
    
    def df_hupa():
        # Path to the base directory
        base_path_hupa = "datasets for T1D/HUPA-UCM/Raw_Data"

        # List to store all dataframes
        all_data_HR_h = []
        all_data_GLC_h = []


        # Loop through each subject directory (e.g., 001, 002, 003)
        for subject_id_h in os.listdir(base_path_hupa):

            # read all files including CGM readings
            person_path_h = os.path.join(base_path_hupa, subject_id_h)
            if not os.path.isdir(person_path_h):
                continue
            # combine the path name of the person and the folder
            for folder_h in os.listdir(person_path_h):
                folder_path_h = os.path.join(person_path_h, folder_h)

                if not os.path.isdir(folder_path_h):
                    continue
                
                for file in os.listdir(folder_path_h):
                    
                    if not os.path.isdir(folder_path_h):
                        continue
                    
                    # if file contains heart and is a csv file, read the file
                    if file.endswith(".csv") and "heart" in file:
                        file_path_hr_h = os.path.join(folder_path_h, file)

                        try:
                            hupa_hr = smart_read(file_path_hr_h)
                            # add a PtID
                            hupa_hr["PtID"] = subject_id_h
                            # extract the date from the filename
                            match = re.search(r"([\d]{4}-[\d]{2}-[\d]{2})", file_path_hr_h)
                            if match:
                                date_str = match.group(1)
                                hupa_hr["date"] = date_str
                                hupa_hr["Time"] = hupa_hr["Time"].astype(str)
                                # combine the date with the time
                                hupa_hr["ts"] = pd.to_datetime(hupa_hr["date"] + " " + hupa_hr["Time"])
                                # rename column names for semantic equality 
                                hupa_hr = hupa_hr.rename(columns={"Heart Rate" : "HR"})
                                hupa_hr = hupa_hr[["ts", "PtID", "HR"]]
                            else:
                                print("No date found in the file path.")
                            # add all files containing HR to the same list
                            all_data_HR_h.append(hupa_hr)

                        except Exception as e:
                            print(f"Failed to read {file_path_hr_h}: {e}")

                    # if file contains free style sensor and is a csv file, read the file
                    # this contains CGM measurements in 15 minute intervals
                    elif file.endswith(".csv") and "free_style_sensor" in file:
                        file_path_glc_h = os.path.join(folder_path_h, file)
                        # subjects 25-28 have a different schema , thus these are loaded differently
                        if re.search(r"2[5-8]P", subject_id_h):
                            try:
                                hupa_glc = smart_read(file_path_glc_h, skip=2) 
                                hupa_glc["PtID"] = subject_id_h
                                # column names are renamed for semantic equality
                                hupa_glc = hupa_glc.rename(columns={"Sello de tiempo del dispositivo": "ts", "Historial de glucosa mg/dL" : "Historic Glucose", 
                                        "Escaneo de glucosa mg/dL": "Scan Glucose", "Tira reactiva para glucosa mg/dL": "MGlucose"})
                                # timestamp is converted to datetime
                                hupa_glc["ts"] = pd.to_datetime(hupa_glc["ts"], format="mixed", dayfirst=True)
                                # histroic glucose is replaced with aligning scan glucose
                                hupa_glc["GlucoseCGM"] = hupa_glc["Scan Glucose"].where(hupa_glc["Scan Glucose"].notna(), hupa_glc["Historic Glucose"])
                                hupa_glc = hupa_glc[["ts", "PtID", "GlucoseCGM"]]
                                # undersampled to 5 minute intervals 
                                hupa_glc = fill_gaps_sampling(hupa_glc, "ts", "PtID", "GlucoseCGM")
                                
                                # add all glucose data into one list 
                                all_data_GLC_h.append(hupa_glc)

                            except Exception as e:
                                print(f"Failed to read {file_path_glc_h}: {e}")
                        else: 
                            # read remaining subjects
                            try: 
                                hupa_glc = smart_read(file_path_glc_h, skip=1) 
                                hupa_glc["PtID"] = subject_id_h
                                # rename columns
                                hupa_glc = hupa_glc.rename(columns={"Hora": "ts", "Histórico glucosa (mg/dL)" : "Historic Glucose", 
                                        "Glucosa leída (mg/dL)": "Scan Glucose", "Glucosa de la tira (mg/dL)": "MGlucose"})
                                # convert timestamp to datetime 
                                hupa_glc["ts"] = pd.to_datetime(hupa_glc["ts"], format="mixed", dayfirst=True)
                                # replace historic glucose with scan glucose
                                hupa_glc["GlucoseCGM"] = hupa_glc["Scan Glucose"].where(hupa_glc["Scan Glucose"].notna(), hupa_glc["Historic Glucose"])
                                hupa_glc = hupa_glc[["ts", "PtID", "GlucoseCGM"]]
                                # undersample to 5 minutes
                                hupa_glc = fill_gaps_sampling(hupa_glc, "ts", "PtID", "GlucoseCGM")
                                
                                # add the glucose file to a list of glucose files
                                all_data_GLC_h.append(hupa_glc)
                            except Exception as e:
                                print(f"Failed to read {file_path_glc_h}: {e}") 
                    # if file contains dexcom and is a csv file, read the file               
                    elif file.endswith(".csv") and "dexcom" in file:
                        file_path_glc_h = os.path.join(folder_path_h, file)
                        try:
                            hupa_glc_d = smart_read(file_path_glc_h)
                            hupa_glc_d["PtID"] = subject_id_h
                            # rename columns for semantic equality
                            hupa_glc_d = hupa_glc_d .rename(columns={"Marca temporal (AAAA-MM-DDThh:mm:ss)" : "ts", "Tipo de evento": "Type", "Nivel de glucosa (mg/dl)": "GlucoseCGM"})
                            # keep only eventtype = Niveles estimados de glucosa
                            hupa_glc_d = hupa_glc_d [hupa_glc_d ["Type"] == "Niveles estimados de glucosa"][["ts", "PtID", "GlucoseCGM"]]
                            hupa_glc_d["ts"] = pd.to_datetime(hupa_glc_d["ts"]) 
                            # add dataframe to list of glucose dataframes
                            all_data_GLC_h.append(hupa_glc_d)

                        except Exception as e:
                            print(f"Failed to read {file_path_glc_h}: {e}")


        # Concatenate all HR DataFrames into one and resample to 5 mintues
        df_hupa_HR = pd.concat(all_data_HR_h, ignore_index=True)
        df_hupa_HR = df_hupa_HR.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "PtID", value = "HR"))

        # Concatenate all glucose DataFrames into one and resample to 5 mintues
        df_hupa_GLC = pd.concat(all_data_GLC_h, ignore_index=True)
        df_hupa_GLC = df_hupa_GLC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency = "5min", mode = "glucose", fillid = "PtID", value = "GlucoseCGM"))

        # merge HR, and GLC
        df_HUPA = pd.merge(df_hupa_GLC, df_hupa_HR, on=["PtID", "ts"], how="left")
        # add database name to PtID
        df_HUPA["PtID"] = df_HUPA["PtID"] + "_HUPA-UCM"
        # add Database column including the name of the dataset
        df_HUPA["Database"] = "HUPA-UCM"

        return df_HUPA
    

    def df_pedap():
        # read both CGM data from different sources and concatenate them
        df_pedap= smart_read("datasets for T1D/PEDAP/Data Files/PEDAPDexcomClarityCGM.txt")
        df_pedap = df_pedap[["PtID", "RecID", "DeviceDtTm", "CGM"]]

        df_pedap_other = smart_read("datasets for T1D/PEDAP/Data Files/PEDAPOtherCGM.txt")
        df_pedap_other = df_pedap_other[["PtID", "RecID", "DeviceDtTm", "CGM"]]
        df_PEDAP = pd.concat([df_pedap, df_pedap_other])

        # convert timestamps to datetime
        df_PEDAP["ts"] = pd.to_datetime(df_PEDAP["DeviceDtTm"], format = "mixed")
        # resample to 5 minutes for each subject seperately
        df_PEDAP = df_PEDAP.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "CGM"))

        # read sex
        df_PEDAP_screen = smart_read("datasets for T1D/PEDAP/Data Files/PEDAPDiabScreening.txt")
        df_PEDAP_screen = df_PEDAP_screen[["PtID", "Sex"]]

        # read age
        df_PEDAP_age = smart_read("datasets for T1D/PEDAP/Data Files/PtRoster.txt")
        df_PEDAP_age = df_PEDAP_age[["PtID", "AgeAsofEnrollDt"]]

        # merge age and sex
        df_PEDAP_screen = pd.merge(df_PEDAP_screen, df_PEDAP_age, on="PtID", how="inner")
        # merge demographics with CGM
        df_PEDAP = pd.merge(df_PEDAP, df_PEDAP_screen, on=["PtID"], how="left")

        # rename columns for semantic equality
        df_PEDAP = df_PEDAP.rename(columns={"CGM": "GlucoseCGM", "AgeAsofEnrollDt" : "Age"})
        df_PEDAP["PtID"] = df_PEDAP["PtID"].astype(str) + "_PEDAP"
        df_PEDAP["Database"] = "PEDAP"

        return df_PEDAP

    def df_replace():
        # read CGM data
        df_RBG = smart_read("datasets for T1D/ReplaceBG/Data Tables/HDeviceCGM.txt")

        # split CGM and finger prick glucose levels
        df_RBG = df_RBG.rename(columns={"GlucoseValue": "GlucoseCGM"})
        df_RBG["mGLC"] = df_RBG["GlucoseCGM"].where(df_RBG["RecordType"] == "Calibration")
        df_RBG["GlucoseCGM"] = df_RBG["GlucoseCGM"].where(df_RBG["RecordType"] == "CGM")

        # initial date 
        df_RBG["initdate"] = pd.to_datetime("2024-01-01")
        df_RBG["datetime"] = df_RBG["initdate"] + pd.to_timedelta(df_RBG["DeviceDtTmDaysFromEnroll"], unit="D")
        df_RBG["DeviceTm"] = df_RBG["DeviceTm"].astype(str)

        # Combine date and time columns correctly
        df_RBG["ts"] = pd.to_datetime(df_RBG["datetime"].dt.strftime("%Y-%m-%d") + " " + df_RBG["DeviceTm"])

        # resample to 5 minutes
        df_RBG = df_RBG.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # read sex data
        df_RBG_screen = smart_read("datasets for T1D/REPLACEBG/Data Tables/HScreening.txt")
        df_RBG_screen = df_RBG_screen[["PtID", "Gender"]]

        # read age data
        df_RBG_age = smart_read("datasets for T1D/REPLACEBG/Data Tables/HPtRoster.txt")
        df_RBG_age = df_RBG_age[["PtID", "AgeAsOfEnrollDt"]]

        # merge sex and age data
        df_RBG_screen = pd.merge(df_RBG_screen, df_RBG_age, on="PtID", how="inner")

        # merge demographics with cgm data
        df_RBG = pd.merge(df_RBG, df_RBG_screen, on=["PtID"], how="left")

        # rename columns for semantic equality
        df_RBG_screen = df_RBG_screen.rename(columns={"AgeAsOfEnrollDt" : "Age", "Gender": "Sex"})
        # add database to PtID
        df_RBG["PtID"] = df_RBG["PtID"].astype(str) + "_RBG"
        # add database column holding the database name
        df_RBG["Database"] = "RBG"

        return df_RBG

    def df_sence():
        # read CGM data
        df_SENCE = smart_read("datasets for T1D/SENCE/Data Tables/DeviceCGM.txt")

        # convert timestamp to datetime column 
        df_SENCE["ts"] = pd.to_datetime(df_SENCE["DeviceDtTm"])
        # resample to 5 minutes. This is done for each subject individually
        df_SENCE = df_SENCE.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "Value"))

        # read sex
        df_SENCE_screen = smart_read("datasets for T1D/SENCE/Data Tables/DiabScreening.txt")
        df_SENCE_screen = df_SENCE_screen[["PtID", "Gender"]]

        # read age
        df_SENCE_age = smart_read("datasets for T1D/SENCE/Data Tables/PtRoster.txt")
        df_SENCE_age = df_SENCE_age[["PtID", "AgeAsOfEnrollDt", "EnrollDt"]]

        # merge age and sex    
        df_SENCE_screen = pd.merge(df_SENCE_screen, df_SENCE_age, on="PtID", how="inner")

        # merge demographisc with CGM 
        df_SENCE = pd.merge(df_SENCE, df_SENCE_screen, on=["PtID"], how="left")

        # reanme columns 
        df_SENCE = df_SENCE.rename(columns={"Value": "GlucoseCGM", "AgeAsOfEnrollDt" : "Age", "HbA1cTestRes": "Hba1c", "Gender": "Sex"})
        # add database name to PtID
        df_SENCE["PtID"] = df_SENCE["PtID"].astype(str) + "_SENCE"
        # add database column holding the database name
        df_SENCE["Database"] = "SENCE"

        return df_SENCE

    def df_shd():
        # read CGM data
        df_SHD = smart_read("datasets for T1D/SevereHypoDataset/Data Tables/BDataCGM.txt")

        # initial date 
        df_SHD["initdate"] = pd.to_datetime("2023-01-01")
        df_SHD["datetime"] = df_SHD["initdate"] + pd.to_timedelta(df_SHD["DeviceDaysFromEnroll"], unit="D")
        df_SHD["DeviceTm"] = df_SHD["DeviceTm"].astype(str)

        # Combine date and time columns correctly
        df_SHD["ts"] = pd.to_datetime(df_SHD["datetime"].dt.strftime("%Y-%m-%d") + " " + df_SHD["DeviceTm"])

        # reasmple to 5 minutes
        df_SHD = df_SHD.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "Glucose"))

        # read sex
        df_SHD_screen = smart_read("datasets for T1D/SevereHypoDataset/Data Tables/BDemoLifeDiabHxMgmt.txt")
        df_SHD_screen = df_SHD_screen[["PtID", "Gender"]]
        # true age is not given
        df_SHD_screen["Age"] = "60-100" 

        # merge with CGM data
        df_SHD = pd.merge(df_SHD, df_SHD_screen, on="PtID", how="left")

        # rename columns for semantic equality
        df_SHD = df_SHD.rename(columns={"Glucose": "GlucoseCGM", "Gender": "Sex"})
        df_SHD["PtID"] = df_SHD["PtID"].astype(str) + "_SHD"
        df_SHD["Database"] = "SHD"

        return df_SHD

    def df_wisdm():
        # read CGM
        df_WISDM = smart_read("datasets for T1D/WISDM/Data Tables/DeviceCGM.txt") 

        # reasmple to 5 minutes
        df_WISDM["ts"] = pd.to_datetime(df_WISDM["DeviceDtTm"])
        df_WISDM = df_WISDM.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "Value"))

        # read sex
        df_WISDM_screen = smart_read("datasets for T1D/WISDM/Data Tables/DiabScreening.txt")
        df_WISDM_screen = df_WISDM_screen[["PtID", "Gender"]]

        # read age
        df_WISDM_age = smart_read("datasets for T1D/WISDM/Data Tables/PtRoster.txt")
        df_WISDM_age = df_WISDM_age[["PtID", "AgeAsOfEnrollDt", "EnrollDt"]]

        # merge sex and age
        df_WISDM_screen = pd.merge(df_WISDM_screen, df_WISDM_age, on="PtID", how="inner")

        # merge demographics with CGM
        df_WISDM = pd.merge(df_WISDM, df_WISDM_screen, on=["PtID"], how="left")

        # rename columns for semantic equality
        df_WISDM = df_WISDM.rename(columns={"Value": "GlucoseCGM", "AgeAsOfEnrollDt" : "Age", "Gender": "Sex", "HbA1cTestRes": "Hba1c"})
        df_WISDM["PtID"] = df_WISDM["PtID"].astype(str) + "_WISDM"
        df_WISDM["Database"] = "WISDM"

        return df_WISDM

    def df_shanghai():
        # Path to your Excel files (adjust the path as needed)
        file_paths_shang = glob.glob("datasets for T1D/shanghai/Shanghai_T1DM/*.xlsx")  # Change path accordingly

        # Initialize an empty list to store DataFrames
        df_list_shang = []

        # Loop through each file, reading it into a DataFrame and assigning a PtID
        for idx, file in enumerate(file_paths_shang, start=1):
            df = smart_read(file)  # Read the Excel file
            name = os.path.splitext(os.path.basename(file))[0] # extract Subject ID from filename
            df["PtID"] = str(name)  # Assign a unique PtID
            df_list_shang.append(df)
            
        # Concatenate all DataFrames into one
        df_shang = pd.concat(df_list_shang, ignore_index=True)

        # Path to your Excel files (adjust the path as needed)
        file_paths_shang_2 = glob.glob("datasets for T1D/shanghai/Shanghai_T1DM/*.xls")  # Change path accordingly

        # Initialize an empty list to store DataFrames
        df_list_shang_2  = []
        idxx = 5
        # Loop through each file, reading it into a DataFrame and assigning a PtID
        for idx, file in enumerate(file_paths_shang_2 , start=1):
            df = smart_read(file)  # Read the Excel file
            name = os.path.splitext(os.path.basename(file))[0] # extract Subject ID from filename
            df["PtID"] = str(name) 
            df_list_shang_2 .append(df)
            idxx = idxx + 1
            
        # Concatenate all DataFrames into one
        df_shang_2  = pd.concat(df_list_shang_2, ignore_index=True)
        df_shang  = pd.concat([df_shang, df_shang_2], ignore_index=True)

        # reasmple to 5 minutes
        df_shang["ts"] = pd.to_datetime(df_shang["Date"])
        df_shang = df_shang.groupby("PtID", group_keys=False).apply(lambda x: fill_gaps_sampling(x, "ts", "PtID", "CGM (mg / dl)"))
      
        df_shang_info = smart_read("datasets for T1D/shanghai/Shanghai_T1DM_Summary.xlsx")
        df_shang_info["Sex"] = df_shang_info["Gender (Female=1, Male=2)"].replace({2: "M", 1: "F"})
        df_shang_info = df_shang_info.rename(columns={"Patient Number": "PtID"})
        df_shang_info = df_shang_info[["PtID", "Sex", "Age (years)"]]


        df_shang = pd.merge(df_shang, df_shang_info, on=["PtID"], how="left")
        # rename columns for semantic equality
        df_shang = df_shang.rename(columns={"CGM (mg / dl)": "GlucoseCGM", "Patient Number": "PtID", "Age (years)": "Age"})
        df_shang["PtID"] = df_shang["PtID"] + "_ShanghaiT1D" 
        df_shang["Database"] = "ShanghaiT1D"

        return df_shang

    def df_d1namo():
        # Define base path
        base_path_D1namo_ECG = "datasets for T1D/D1NAMO/diabetes_subset"

        # List to store all dataframes
        all_data_HR = []
        all_data_GLC = []

        # Loop through each subject directory 
        for subject_id in os.listdir(base_path_D1namo_ECG):

            # read all files including CGM readings
            person_path_glc = os.path.join(base_path_D1namo_ECG, subject_id)

            if not os.path.isdir(person_path_glc):
                continue

            # Loop through each file of the subject
            for file in os.listdir(person_path_glc):
                
                # if file is a csv and ends with glucose, read it 
                if file.endswith("glucose.csv"): 
                    file_path_glc = os.path.join(person_path_glc, file)

                    try:
                        D1NAMO_glc = smart_read(file_path_glc)
                        D1NAMO_glc["PtID"] = subject_id
                        # rename for semantic equality
                        D1NAMO_glc = D1NAMO_glc.rename(columns={"glucose": "GlucoseCGM"})
                        # convert mmol/L to mg/dL
                        D1NAMO_glc["GlucoseCGM"] = D1NAMO_glc["GlucoseCGM"] * 18.02
                        # convert timestamp to datetime
                        D1NAMO_glc["ts"] = pd.to_datetime(D1NAMO_glc["date"] + " " + D1NAMO_glc["time"])

                        # split manual and contiunous glucose data into seperate columns
                        D1NAMO_glc["mGLC"] = D1NAMO_glc["GlucoseCGM"].where(D1NAMO_glc["type"] == "manual")
                        D1NAMO_glc["GlucoseCGM"] = D1NAMO_glc["GlucoseCGM"].where(D1NAMO_glc["type"] == "cgm")


                        all_data_GLC.append(D1NAMO_glc)

                    except Exception as e:
                        print(f"Failed to read {file_path_glc}: {e}")

            # read HR 
            person_path = os.path.join(base_path_D1namo_ECG, subject_id, "sensor_data")

            if os.path.isdir(person_path):

                for session_folder in os.listdir(person_path):
                    session_path = os.path.join(person_path, session_folder)

                    # Skip if not a directory
                    if not os.path.isdir(session_path):
                        continue
                    
                    for file in os.listdir(session_path):
                        if file.endswith("_Summary.csv"): 
                            file_path = os.path.join(session_path, file)
                            
                            try:
                                D1NAMO_hr = smart_read(file_path)
                                D1NAMO_hr["PtID"] = subject_id
                                all_data_HR.append(D1NAMO_hr)
                            except Exception as e:
                                print(f"Failed to read {file_path}: {e}")

        # Concatenate all HR DataFrames into one
        df_D1NAMO_HR = pd.concat(all_data_HR, ignore_index=True)
        # Concatenate all glucose DataFrames into one
        df_D1NAMO_GLC = pd.concat(all_data_GLC, ignore_index=True)

        # take only a subset since we do not need the other values
        df_D1NAMO_HR = df_D1NAMO_HR[["Time", "PtID", "HR"]]
        # the datetime needs to be converted to the same format 
        df_D1NAMO_HR["ts"] = pd.to_datetime(df_D1NAMO_HR["Time"], format="%d/%m/%Y %H:%M:%S.%f")
        df_D1NAMO_HR["ts"] = pd.to_datetime(df_D1NAMO_HR["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
        # resample Hr and glucose values individually for each subject to 5 minutes
        df_D1NAMO_HR = df_D1NAMO_HR.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "PtID", value = "HR"))
        df_D1NAMO_GLC = df_D1NAMO_GLC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # merge HR and GLC
        df_D1NAMO = pd.merge(df_D1NAMO_GLC, df_D1NAMO_HR, on=["PtID", "ts"], how="left")
        # include d1namo so we can identify the dataframe
        df_D1NAMO["PtID"] = df_D1NAMO["PtID"] + "_D1NAMO" 
        df_D1NAMO["Database"] = "D1NAMO"

        return df_D1NAMO

    def df_DDATSHR():
        # subjects wear the metronic or abbott CGM device, so bith files are read         
        df_DDATSHR_ins = smart_read("datasets for T1D/DDATSHR/data-csv/Medtronic.csv")

        df_DDATSHR_glc = smart_read("datasets for T1D/DDATSHR/data-csv/Abbott.csv")

        # abbott estiamtes glucose evry 15 minutes
        df_DDATSHR_glc["ts"] = pd.to_datetime(df_DDATSHR_glc["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_glc["Local time [hh:mm]"])
        # for each subject replace the scan values with the historic glucose values
        df_DDATSHR_glc["Historic Glucose [mmol/l]"] = df_DDATSHR_glc.groupby("Subject code number").apply(
            lambda group: group["Scan Glucose [mmol/l]"].where(
                group["Scan Glucose [mmol/l]"].notna(), group["Historic Glucose [mmol/l]"]
            )
        ).reset_index(drop=True)

        df_DDATSHR_glc = df_DDATSHR_glc.groupby("Subject code number", group_keys=False).apply(lambda x: fill_gaps_sampling(x, "ts", "Subject code number", "Historic Glucose [mmol/l]"))

        # first remove all other timestamps which were used for ins but have no glucose entry
        df_DDATSHR_ins = df_DDATSHR_ins.dropna(subset=["Sensor Glucose [mmol/l]"])
        # medtronic estimates glucose every 5 minutes
        df_DDATSHR_ins["ts"] = pd.to_datetime(df_DDATSHR_ins["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_ins["Local time [hh:mm:ss]"])
        # rename so that both columns match of both dataframes
        df_DDATSHR_ins = df_DDATSHR_ins.rename(columns={"Sensor Glucose [mmol/l]": "Historic Glucose [mmol/l]"})
        df_DDATSHR_ins = df_DDATSHR_ins[["ts", "Subject code number", "Historic Glucose [mmol/l]"]]

        df_DDATSHR_glc_ins = pd.concat([df_DDATSHR_glc, df_DDATSHR_ins])
        # reasmple to 5 minutes 
        df_DDATSHR_glc_ins = df_DDATSHR_glc_ins.groupby("Subject code number", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "Subject code number", value = "Historic Glucose [mmol/l]"))
        df_DDATSHR_glc_ins["Historic Glucose [mmol/l]"] = df_DDATSHR_glc_ins["Historic Glucose [mmol/l]"] * 18.02

        # read age and gender and merge with glucose
        df_DDATSHR_info = smart_read("datasets for T1D/DDATSHR/data-csv/population.csv")
        df_DDATSHR_info = df_DDATSHR_info[["Subject code number", "Gender [M=male F=female]", "Age [yr]"]]
        df_DDATSHR = pd.merge(df_DDATSHR_glc_ins, df_DDATSHR_info, on=["Subject code number"], how="left")

        #read HR 
        df_DDATSHR_hr = smart_read("datasets for T1D/DDATSHR/data-csv/Fitbit/Fitbit-heart-rate.csv")
        
        # convert date and time to datetime and resample to 1 minute intervals
        df_DDATSHR_hr["ts"] = pd.to_datetime(df_DDATSHR_hr["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_hr["Local time [hh:mm]"])
        # resample to 5 minutes
        df_DDATSHR_hr = df_DDATSHR_hr.groupby("Subject code number", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "Subject code number", value = "heart rate [#/min]"))


        # take only a subset since we do not need the other values
        df_DDATSHR_hr = df_DDATSHR_hr[["ts", "Subject code number", "heart rate [#/min]"]]

        # merge HR, and GLC
        df_DDATSHR = pd.merge(df_DDATSHR, df_DDATSHR_hr, on=["Subject code number", "ts"], how="left")

        # rename columns for semantic equality
        df_DDATSHR = df_DDATSHR.rename(columns={"Subject code number":"PtID", "Historic Glucose [mmol/l]": "GlucoseCGM", "Gender [M=male F=female]" : "Sex", "Age [yr]": "Age", "heart rate [#/min]": "HR", "steps [#]": "Steps" })
        df_DDATSHR["PtID"] = df_DDATSHR["PtID"].astype(str) + "_DDATSHR"
        df_DDATSHR["Database"] = "DDATSHR"


        return df_DDATSHR

    def df_rtc():
        # Path to your Excel files (adjust the path as needed)
        file_paths_rtc = glob.glob("datasets for T1D/RT_CGM/DataTables/tblADataRTCGM*.csv") 

        # Initialize an empty list to store DataFrames
        df_list_rtc = []

        # Loop through each file, reading it into a DataFrame and assigning a PtID
        for idx, file in enumerate(file_paths_rtc, start=1):
            df = smart_read(file)  # Read the Excel file
            # Assign a unique PtID
            df_list_rtc.append(df)

        # Concatenate all DataFrames into one
        df_rtc  = pd.concat(df_list_rtc , ignore_index=True)

        # since timestamps have mixed formats
        try: 
            df_rtc["ts"] = pd.to_datetime(df_rtc["DeviceDtTm"].str.split(".").str[0], format="%Y-%m-%d %H:%M:%S")
        except:
            df_rtc["ts"] = pd.to_datetime(df_rtc["DeviceDtTm"], format="%Y-%m-%d %H:%M:%S")

        # detect the samle rate since some subjects have glucose collected in 10 minute intervals
        fre_rtc = df_rtc.groupby("PtID", group_keys=False).apply(lambda x: detect_sample_rate(x, time_col = "ts")).reset_index(name="Frequency")
        df_rtc = df_rtc.merge(fre_rtc, on="PtID", how="left")

        # since not all have the frequency of 5 minutes we have to ffill the values 
        df_rtc["Frequency"] = df_rtc["Frequency"].fillna(method="ffill")

        # assign condition of 10 minutes to resample those seperately
        condition = df_rtc["Frequency"] == "10 min"
        # split the dataframe based on the condition
        df_10min = df_rtc[condition]   
        df_5min = df_rtc[~condition] 

        # undersample dataframe with 10 minute intervals to 5 minutes
        df_10min = df_10min.groupby("PtID", group_keys=False).apply(lambda x: fill_gaps_sampling(x, "ts", "PtID", "Glucose", 10))
        # concatenate both again into one database
        df_RTC = pd.concat([df_10min, df_5min], ignore_index=True)
        # resample dataframe to 5 minutes
        df_RTC = df_RTC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "Glucose"))

        # read demographics
        df_rtc_info = smart_read("datasets for T1D/RT_CGM/DataTables/tblAPtSummary.csv")
        df_rtc_info = df_rtc_info[["PtID", "Gender", "AgeAsOfRandDt"]]

        # merge demographics with cgm data
        df_RTC = pd.merge(df_RTC, df_rtc_info, on=["PtID"], how="left")

        # rename for semantic equality
        df_RTC = df_RTC.rename(columns={"Glucose": "GlucoseCGM", "AgeAsOfRandDt" : "Age", "Gender": "Sex"})
        df_RTC["PtID"] = df_RTC["PtID"].astype(str) + "_RT-CGM"
        df_RTC["Database"] = "RT-CGM"

        return df_RTC
    
    def df_T1GDUJA():

        # read T1GDUJA
        df_T1G = smart_read("datasets for T1D/T1GDUJA/glucose_data.csv")

        # since timestamps are stored in mixed formats 
        try: 
            df_T1G["ts"] = pd.to_datetime(df_T1G["date"].str.split(".").str[0], format="%Y-%m-%d %H:%M:%S")
        except:
            df_T1G["ts"] = pd.to_datetime(df_T1G["date"], format="%Y-%m-%d %H:%M:%S")
        # rename column for semantic equality
        df_T1G = df_T1G.rename(columns={"sgv": "GlucoseCGM"})
        df_T1G["PtID"] = "T1GDUJA"
        # resample to 5 minutes
        df_T1G = df_resample(df_T1G, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM")
        
        df_T1G["Database"] = "T1GDUJA"
        return df_T1G

    # this function calls the desired functions reading the datasets which are given as a list and returns them as a list of dataframes 
    def try_call_functions(functions):
        combined_df_dict = {}
        # each function is called seperately
        for func in functions:
            try:
                dataframe = func()
                combined_df_dict[func.__name__] = dataframe
            except Exception as e:
                print(f"Error in {func.__name__}(): {e}")
        # dataframes are combined and stored as a list 
        combined_df_list = list(combined_df_dict.values())
        return combined_df_list
    
    # if the integrated datasets of publicly available databases are found, only read restricted datasets
    if (read_all == False):
        datasets = [df_granada, df_diatrend]
    elif(read_all == True):
        datasets = [df_granada, df_diatrend, df_city, df_dclp, df_pedap, df_replace, df_sence, df_shd, df_wisdm, df_shanghai, df_hupa,df_d1namo, df_DDATSHR, df_rtc, df_T1GDUJA]
    
    # calls all functions to return a list of all datasets 
    combined_df_list = try_call_functions(datasets)
    return combined_df_list



def combine_data(modus, restricted_list, columns_to_check = ["Age", "Sex"]):

    """
    This function integrates all dataframes into one database depending on the modus 
    Parameters:
    - modus: can be either 1, 2, or 3. 
        - 1 is for the main database and integrates all dataframes which include CGM values
        - 2 is for subdatabase I and integrates all dataframes which include CGM values and demographics of age and sex
        - 3 is for subdatabase II and integrates all dataframes which include CGM values and HR 
    - retricted_list: list of datasets which were not shared due to licensing restrictions. These need to be loaded and integrated seperately
    - columns_to_check are default values to set age groups if modus two is applied
    Output: returns the integrated subset of restricted data 
    """

    # this function takes the original database as input and converts the value of the Age column to integers
    def set_ages(df, column = "Age"):
        
        # convert each string to a numeric value
        def to_numeric(val):
            if isinstance(val, str) and "-" in val:
                val = val.replace("yrs", "")  
                val = val.strip()  # clean up any extra spaces
                start, end = map(int, val.replace(" ", "").split("-"))
                return (start + end) / 2  # or start, or end
            return int(val)

        all_ages = set(df[column])

        # Initialize one empty lists for the ages reported as strings
        Age_str = []

        # Iterate over the values and extract ages reported in strings
        for value in all_ages:
            if isinstance(value, str) or isinstance(value, object):
                Age_str.append(str(value)) 

        df = df.copy()
        df[column] = df[column].apply(to_numeric)

        # create age groups 
        bins = [0, 2,6, 10, 13, 17,25, 35, 55, 100]
        labels = ["0-2", "3-6", "7-10", "11-13", "14-17", "18-25", "26-35", "36-55", "56+"]
        
        # categorize the ages of the Age colum into the defined age groups and assign them to new column AgeGroup
        df["AgeGroup"] = pd.cut(df[column], bins=bins, labels=labels, right=True)
        
        # return the dataframe with the new colum 
        return df

    # this function concatenate the dataframes 
    def concat_rows_on_columns(dfs, columns):
    
        # Select only the specified columns from each dataframe
        dfs = [df[columns] for df in dfs]
        # Concatenate all dataframes vertically (adding rows)
        result = pd.concat(dfs, ignore_index=True)
        # remove subjects only including nan values in the GlucoseCGM column
        subjects_to_keep = result.groupby("PtID")["GlucoseCGM"].transform(lambda x: not x.isna().all())
        # only include subjects with columns
        result = result[subjects_to_keep]

        # if columns to check are all within the dataframe columns 
        if all(col in result.columns for col in columns_to_check):
            #  remove nan values since the final datasets should have all columns included for each subject
            df_cleaned = result.dropna(subset=columns_to_check)
            # set the ages 
            df_cleaned = set_ages(df_cleaned)
            return df_cleaned
        else:
            return result

    # either 1: all CGM values, 2: CGM and demographics, or 3: CGM and HR
    allowed_values = [1, 2, 3] 
    if modus not in allowed_values:
        print("Invalid input. Please enter: 1, 2, 3")
        return
    
    if modus == 1: 
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "Database"]
    elif modus == 2:
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "Age", "Sex", "Database"] 
    else:
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "HR", "Database"]
    
    # take only the subset of columns of interest
    filtered_dfs = [df for df in restricted_list if all(col in df.columns for col in columns_to_keep)]
    # call the concat_rows_on_columns function
    combined_df = concat_rows_on_columns(filtered_dfs, columns=columns_to_keep)

    # return combined_df
    return combined_df