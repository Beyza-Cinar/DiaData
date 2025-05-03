"""
This file provides functions to read the seperate datasets, store them in a list and integrate them into three separate datasets.
"""


# necessary imports
import pandas as pd
import glob
import os
import csv
import re




def df_resample(df_org, timestamp, frequency, mode, fillid, value):
    """
    This function is used to resample the dataset to the same frequency.
    Parameters:
   - df_org is the original dataset, 
   - timestamp is the name of the column with the timestamps
   - frequency is the target frequency
   - the mode—either "glucose" or "vitals"—specifies whether PtID fillforward is necessary; heart rate data does not require this
   - fillid is the name of the column which needs fillforward to impute produced missing values
   - value-either "glucose" or "heartrate"-is the column name which should be converted into integer values 
    Output: It returns the resampled dataframe with no missing values in the PtID column and glucose/heartrate values as float values
   """ 
    df = df_org.copy()
    # if the mode is set to "glucose", the timestamps are first sorted, then rounded, and finally resampled to match the target frequency
    if mode == "glucose":
        df = df.sort_values(by=timestamp)
        df[timestamp] = df[timestamp].dt.round(frequency)
        # rounding can induce duplicates which need to be removed to enable resampling
        df = df.drop_duplicates(subset=[timestamp])
        # the timestamps column is set to be the index 
        df = df.set_index(timestamp)
        # based on the index, the dataframe is resampled to the target frequency
        df = df.resample(frequency).asfreq()
        # the index is resetted 
        df = df.reset_index()
        # missing values which occured due to resampling but are essential are feedforward filled 
        df[fillid] = df[fillid].fillna(method="ffill")
    # if the mode is set to "vitals", the timestamps are only rounded to the target frequency
    elif mode == "vitals":
        df[timestamp] = df[timestamp].dt.round(frequency)
        df = df.drop_duplicates(subset=[timestamp])
    # if a wrong input is given, a warning is outputted
    else:
        print("Mode can be either glucose or vitals")
    # finally, the column with the specified values is converted to numericals (floats)
    df[value] = pd.to_numeric(df[value], errors="coerce")
    # the reasmpled dataframe is returned
    return df


def fill_gaps_sampling(df, timestamp, subject_id, glucose, fillmin=15):
    """
    This function applies feedforward filling to glucose values missing as a result of undersampling.
    Parameters:
    - df is the original dataframe
    - timestamp is the name of the column with the timestamps
    - subjects id is the name of the column with the subject ids
    - glucose is the name of the column with the glucose measurements
    - fillmin is the original frequency of the dataset; by default it is set to 15 minutes
    """ 
    # first, the timestamp column of the original dataframe is sorted and then copied
    sorted_df = df.sort_values(timestamp).copy()
    # continuous true glucose measurements are identified having a time difference of the dataset's original frequency
    sorted_df["time_diff"] = sorted_df[timestamp].diff()
    sorted_df["gap"] = sorted_df["time_diff"] > pd.Timedelta(minutes=fillmin)
    # consecutive glucose measurements are grouped and assigned a group id 
    sorted_df["group_id"] = sorted_df["gap"].cumsum()

    # an empty list is intialized which will store individually resampled groups 
    resampled_groups = []

    # for each group of consecutive measurements, the timestamp is undersampled to 5 minutes and occuring gaps are filled with the fillforward method
    for _, group in sorted_df.groupby("group_id"):
        # to resample the group, the function "df_resample()" is called
        group_resampled = df_resample(group, timestamp = timestamp, frequency= "5min", mode="glucose", fillid = subject_id, value = glucose)
        group_resampled[glucose] = group_resampled[glucose].ffill()
        # the resampled group without gaps is appended to the initialized list
        resampled_groups.append(group_resampled)

    # all groups are concatenated to one dataframe
    resampled_df = pd.concat(resampled_groups)
    # finally, the whole dataset is resampled to 5 minute intervals
    resampled_df = df_resample(resampled_df, timestamp = timestamp, frequency= "5min", mode="glucose", fillid = subject_id, value = glucose)
    # the resamoked dataset is returned
    return resampled_df.reset_index()



def detect_best_separator(file_path, sample_size=10000):
    """ 
    This function finds the best seperator enable the automatic read of the dataset.
    Parameter: 
    - file_path is the path of the file 
    - sample size is the amount of characters which should be read to find the sepatator
    Output: best found seperator is outputted
    """
    # possible encodings are defined
    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin1"]
    # possible delimeters are defined
    delimiters = [",", ";", "\t", "|"]

    # the file is opened with the correct encoding by trying each encoding
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                # the file is opened to read
                sample = f.read(sample_size)
                # if UTF-16, the delimeters are count to find the best option instead of using the Sniffer function
                if enc == "utf-16":
                    counts = {d: sample.count(d) for d in delimiters}
                    # the delimeter with the majority of counts is chosen 
                    best_guess = max(counts, key=counts.get)
                    # if the majority is larger than 0 it is returned as the best guess
                    if counts[best_guess] > 0:
                        return best_guess
                    continue
                # else, try the Sniffer function which automatically returns the delimeter
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
                    return dialect.delimiter
                # if an error occurs, fallback: use count-based detection
                except csv.Error:
                    counts = {d: sample.count(d) for d in delimiters}
                    best_guess = max(counts, key=counts.get)
                    if counts[best_guess] > 0:
                        return best_guess
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not detect delimiter or unsupported encoding.")



def smart_read(file_path, skip = 0):
    """
    This function automatically reads a file into a pandas DataFrame based on its extension.
    It automatically detects the delimiter for .csv and .txt files from function "detect_best_separator()".
    Parameter: 
    - file_path is the path of the file
    - skip is the number of rows which should be skipped 
    Output: a pandas dataframe is outputted
    """

    # extracts the extension of the file 
    ext = os.path.splitext(file_path)[1].lower()

    # checks if the file is an excel file
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    # checks if the file is a csv or txt file
    elif ext in [".csv", ".txt"]:
        # based on the returned best separator, the file is read
        sep = detect_best_separator(file_path)
        # try with normal encoding
        try:
            df = pd.read_csv(file_path, sep=sep, engine="python", on_bad_lines="skip", skiprows=skip)
        # if an error occurs, try with the utf-16 encoding 
        except:
            df = pd.read_csv(file_path, sep=sep, engine="python", on_bad_lines="skip", encoding="utf-16", skiprows=skip)
    # if still an error occurs, output warning 
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    # returns the pandas dataframe
    return df



def detect_sample_rate(df, time_col=None, expected_rates=(5, 10, 15)):
    """
    This functions detects the sample rate of the glucose measurements.
    A frequency of 5, 10, and 15 minutes are allowed which is typical for CGM devices.
    Parameter: df is the original dataframe 
    Output: most common frequency is outputted
    """
    # extracts time column or index
    times = pd.to_datetime(df[time_col]) if time_col else pd.to_datetime(df.index)
    times = times.sort_values()

    # computes time differences in minutes
    deltas = times.diff().dropna().dt.total_seconds() / 60  # in minutes

    # rounds to nearest whole minute
    deltas_rounded = deltas.round().astype(int)

    # finds the most common interval
    most_common = deltas_rounded.value_counts().idxmax()

    # matches to known expected rates
    if most_common in expected_rates:
        # prints the most common frequency in monutes 
        return f"{most_common} min"
    else:
        return "Unknown"
    


def read_data(read_all = True):
    """
    This function reads the datasets individually and returns them as a list of dataframes.
    Parameter: read_all can be True or False. If true, all datasets are read and returned. If false, only the set of restricted datasets are read.
    Output: returns a list of pandas dataframes
    """
    def df_granada():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_granada = smart_read("datasets for T1D/granada/T1DiabetesGranada/glucose_measurements.csv")
        # column names are renamed for semantic equality
        df_granada = df_granada.rename(columns={"Measurement": "GlucoseCGM", "Patient_ID": "PtID"})

        # all datasets should keep the same format of date and time -> combine both and convert to datetime
        df_granada["ts"] = pd.to_datetime(df_granada["Measurement_date"] + " " + df_granada["Measurement_time"])
        # undersamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_granada = df_granada.groupby("PtID", group_keys=False).apply(lambda x: fill_gaps_sampling(x, "ts", "PtID", "GlucoseCGM"))

        # reads the dataframe with demographics
        df_granada_info = smart_read("datasets for T1D/granada/T1DiabetesGranada/Patient_info.csv")
        # converts timstamps to datetime
        df_granada_info["Initial_measurement_date_date"] = pd.to_datetime(df_granada_info["Initial_measurement_date"])
        # reduces the columns to only important columns
        df_granada_info = df_granada_info[["Sex", "Birth_year", "Patient_ID"]]
        # column names are renamed for semantic equality
        df_granada_info = df_granada_info.rename(columns={"Patient_ID": "PtID"})
        # merges both dataframes
        df_granada = pd.merge(df_granada, df_granada_info, on="PtID", how="inner")
        # computes the age based on the birth year and the datetime of measurement
        df_granada["Age"] = df_granada["ts"].dt.year - df_granada["Birth_year"]
        # removes the "Birth_year" column
        df_granada = df_granada.drop(["Birth_year"], axis=1)

        # adds the database name to the patient ID to enable reidentification 
        df_granada["PtID"] = df_granada["PtID"].astype(str) + "_T1DGranada"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_granada["Database"] = "T1DGranada"
        return df_granada


    def df_diatrend():
        # path to the single CGM datasets of each subject
        file_paths_diatrend = glob.glob("datasets for T1D/Diatrend/DiaTrendAll/*.xlsx") 

        # initializes an empty list to store the dataframes
        df_list_diatrend = []

        # loops through each file
        for idx, file in enumerate(file_paths_diatrend, start=1):
            # reads the file with the "smart_read()" function
            df = smart_read(file)  
            # extracts the PtID from the filename
            name = os.path.splitext(os.path.basename(file))[0] 
            # adds the extracted ID to the "PtID" column
            df["PtID"] = str(name)
            # adds the dataframe to the list
            df_list_diatrend.append(df)

        # concatenates all dataframes into one dataframe
        df_diatrend = pd.concat(df_list_diatrend, ignore_index=True)

        # column names are renamed for semantic equality
        df_diatrend = df_diatrend.rename(columns={"mg/dl": "GlucoseCGM"})
        # converts timstamps to datetime
        df_diatrend["ts"] = pd.to_datetime(df_diatrend["date"], format="%Y-%m-%d %H:%M:%S")
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_diatrend = df_diatrend.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # reads dataframe including demographics
        df_diatrend_info = smart_read("datasets for T1D/Diatrend/SubjectDemographics_3-15-23.xlsx") 
        # column names are renamed for semantic equality
        df_diatrend_info["Sex"] = df_diatrend_info["Gender"].replace({"Male": "M", "Female": "F"})
        # adds a "PtID" column storing the SubjectIDs
        df_diatrend_info["PtID"] = "Subject" + df_diatrend_info["Subject"].astype(str)
        # reduces the columns to only important columns
        df_diatrend_info = df_diatrend_info[["Sex", "Age","PtID"]]
        # merges both dataframes
        df_diatrend = pd.merge(df_diatrend, df_diatrend_info, on="PtID", how="inner")
        # adds the database name to the patient ID to enable reidentification 
        df_diatrend["PtID"] = df_diatrend["PtID"].astype(str) + "_DiaTrend"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_diatrend["Database"] = "DiaTrend"
        return df_diatrend

    def df_city():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_city = smart_read("datasets for T1D/CITYPublicDataset/Data Tables/DeviceCGM.txt")
        # column names are renamed for semantic equality
        df_city = df_city.rename(columns={"Value": "GlucoseCGM"})

        # differentiates between CGM glucose and finger prick glucose
        df_city["mGLC"] = df_city["GlucoseCGM"].where(df_city["RecordType"] == "Calibration")
        df_city["GlucoseCGM"] = df_city["GlucoseCGM"].where(df_city["RecordType"] == "CGM")

        # converts timstamps to datetime
        df_city["ts"] = pd.to_datetime(df_city["DeviceDtTm"])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_city = df_city.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # reads dataframe including sex data
        df_city_screen = smart_read("datasets for T1D/CITYPublicDataset/Data Tables/DiabScreening.txt")
        # reduces the columns to only important columns
        df_city_screen = df_city_screen[["PtID", "Sex"]]

        # reads dataframe including age data
        df_city_age = smart_read("datasets for T1D/CITYPublicDataset/Data Tables/PtRoster.txt")
        # reduces the columns to only important columns
        df_city_age = df_city_age[["PtID", "AgeAsOfEnrollDt"]]

        # merges dataframes of sex and age 
        df_city_info = pd.merge(df_city_screen, df_city_age, on=["PtID"], how="inner")

        # merges dataframes of demorgaphics and CGM data
        df_city = pd.merge(df_city, df_city_info, on=["PtID"], how="left")
        # column names are renamed for semantic equality
        df_city = df_city.rename(columns={"AgeAtEnrollment" : "Age"})
        # adds the database name to the patient ID to enable reidentification 
        df_city["PtID"] = df_city["PtID"].astype(str) + "_CITY"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_city["Database"] = "CITY"
        return df_city

    def df_dclp():
        # this dataset has two different CGM records which are read seperately with the "smart_read()" function
        df_dclp = smart_read("datasets for T1D/DCLP3/Data Files/DexcomClarityCGM_a.txt")
        # reduces the columns to only important columns
        df_dclp = df_dclp[["PtID", "RecID", "DataDtTm", "CGM", "DataDtTm_adj"]]

        df_dclp_other = smart_read("datasets for T1D/DCLP3/Data Files/OtherCGM_a.txt")
        # reduces the columns to only important columns
        df_dclp_other = df_dclp_other[["PtID", "RecID", "DataDtTm", "CGM", "DataDtTm_adjusted"]]

        # column names are renamed for semantic equality
        df_dclp_other.rename(columns= {"DataDtTm_adjusted": "DataDtTm_adj"})
        # merges both dataframes
        df_DCLP = pd.concat([df_dclp, df_dclp_other])

        # converts timstamps to datetime
        df_DCLP["ts"] = pd.to_datetime(df_DCLP["DataDtTm"])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_DCLP = df_DCLP.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "CGM"))

        # reads dataframes of demographics 
        df_dclp_screen = smart_read("datasets for T1D/DCLP3/Data Files/DiabScreening_a.txt")
        df_dclp_screen = df_dclp_screen[["PtID", "AgeAtEnrollment",	"Gender"]]
        # merges both dataframes
        df_DCLP = pd.merge(df_DCLP, df_dclp_screen, on=["PtID"], how="left")

        # column names are renamed for semantic equality
        df_DCLP = df_DCLP.rename(columns={"CGM": "GlucoseCGM", "Gender" : "Sex", "AgeAtEnrollment" : "Age", "HbA1cTestRes" : "Hba1c"})
        # adds the database name to the patient ID to enable reidentification 
        df_DCLP["PtID"] = df_DCLP["PtID"].astype(str) + "_DLCP3"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_DCLP["Database"] = "DLCP3"
        return df_DCLP
    
    def df_hupa():
        # path to the base directory
        base_path_hupa = "datasets for T1D/HUPA-UCM/Raw_Data"

        # initilaizes lists to store all dataframes of heartrate and glucose
        all_data_HR_h = []
        all_data_GLC_h = []


        # loops through each subject directory 
        for subject_id_h in os.listdir(base_path_hupa):

            # reads all files including CGM readings
            person_path_h = os.path.join(base_path_hupa, subject_id_h)
            # skips if not a directory
            if not os.path.isdir(person_path_h):
                continue
            # combines the path name of the person and the folder
            for folder_h in os.listdir(person_path_h):
                folder_path_h = os.path.join(person_path_h, folder_h)

                # skips if not a directory
                if not os.path.isdir(folder_path_h):
                    continue
                # loops through each file 
                for file in os.listdir(folder_path_h):
                    
                    # skips if not a directory
                    if not os.path.isdir(folder_path_h):
                        continue
                    
                    # if file contains heart and is a csv file, the file is read
                    if file.endswith(".csv") and "heart" in file:
                        # combines the paths 
                        file_path_hr_h = os.path.join(folder_path_h, file)

                        try:
                            # reads the dataframe with the heartrate measurements with the "smart_read()" function 
                            hupa_hr = smart_read(file_path_hr_h)
                            # adds a PtID
                            hupa_hr["PtID"] = subject_id_h
                            # extracts the date from the filename
                            match = re.search(r"([\d]{4}-[\d]{2}-[\d]{2})", file_path_hr_h)
                            if match:
                                date_str = match.group(1)
                                # adds the date 
                                hupa_hr["date"] = date_str
                                # converts the time to a string
                                hupa_hr["Time"] = hupa_hr["Time"].astype(str)
                                # combines the date with the time and converts to datetime
                                hupa_hr["ts"] = pd.to_datetime(hupa_hr["date"] + " " + hupa_hr["Time"])
                                # column names are renamed for semantic equality
                                hupa_hr = hupa_hr.rename(columns={"Heart Rate" : "HR"})
                                # reduces the columns to only important columns
                                hupa_hr = hupa_hr[["ts", "PtID", "HR"]]
                            else:
                                print("No date found in the file path.")
                            # adds all files containing HR to the same list
                            all_data_HR_h.append(hupa_hr)

                        except Exception as e:
                            print(f"Failed to read {file_path_hr_h}: {e}")

                    # if file contains free style sensor and is a csv file, read the file; these files contain CGM measurements in 15 minute intervals
                    elif file.endswith(".csv") and "free_style_sensor" in file:
                        # joins the paths
                        file_path_glc_h = os.path.join(folder_path_h, file)
                        # subjects 25-28 have a different schema , thus these are loaded differently
                        if re.search(r"2[5-8]P", subject_id_h):
                            try:
                                # dataframes with the CGM measurements are read with the "smart_read()" function
                                hupa_glc = smart_read(file_path_glc_h, skip=2) 
                                # adds Subjects ID
                                hupa_glc["PtID"] = subject_id_h
                                # column names are renamed for semantic equality
                                hupa_glc = hupa_glc.rename(columns={"Sello de tiempo del dispositivo": "ts", "Historial de glucosa mg/dL" : "Historic Glucose", 
                                        "Escaneo de glucosa mg/dL": "Scan Glucose", "Tira reactiva para glucosa mg/dL": "MGlucose"})
                                # converts timstamps to datetime
                                hupa_glc["ts"] = pd.to_datetime(hupa_glc["ts"], format="mixed", dayfirst=True)
                                # historic glucose is replaced with aligning scan glucose
                                hupa_glc["GlucoseCGM"] = hupa_glc["Scan Glucose"].where(hupa_glc["Scan Glucose"].notna(), hupa_glc["Historic Glucose"])
                                # reduces the columns to only important columns
                                hupa_glc = hupa_glc[["ts", "PtID", "GlucoseCGM"]]
                                # resamples to 5 minute intervals to have unifrom sample rate
                                hupa_glc = fill_gaps_sampling(hupa_glc, "ts", "PtID", "GlucoseCGM")
                                
                                # adds all glucose data into one list 
                                all_data_GLC_h.append(hupa_glc)

                            except Exception as e:
                                print(f"Failed to read {file_path_glc_h}: {e}")
                        else: 
                            # reads remaining subjects
                            try: 
                                # dataframes with the CGM measurements are read with the "smart_read()" function
                                hupa_glc = smart_read(file_path_glc_h, skip=1) 
                                # adds Subject ID 
                                hupa_glc["PtID"] = subject_id_h
                                # column names are renamed for semantic equality
                                hupa_glc = hupa_glc.rename(columns={"Hora": "ts", "Histórico glucosa (mg/dL)" : "Historic Glucose", 
                                        "Glucosa leída (mg/dL)": "Scan Glucose", "Glucosa de la tira (mg/dL)": "MGlucose"})
                                # converts timestamp to datetime 
                                hupa_glc["ts"] = pd.to_datetime(hupa_glc["ts"], format="mixed", dayfirst=True)
                                # replaces historic glucose with scan glucose
                                hupa_glc["GlucoseCGM"] = hupa_glc["Scan Glucose"].where(hupa_glc["Scan Glucose"].notna(), hupa_glc["Historic Glucose"])
                                hupa_glc = hupa_glc[["ts", "PtID", "GlucoseCGM"]]
                                # resamples to 5 minute intervals to have unifrom sample rate
                                hupa_glc = fill_gaps_sampling(hupa_glc, "ts", "PtID", "GlucoseCGM")
                                
                                # adds the glucose file to a list of glucose files
                                all_data_GLC_h.append(hupa_glc)
                            except Exception as e:
                                print(f"Failed to read {file_path_glc_h}: {e}") 
                    # if file contains dexcom and is a csv file, the file is read; these contain CGM measurements every 5 minutes              
                    elif file.endswith(".csv") and "dexcom" in file:
                        # joins paths
                        file_path_glc_h = os.path.join(folder_path_h, file)
                        try:
                            # dataframes with the CGM measurements are read with the "smart_read()" function
                            hupa_glc_d = smart_read(file_path_glc_h)
                            # adds Subject ID
                            hupa_glc_d["PtID"] = subject_id_h
                            # column names are renamed for semantic equality
                            hupa_glc_d = hupa_glc_d .rename(columns={"Marca temporal (AAAA-MM-DDThh:mm:ss)" : "ts", "Tipo de evento": "Type", "Nivel de glucosa (mg/dl)": "GlucoseCGM"})
                            # keeps only eventtype = Niveles estimados de glucosa
                            hupa_glc_d = hupa_glc_d [hupa_glc_d ["Type"] == "Niveles estimados de glucosa"][["ts", "PtID", "GlucoseCGM"]]
                            # timestamp is converted to datetime
                            hupa_glc_d["ts"] = pd.to_datetime(hupa_glc_d["ts"]) 
                            # adds dataframe to list of glucose dataframes
                            all_data_GLC_h.append(hupa_glc_d)

                        except Exception as e:
                            print(f"Failed to read {file_path_glc_h}: {e}")


        # concatenates all HR dataframes into one dataframe
        df_hupa_HR = pd.concat(all_data_HR_h, ignore_index=True)
        # resamples to 1 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_hupa_HR = df_hupa_HR.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "PtID", value = "HR"))

        # concatenates all HR dataframes into one dataframe
        df_hupa_GLC = pd.concat(all_data_GLC_h, ignore_index=True)
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_hupa_GLC = df_hupa_GLC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency = "5min", mode = "glucose", fillid = "PtID", value = "GlucoseCGM"))

        # merges dataframes of HR and CGM data
        df_HUPA = pd.merge(df_hupa_GLC, df_hupa_HR, on=["PtID", "ts"], how="left")
        # adds the database name to the patient ID to enable reidentification 
        df_HUPA["PtID"] = df_HUPA["PtID"] + "_HUPA-UCM"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_HUPA["Database"] = "HUPA-UCM"
        return df_HUPA
    

    def df_pedap():
        # This database contains two different CGM dataframes for some subjects, both dataframe with the CGM measurements are read with the "smart_read()" function
        df_pedap= smart_read("datasets for T1D/PEDAP/Data Files/PEDAPDexcomClarityCGM.txt")
        # reduces the columns to only important columns
        df_pedap = df_pedap[["PtID", "RecID", "DeviceDtTm", "CGM"]]

        df_pedap_other = smart_read("datasets for T1D/PEDAP/Data Files/PEDAPOtherCGM.txt")
        # reduces the columns to only important columns
        df_pedap_other = df_pedap_other[["PtID", "RecID", "DeviceDtTm", "CGM"]]
        # concatenates both dataframes into one dataframes
        df_PEDAP = pd.concat([df_pedap, df_pedap_other])

        # converts timstamps to datetime
        df_PEDAP["ts"] = pd.to_datetime(df_PEDAP["DeviceDtTm"], format = "mixed")
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_PEDAP = df_PEDAP.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "CGM"))

        # reads dataframe of sex data
        df_PEDAP_screen = smart_read("datasets for T1D/PEDAP/Data Files/PEDAPDiabScreening.txt")
        # reduces the columns to only important columns
        df_PEDAP_screen = df_PEDAP_screen[["PtID", "Sex"]]

        # reads dataframe of age data
        df_PEDAP_age = smart_read("datasets for T1D/PEDAP/Data Files/PtRoster.txt")
        # reduces the columns to only important columns
        df_PEDAP_age = df_PEDAP_age[["PtID", "AgeAsofEnrollDt"]]

        # merges dataframes including age and sex
        df_PEDAP_screen = pd.merge(df_PEDAP_screen, df_PEDAP_age, on="PtID", how="inner")
        # merges dataframes of demographics and CGM data
        df_PEDAP = pd.merge(df_PEDAP, df_PEDAP_screen, on=["PtID"], how="left")

        # column names are renamed for semantic equality
        df_PEDAP = df_PEDAP.rename(columns={"CGM": "GlucoseCGM", "AgeAsofEnrollDt" : "Age"})
        # adds the database name to the patient ID to enable reidentification 
        df_PEDAP["PtID"] = df_PEDAP["PtID"].astype(str) + "_PEDAP"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_PEDAP["Database"] = "PEDAP"
        return df_PEDAP

    def df_replace():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_RBG = smart_read("datasets for T1D/ReplaceBG/Data Tables/HDeviceCGM.txt")

        # column names are renamed for semantic equality
        df_RBG = df_RBG.rename(columns={"GlucoseValue": "GlucoseCGM"})
        # splits CGM and finger prick glucose levels into separate columns
        df_RBG["mGLC"] = df_RBG["GlucoseCGM"].where(df_RBG["RecordType"] == "Calibration")
        df_RBG["GlucoseCGM"] = df_RBG["GlucoseCGM"].where(df_RBG["RecordType"] == "CGM")

        # initial date is set
        df_RBG["initdate"] = pd.to_datetime("2024-01-01")
        # time is added to the data 
        df_RBG["datetime"] = df_RBG["initdate"] + pd.to_timedelta(df_RBG["DeviceDtTmDaysFromEnroll"], unit="D")
        df_RBG["DeviceTm"] = df_RBG["DeviceTm"].astype(str)

        # converts date and time to datetime and combines them into one column
        df_RBG["ts"] = pd.to_datetime(df_RBG["datetime"].dt.strftime("%Y-%m-%d") + " " + df_RBG["DeviceTm"])

        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_RBG = df_RBG.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # reads dataframe with sex data
        df_RBG_screen = smart_read("datasets for T1D/REPLACEBG/Data Tables/HScreening.txt")
        # reduces the columns to only important columns
        df_RBG_screen = df_RBG_screen[["PtID", "Gender"]]

        # reads dataframe with age data
        df_RBG_age = smart_read("datasets for T1D/REPLACEBG/Data Tables/HPtRoster.txt")
        # reduces the columns to only important columns
        df_RBG_age = df_RBG_age[["PtID", "AgeAsOfEnrollDt"]]

        # merges dataframes of sex and age data
        df_RBG_screen = pd.merge(df_RBG_screen, df_RBG_age, on="PtID", how="inner")

        # merges dataframes of demographics and CGM data 
        df_RBG = pd.merge(df_RBG, df_RBG_screen, on=["PtID"], how="left")

        # column names are renamed for semantic equality
        df_RBG_screen = df_RBG_screen.rename(columns={"AgeAsOfEnrollDt" : "Age", "Gender": "Sex"})
        # adds the database name to the patient ID to enable reidentification 
        df_RBG["PtID"] = df_RBG["PtID"].astype(str) + "_RBG"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_RBG["Database"] = "RBG"
        return df_RBG

    def df_sence():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_SENCE = smart_read("datasets for T1D/SENCE/Data Tables/DeviceCGM.txt")

        # converts timstamps to datetime
        df_SENCE["ts"] = pd.to_datetime(df_SENCE["DeviceDtTm"])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_SENCE = df_SENCE.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "Value"))

        # reads dataframe with sex data
        df_SENCE_screen = smart_read("datasets for T1D/SENCE/Data Tables/DiabScreening.txt")
        # reduces the columns to only important columns
        df_SENCE_screen = df_SENCE_screen[["PtID", "Gender"]]

        # reads dataframe with age data
        df_SENCE_age = smart_read("datasets for T1D/SENCE/Data Tables/PtRoster.txt")
        # reduces the columns to only important columns
        df_SENCE_age = df_SENCE_age[["PtID", "AgeAsOfEnrollDt", "EnrollDt"]]

        # merges dataframes of age and sex data   
        df_SENCE_screen = pd.merge(df_SENCE_screen, df_SENCE_age, on="PtID", how="inner")

        # merges dataframes of demographics with CGM data
        df_SENCE = pd.merge(df_SENCE, df_SENCE_screen, on=["PtID"], how="left")

        # column names are renamed for semantic equality
        df_SENCE = df_SENCE.rename(columns={"Value": "GlucoseCGM", "AgeAsOfEnrollDt" : "Age", "HbA1cTestRes": "Hba1c", "Gender": "Sex"})
        # adds the database name to the patient ID to enable reidentification 
        df_SENCE["PtID"] = df_SENCE["PtID"].astype(str) + "_SENCE"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_SENCE["Database"] = "SENCE"
        return df_SENCE

    def df_shd():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_SHD = smart_read("datasets for T1D/SevereHypoDataset/Data Tables/BDataCGM.txt")

        # initial date is set
        df_SHD["initdate"] = pd.to_datetime("2023-01-01")
        # adds date with time and converts into datetime
        df_SHD["datetime"] = df_SHD["initdate"] + pd.to_timedelta(df_SHD["DeviceDaysFromEnroll"], unit="D")
        df_SHD["DeviceTm"] = df_SHD["DeviceTm"].astype(str)

        # converts timstamps to datetime
        df_SHD["ts"] = pd.to_datetime(df_SHD["datetime"].dt.strftime("%Y-%m-%d") + " " + df_SHD["DeviceTm"])

        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_SHD = df_SHD.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "Glucose"))

        # reads dataframe with sex data
        df_SHD_screen = smart_read("datasets for T1D/SevereHypoDataset/Data Tables/BDemoLifeDiabHxMgmt.txt")
        # reduces the columns to only important columns
        df_SHD_screen = df_SHD_screen[["PtID", "Gender"]]
        # true age is not given but it is told that patients are aged at least 60
        df_SHD_screen["Age"] = "60-100" 

        # merges dataframe of demographics with CGM data
        df_SHD = pd.merge(df_SHD, df_SHD_screen, on="PtID", how="left")

        # column names are renamed for semantic equality
        df_SHD = df_SHD.rename(columns={"Glucose": "GlucoseCGM", "Gender": "Sex"})
        # adds the database name to the patient ID to enable reidentification 
        df_SHD["PtID"] = df_SHD["PtID"].astype(str) + "_SHD"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_SHD["Database"] = "SHD"
        return df_SHD

    def df_wisdm():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_WISDM = smart_read("datasets for T1D/WISDM/Data Tables/DeviceCGM.txt") 

        # converts timstamps to datetime
        df_WISDM["ts"] = pd.to_datetime(df_WISDM["DeviceDtTm"])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_WISDM = df_WISDM.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "Value"))

        # reads dataframe with sex data
        df_WISDM_screen = smart_read("datasets for T1D/WISDM/Data Tables/DiabScreening.txt")
        # reduces the columns to only important columns
        df_WISDM_screen = df_WISDM_screen[["PtID", "Gender"]]

        # reads dataframe with age data
        df_WISDM_age = smart_read("datasets for T1D/WISDM/Data Tables/PtRoster.txt")
        # reduces the columns to only important columns
        df_WISDM_age = df_WISDM_age[["PtID", "AgeAsOfEnrollDt", "EnrollDt"]]

        # merge dataframes of sex and age
        df_WISDM_screen = pd.merge(df_WISDM_screen, df_WISDM_age, on="PtID", how="inner")

        # merges dataframes of demographics with CGM data
        df_WISDM = pd.merge(df_WISDM, df_WISDM_screen, on=["PtID"], how="left")

        # column names are renamed for semantic equality
        df_WISDM = df_WISDM.rename(columns={"Value": "GlucoseCGM", "AgeAsOfEnrollDt" : "Age", "Gender": "Sex", "HbA1cTestRes": "Hba1c"})
        # adds the database name to the patient ID to enable reidentification 
        df_WISDM["PtID"] = df_WISDM["PtID"].astype(str) + "_WISDM"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_WISDM["Database"] = "WISDM"
        return df_WISDM

    def df_shanghai():
        # path to the excel files
        file_paths_shang = glob.glob("datasets for T1D/shanghai/Shanghai_T1DM/*.xlsx")  # Change path accordingly

        # initializes an empty list to store dataframes
        df_list_shang = []

        # loops through each file
        for idx, file in enumerate(file_paths_shang, start=1):
            # reads the dataframes containing CGM measurements with the "smart_read()" function
            df = smart_read(file) 
            # extracts the Subject IDs from the filename
            name = os.path.splitext(os.path.basename(file))[0] 
            # assings the unique PtID
            df["PtID"] = str(name)  
            # adds the dataframe to a list
            df_list_shang.append(df)
            
        # concatenates all dataframes into one dataframe
        df_shang = pd.concat(df_list_shang, ignore_index=True)

        # path to the second set of files with the xls extension
        file_paths_shang_2 = glob.glob("datasets for T1D/shanghai/Shanghai_T1DM/*.xls")  # Change path accordingly

        # initializes an empty list to store dataframes
        df_list_shang_2  = []
        idxx = 5
        # loops through each file
        for idx, file in enumerate(file_paths_shang_2 , start=1):
            # reads the dataframes containing CGM measurements with the "smart_read()" function
            df = smart_read(file)  
            # extracts the Subject IDs from the filename
            name = os.path.splitext(os.path.basename(file))[0] 
            # assings the unique PtID
            df["PtID"] = str(name) 
            # adds the dataframe to a list
            df_list_shang_2 .append(df)
            idxx = idxx + 1
            
        # concatenates all dataframes into one dataframe
        df_shang_2  = pd.concat(df_list_shang_2, ignore_index=True)
        df_shang  = pd.concat([df_shang, df_shang_2], ignore_index=True)

        # converts timstamps to datetime
        df_shang["ts"] = pd.to_datetime(df_shang["Date"])
        # undersamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_shang = df_shang.groupby("PtID", group_keys=False).apply(lambda x: fill_gaps_sampling(x, "ts", "PtID", "CGM (mg / dl)"))
      
        # reads the dataframe with demographic data
        df_shang_info = smart_read("datasets for T1D/shanghai/Shanghai_T1DM_Summary.xlsx")
        # column names are renamed for semantic equality
        df_shang_info["Sex"] = df_shang_info["Gender (Female=1, Male=2)"].replace({2: "M", 1: "F"})
        # column names are renamed for semantic equality
        df_shang_info = df_shang_info.rename(columns={"Patient Number": "PtID"})
        # reduces the columns to only important columns
        df_shang_info = df_shang_info[["PtID", "Sex", "Age (years)"]]

        # merges dataframes with demographics and CGM data
        df_shang = pd.merge(df_shang, df_shang_info, on=["PtID"], how="left")
        # column names are renamed for semantic equality
        df_shang = df_shang.rename(columns={"CGM (mg / dl)": "GlucoseCGM", "Patient Number": "PtID", "Age (years)": "Age"})
        # adds the database name to the patient ID to enable reidentification 
        df_shang["PtID"] = df_shang["PtID"] + "_ShanghaiT1D" 
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_shang["Database"] = "ShanghaiT1D"
        return df_shang

    def df_d1namo():
        # base path 
        base_path_D1namo_ECG = "datasets for T1D/D1NAMO/diabetes_subset"

        # initiliazes list to store all dataframes
        all_data_HR = []
        all_data_GLC = []

        # loop through each subject directory 
        for subject_id in os.listdir(base_path_D1namo_ECG):

            # reads all files including CGM readings
            person_path_glc = os.path.join(base_path_D1namo_ECG, subject_id)

            # skips if not a directory
            if not os.path.isdir(person_path_glc):
                continue

            # loops through each file of the subject
            for file in os.listdir(person_path_glc):
                
                # reads the file if file is a csv and ends with glucose
                if file.endswith("glucose.csv"):
                    # joins paths 
                    file_path_glc = os.path.join(person_path_glc, file)

                    try:
                        # reads the dataframes with the CGM measurements
                        D1NAMO_glc = smart_read(file_path_glc)
                        # assigns a Subject ID
                        D1NAMO_glc["PtID"] = subject_id
                        # column names are renamed for semantic equality
                        D1NAMO_glc = D1NAMO_glc.rename(columns={"glucose": "GlucoseCGM"})
                        # converts mmol/L to mg/dL
                        D1NAMO_glc["GlucoseCGM"] = D1NAMO_glc["GlucoseCGM"] * 18.02
                        # converts timstamps to datetime
                        D1NAMO_glc["ts"] = pd.to_datetime(D1NAMO_glc["date"] + " " + D1NAMO_glc["time"])

                        # splits manual and continuous glucose data into seperate columns
                        D1NAMO_glc["mGLC"] = D1NAMO_glc["GlucoseCGM"].where(D1NAMO_glc["type"] == "manual")
                        D1NAMO_glc["GlucoseCGM"] = D1NAMO_glc["GlucoseCGM"].where(D1NAMO_glc["type"] == "cgm")
                        # add the single dataframes to the list
                        all_data_GLC.append(D1NAMO_glc)

                    except Exception as e:
                        print(f"Failed to read {file_path_glc}: {e}")

            # read the dataframes including heartrate measurements
            person_path = os.path.join(base_path_D1namo_ECG, subject_id, "sensor_data")

            if os.path.isdir(person_path):

                for session_folder in os.listdir(person_path):
                    session_path = os.path.join(person_path, session_folder)

                    # skips if not a directory
                    if not os.path.isdir(session_path):
                        continue
                    # loops through each file
                    for file in os.listdir(session_path):
                        # reads the file if it ends with _Summary.csv
                        if file.endswith("_Summary.csv"): 
                            # joins the paths
                            file_path = os.path.join(session_path, file)
                            
                            try:
                                # reads the dataframes with the heartrate values
                                D1NAMO_hr = smart_read(file_path)
                                # assigns the Subject ID 
                                D1NAMO_hr["PtID"] = subject_id
                                # adds the single dataframes to the list 
                                all_data_HR.append(D1NAMO_hr)
                            except Exception as e:
                                print(f"Failed to read {file_path}: {e}")

        # concatenates all HR dataframes into one dataframe
        df_D1NAMO_HR = pd.concat(all_data_HR, ignore_index=True)
        # concatenates all glucose dataframes into one dataframe
        df_D1NAMO_GLC = pd.concat(all_data_GLC, ignore_index=True)

        # reduces columns to only important columns
        df_D1NAMO_HR = df_D1NAMO_HR[["Time", "PtID", "HR"]]
        # converts timstamps to datetime
        df_D1NAMO_HR["ts"] = pd.to_datetime(df_D1NAMO_HR["Time"], format="%d/%m/%Y %H:%M:%S.%f")
        # the datetime needs to be converted to the same format 
        df_D1NAMO_HR["ts"] = pd.to_datetime(df_D1NAMO_HR["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
        # resamples to 1 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_D1NAMO_HR = df_D1NAMO_HR.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "PtID", value = "HR"))
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_D1NAMO_GLC = df_D1NAMO_GLC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM"))

        # merge dataframes of HR and CGM data
        df_D1NAMO = pd.merge(df_D1NAMO_GLC, df_D1NAMO_HR, on=["PtID", "ts"], how="left")
        # adds the database name to the patient ID to enable reidentification 
        df_D1NAMO["PtID"] = df_D1NAMO["PtID"] + "_D1NAMO" 
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_D1NAMO["Database"] = "D1NAMO"
        return df_D1NAMO
    

    def df_DDATSHR():
        # subjects wear the metronic or abbott CGM device, so both files are read separately
        # reads the dataframe with the CGM measurements with the "smart_read()" function 
        #  medtronic estimates glucose every 5 minutes      
        df_DDATSHR_ins = smart_read("datasets for T1D/DDATSHR/data-csv/Medtronic.csv")
        # abbott estiamtes glucose evry 15 minutes
        df_DDATSHR_glc = smart_read("datasets for T1D/DDATSHR/data-csv/Abbott.csv")

        # converts timstamps to datetime
        df_DDATSHR_glc["ts"] = pd.to_datetime(df_DDATSHR_glc["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_glc["Local time [hh:mm]"])
        # for each subject, replaces the scan values with the historic glucose values
        df_DDATSHR_glc["Historic Glucose [mmol/l]"] = df_DDATSHR_glc.groupby("Subject code number").apply(
            lambda group: group["Scan Glucose [mmol/l]"].where(
                group["Scan Glucose [mmol/l]"].notna(), group["Historic Glucose [mmol/l]"]
            )
        ).reset_index(drop=True)
        # undersamples abbott data to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_DDATSHR_glc = df_DDATSHR_glc.groupby("Subject code number", group_keys=False).apply(lambda x: fill_gaps_sampling(x, "ts", "Subject code number", "Historic Glucose [mmol/l]"))

        # removes all other timestamps which were used for insulin but have no glucose entry
        df_DDATSHR_ins = df_DDATSHR_ins.dropna(subset=["Sensor Glucose [mmol/l]"])
       
        # converts timstamps to datetime
        df_DDATSHR_ins["ts"] = pd.to_datetime(df_DDATSHR_ins["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_ins["Local time [hh:mm:ss]"])
        # column names are renamed for semantic equality
        df_DDATSHR_ins = df_DDATSHR_ins.rename(columns={"Sensor Glucose [mmol/l]": "Historic Glucose [mmol/l]"})
        # reduces the columns to only important columns
        df_DDATSHR_ins = df_DDATSHR_ins[["ts", "Subject code number", "Historic Glucose [mmol/l]"]]

        # merges dataframes of CGM measurements for both sensors into one dataframe
        df_DDATSHR_glc_ins = pd.concat([df_DDATSHR_glc, df_DDATSHR_ins])
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_DDATSHR_glc_ins = df_DDATSHR_glc_ins.groupby("Subject code number", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "Subject code number", value = "Historic Glucose [mmol/l]"))
        # converts CGM measured in mmol/L to mg/dL 
        df_DDATSHR_glc_ins["Historic Glucose [mmol/l]"] = df_DDATSHR_glc_ins["Historic Glucose [mmol/l]"] * 18.02

        # reads dataframe of age and gender 
        df_DDATSHR_info = smart_read("datasets for T1D/DDATSHR/data-csv/population.csv")
        df_DDATSHR_info = df_DDATSHR_info[["Subject code number", "Gender [M=male F=female]", "Age [yr]"]]
        # merges dataframes of demographics with CGM data
        df_DDATSHR = pd.merge(df_DDATSHR_glc_ins, df_DDATSHR_info, on=["Subject code number"], how="left")

        #reads dataframes of heartrate data 
        df_DDATSHR_hr = smart_read("datasets for T1D/DDATSHR/data-csv/Fitbit/Fitbit-heart-rate.csv")
        
        # converts timstamps to datetime
        df_DDATSHR_hr["ts"] = pd.to_datetime(df_DDATSHR_hr["Local date [yyyy-mm-dd]"] + " " + df_DDATSHR_hr["Local time [hh:mm]"])
        # resamples to 1 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_DDATSHR_hr = df_DDATSHR_hr.groupby("Subject code number", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "1min", mode="vitals", fillid = "Subject code number", value = "heart rate [#/min]"))

        # reduces the columns to only important columns
        df_DDATSHR_hr = df_DDATSHR_hr[["ts", "Subject code number", "heart rate [#/min]"]]

        # merge dataframes of heartrate and CGM data 
        df_DDATSHR = pd.merge(df_DDATSHR, df_DDATSHR_hr, on=["Subject code number", "ts"], how="left")

        # column names are renamed for semantic equality
        df_DDATSHR = df_DDATSHR.rename(columns={"Subject code number":"PtID", "Historic Glucose [mmol/l]": "GlucoseCGM", "Gender [M=male F=female]" : "Sex", "Age [yr]": "Age", "heart rate [#/min]": "HR", "steps [#]": "Steps" })
        # adds the database name to the patient ID to enable reidentification 
        df_DDATSHR["PtID"] = df_DDATSHR["PtID"].astype(str) + "_DDATSHR"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_DDATSHR["Database"] = "DDATSHR"
        return df_DDATSHR

    def df_rtc():
        # path to the excel files 
        file_paths_rtc = glob.glob("datasets for T1D/RT_CGM/DataTables/tblADataRTCGM*.csv") 

        # initializes an empty list to store dataframes
        df_list_rtc = []

        # loops through each file
        for idx, file in enumerate(file_paths_rtc, start=1):
            # reads the files with the "smart_read()" function
            df = smart_read(file)
            # adds the dataframe to the list
            df_list_rtc.append(df)

        # concatenates all dataframes into one dataframe
        df_rtc  = pd.concat(df_list_rtc , ignore_index=True)

        # converts timstamps to datetime depending on the format
        try: 
            df_rtc["ts"] = pd.to_datetime(df_rtc["DeviceDtTm"].str.split(".").str[0], format="%Y-%m-%d %H:%M:%S")
        except:
            df_rtc["ts"] = pd.to_datetime(df_rtc["DeviceDtTm"], format="%Y-%m-%d %H:%M:%S")

        # detects the sample rate since some subjects have glucose collected in 10 minute intervals; this is done separately for each subject
        fre_rtc = df_rtc.groupby("PtID", group_keys=False).apply(lambda x: detect_sample_rate(x, time_col = "ts")).reset_index(name="Frequency")
        # adds teh frequency column to the dataframe with CGM measurements
        df_rtc = df_rtc.merge(fre_rtc, on="PtID", how="left")

        # fills missing values with the feedforward method in the "Frequency" column
        df_rtc["Frequency"] = df_rtc["Frequency"].fillna(method="ffill")

        # assigns condition of 10 minutes to resample those seperately
        condition = df_rtc["Frequency"] == "10 min"
        # splits the dataframe based on the condition
        df_10min = df_rtc[condition]   
        df_5min = df_rtc[~condition] 

        # undersamples CGM values with 10 minute measurement frequency to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_10min = df_10min.groupby("PtID", group_keys=False).apply(lambda x: fill_gaps_sampling(x, "ts", "PtID", "Glucose", 10))
        # concatenates both dataframes with separate frequencies again into one database
        df_RTC = pd.concat([df_10min, df_5min], ignore_index=True)
        # resamples the whole dataframe to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_RTC = df_RTC.groupby("PtID", group_keys=False).apply(lambda x: df_resample(x, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "Glucose"))

        # reads dataframe with demographics
        df_rtc_info = smart_read("datasets for T1D/RT_CGM/DataTables/tblAPtSummary.csv")
        # reduces the columns to only important columns
        df_rtc_info = df_rtc_info[["PtID", "Gender", "AgeAsOfRandDt"]]

        # merges dataframes of demographics with CGM data
        df_RTC = pd.merge(df_RTC, df_rtc_info, on=["PtID"], how="left")

        # column names are renamed for semantic equality
        df_RTC = df_RTC.rename(columns={"Glucose": "GlucoseCGM", "AgeAsOfRandDt" : "Age", "Gender": "Sex"})
        # adds the database name to the patient ID to enable reidentification 
        df_RTC["PtID"] = df_RTC["PtID"].astype(str) + "_RT-CGM"
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_RTC["Database"] = "RT-CGM"
        return df_RTC
    
    
    def df_T1GDUJA():
        # reads the dataframe with the CGM measurements with the "smart_read()" function
        df_T1G = smart_read("datasets for T1D/T1GDUJA/glucose_data.csv")

        # converts timstamps to datetime depending on the format
        try: 
            df_T1G["ts"] = pd.to_datetime(df_T1G["date"].str.split(".").str[0], format="%Y-%m-%d %H:%M:%S")
        except:
            df_T1G["ts"] = pd.to_datetime(df_T1G["date"], format="%Y-%m-%d %H:%M:%S")
        # column names are renamed for semantic equality
        df_T1G = df_T1G.rename(columns={"sgv": "GlucoseCGM"})
        # adds the database name to the patient ID to enable reidentification 
        df_T1G["PtID"] = "T1GDUJA"
        # resamples to 5 minute intervals to have unifrom sample rate; this is done for each subject seperately
        df_T1G = df_resample(df_T1G, timestamp = "ts", frequency= "5min", mode="glucose", fillid = "PtID", value = "GlucoseCGM")
        # adds a "Database" column with the name of the Dataset to enable reidentification
        df_T1G["Database"] = "T1GDUJA"
        return df_T1G

    # this function calls the target functions reading the datasets which are given as a list and returns them as a list of dataframes 
    def try_call_functions(functions):
        # dictionary is initialized
        combined_df_dict = {}
        # each function is called seperately
        for func in functions:
            # try to read the data 
            try:
                dataframe = func()
                combined_df_dict[func.__name__] = dataframe
            # if an error occurs print error and continue to read the next function
            except Exception as e:
                print(f"Error in {func.__name__}(): {e}")
        # dataframes are combined and stored as a list 
        combined_df_list = list(combined_df_dict.values())
        return combined_df_list
    
    # if the integrated datasets of publicly available databases are found, only readsrestricted datasets
    if (read_all == False):
        datasets = [df_granada, df_diatrend]
    # if the integrated dataset is not found, reads all datasets separately
    elif(read_all == True):
        datasets = [df_granada, df_diatrend, df_city, df_dclp, df_pedap, df_replace, df_sence, df_shd, df_wisdm, df_shanghai, df_hupa,df_d1namo, df_DDATSHR, df_rtc, df_T1GDUJA]
    
    # calls all functions to return a list of all datasets 
    combined_df_list = try_call_functions(datasets)
    # returns a list of dataframes
    return combined_df_list



def combine_data(modus, restricted_list, columns_to_check = ["Age", "Sex"]):
    """
    This function integrates all dataframes into one database depending on the defined modus 
    Parameters:
    - modus: can be either 1, 2, or 3. 
        - 1 is for the main database and integrates all dataframes which include CGM values
        - 2 is for subdatabase I and integrates all dataframes which include CGM values and demographics of age and sex
        - 3 is for subdatabase II and integrates all dataframes which include CGM values and HR data
    - retricted_list is the list of datasets which were not shared due to licensing restrictions. These need to be loaded and integrated seperately
    - columns_to_check are default values to set age groups if modus two is applied
    Output: returns the integrated subset of restricted data 
    """
    # this function takes the original database as input and converts the value of the "Age" column into integers
    def set_ages(df, column = "Age"):
        
        # converts each string to a numeric value
        def to_numeric(val):
            if isinstance(val, str) and "-" in val:
                # removes any extra words after the age range
                val = val.replace("yrs", "")  
                val = val.strip()  
                # cleans up any extra spaces
                start, end = map(int, val.replace(" ", "").split("-"))
                # or start, or end
                return (start + end) / 2  
            # returns the numeric value
            return int(val)

        # defines the set of all ages 
        all_ages = set(df[column])

        # initializes one empty lists for the ages reported as strings
        Age_str = []

        # iterates over the set of reported ages
        for value in all_ages:
            # appends each age reported as an age range as a string to the list
            if isinstance(value, str) or isinstance(value, object):
                Age_str.append(str(value)) 

        # copies the original database
        df = df.copy()
        # converts each age range into a numeric value
        df[column] = df[column].apply(to_numeric)

        # creates age groups based on defined bins
        bins = [0, 2,6, 10, 13, 17,25, 35, 55, 100]
        labels = ["0-2", "3-6", "7-10", "11-13", "14-17", "18-25", "26-35", "36-55", "56+"]
        
        # categorizes the ages of the "Age" colum into the defined age groups and assign them to new column "AgeGroup"
        df["AgeGroup"] = pd.cut(df[column], bins=bins, labels=labels, right=True)
        
        # returns the dataframe with the new colum "AgeGroup"
        return df

    # this function concatenates the dataframes 
    def concat_rows_on_columns(dfs, columns):
    
        # selects only the specified columns from each dataframe
        dfs = [df[columns] for df in dfs]
        # concatenates all dataframes vertically 
        result = pd.concat(dfs, ignore_index=True)
        # removes subjects who only includes nan values in the "GlucoseCGM" column
        subjects_to_keep = result.groupby("PtID")["GlucoseCGM"].transform(lambda x: not x.isna().all())
        # only includes subjects with columns
        result = result[subjects_to_keep]

        # if columns to check are all within the dataframe columns 
        if all(col in result.columns for col in columns_to_check):
            # removes nan values since the final datasets should have all columns included for each subject
            df_cleaned = result.dropna(subset=columns_to_check)
            # groups ages into age ranges 
            df_cleaned = set_ages(df_cleaned)
            # returns the cleaned and preprocessed dataset
            return df_cleaned
        else:
            # returns the cleaned and preprocessed dataset
            return result

    # either 1: all CGM values, 2: CGM and demographics, or 3: CGM and HR
    allowed_values = [1, 2, 3] 
    # for a wrong input, output a warning. 
    if modus not in allowed_values:
        print("Invalid input. Please enter: 1, 2, 3 (1: all CGM values, 2: CGM and demographics, or 3: CGM and HR).")
        return
    # based on the modus, the columns to keep are specified
    if modus == 1: 
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "Database"]
    elif modus == 2:
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "Age", "Sex", "Database"] 
    else:
        columns_to_keep = ["ts", "PtID", "GlucoseCGM", "HR", "Database"]
    
    # takes only the subset of columns of interest
    filtered_dfs = [df for df in restricted_list if all(col in df.columns for col in columns_to_keep)]
    # calls the "concat_rows_on_columns()"" function
    combined_df = concat_rows_on_columns(filtered_dfs, columns=columns_to_keep)

    # returns the combined_df
    return combined_df