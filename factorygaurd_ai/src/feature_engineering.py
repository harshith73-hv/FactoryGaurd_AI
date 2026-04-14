import pandas as pd

def load_and_process_data(file_path):

    #------------------------LOAD DATA------------------------
    df = pd.read_csv(file_path)
    print("Data loaded successfully. Shape:", df.shape)
    print("Columns:\n", df.columns)

    #------------------------CLEAN COLUMN NAMES------------------------
    df.columns = df.columns.str.replace(' ', '_').str.replace('[', '').str.replace(']', '').str.replace('(', '').str.replace(')', '')

    #------------------------DROP USELESS COLUMNS------------------------
    df = df.drop(['UDI', 'Product_ID'], axis=1, errors='ignore')  

     # ---------------------- BASIC INFO ----------------------
    print("Columns:\n", df.columns)
    print("\nMissing values:\n", df.isnull().sum())


    #----------------------Convert categorical to numeric-----------------
    df = pd.get_dummies(df, columns=['Type'], drop_first=True)  

                            # FEATURE ENGINEERING

    #-------------------------ROLLING FEATURES-------------------------------
    df['temp_mean_6'] = df['Air_temperature_K'].rolling(6).mean()
    df['temp_std_6'] = df['Air_temperature_K'].rolling(6).std()

    df['torque_mean_6'] = df['Torque_Nm'].rolling(6).mean()
    df['torque_std_6'] = df['Torque_Nm'].rolling(6).std()

    #-------------------------EMA-------------------------------
    df['temp_ema_6'] = df['Air_temperature_K'].ewm(span=6, adjust=False).mean()

    #-------------------------LAG FEATURES-------------------------------
    df['temp_lag_1'] = df['Air_temperature_K'].shift(1)
    df['temp_lag_2'] = df['Air_temperature_K'].shift(2)

    df['torque_lag_1'] = df['Torque_Nm'].shift(1)

     # ---------------------- BASIC INFO ----------------------
    print("Columns:\n", df.columns)
    print("\nMissing values:\n", df.isnull().sum())

    #-------------------------REMOVE NaN VALUES-------------------------------
    df = df.drop(['TWF','HDF','PWF','OSF','RNF'], axis=1, errors='ignore')  
    df = df.dropna()

    return df

# Run only when file is executed directly
if __name__ == "__main__":
    filepath = 'data/data.csv'
    df = load_and_process_data(filepath)

    print("Data loaded and processed successfully.\n")
    print("Columns:\n", list(df.columns))
    