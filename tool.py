#!/usr/bin/env python3

import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")


# List
# Pandas 관련 Function
# 모델 관련 Function
# AWS S3 관련 Function
# Plot Function
# MySql Function
# 기타 Function


##############################
# Pandas 관련 Function
##############################
def get_df_fill_date(df, date_col):
    """
    DataFrame의 빈 날짜를 채워주고 Pandas DataFrame로 리턴
    """
    from pandas.api.types import is_datetime64_any_dtype as is_datetime
    # 컬럼의 데이터타입이 date타입인지 체크
    if not is_datetime(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    df2 = pd.DataFrame({date_col: pd.date_range(min(df[date_col]), max(df[date_col]))})

    # Merge
    merged_data = pd.merge(df, df2, how='right', on=date_col)

    return merged_data


def get_scaling_data(df, col_list, option='train', scaler=None, scale_method='StandardScaler'):
    """
    sklearn을 이용하여 데이터 스케일링을 하고 Pandas DataFrame로 리턴
        :parameter
            option : train or test
            scaler : 테스트셋의 경우 fit 된 scaler을 넘겨줌
            scale_method : StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
        :return
            df_data_scl : scale 된 데이터셋
            data_scaler : scaler 리스트
    """

    if scale_method == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
    elif scale_method == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
    elif scale_method == "MaxAbsScaler":
        from sklearn.preprocessing import MaxAbsScaler
    elif scale_method == "RobustScaler":
        from sklearn.preprocessing import RobustScaler

    if option == "train":
        data_scaler = []
    elif option == "test":
        data_scaler = scaler

    data_scl = []

    def scale_and_transform(data, scaler=None):
        data_ = np.array(data)

        if option == 'train':
            if scale_method == "StandardScaler":
                data_scaler = StandardScaler()
            elif scale_method == "MinMaxScaler":
                data_scaler = MinMaxScaler()
            elif scale_method == "MaxAbsScaler":
                data_scaler = MaxAbsScaler()
            elif scale_method == "RobustScaler":
                data_scaler = RobustScaler()

            data_scaler.fit(data_.reshape((-1, 1)))
        else:
            data_scaler = scaler  # 학습이 아닌 경우 학습에서 사용한 스케일러 사용

        data_ = data_scaler.transform(data_.reshape((-1, 1)))
        data_ = data_.reshape(-1)

        return data_, data_scaler

    for i in range(0, len(col_list)):
        if option == 'train':
            data_, data_scaler_ = scale_and_transform(data=df[col_list[i]])
            data_scaler.append(data_scaler_)
        else:
            data_, data_scaler_ = scale_and_transform(data=df[col_list[i]], scaler=data_scaler[i])
        data_scl.append(data_)

    df_data_scl = pd.DataFrame(data=data_scl).transpose()
    df_data_scl.columns = col_list
    df_data_scl.set_index(df.index, inplace=True)

    return df_data_scl, data_scaler


def get_inverse_scale(df_scl, col_list, scaler):
    df_scaler = scaler

    data = []

    for i in range(0, len(col_list)):
        np_data = np.array(df_scl[col_list[i]]).reshape((-1, 1))
        inverse_data = df_scaler[i].inverse_transform(np_data)
        data.append(inverse_data.reshape(-1))

    inverse_df = pd.DataFrame(data=data).transpose()
    inverse_df.columns = col_list
    inverse_df.set_index(df_scl.index, inplace=True)

    return inverse_df


def get_onehot_encoding(df, col_list, option='train', encoder=None, method='sklearn'):
    """
    pandas get_dummies를 이용하여 Ont Hot 인코딩 후 데이터 리턴
        :parameter
            dummies : pandas get_dummies를 이용한 원핫인코딩
            sklearn : sklearn을 이용한 원핫인코딩
    """

    if method == 'dummies':
        dummies = [pd.get_dummies(df[column], prefix=column) for column in col_list]

        return pd.concat([df, *dummies], axis=1).drop(columns=col_list)
    else:
        from sklearn.preprocessing import OneHotEncoder

        if option == "train":
            data_encoder = []
        elif option == "test":
            data_encoder = encoder

        df_data_onthot = []

        _ = lambda col, x: col + '-' + x
        fun_concat = np.vectorize(_)

        def encode_and_transform(data, encoder=None):
            data_ = np.array(data)

            if option == 'train':
                data_encoder = OneHotEncoder(handle_unknown='ignore')
                data_encoder.fit(data_.reshape((-1, 1)))
            else:
                data_encoder = encoder  # 학습이 아닌 경우 학습에서 사용한 스케일러 사용

            data_ = data_encoder.transform(data_.reshape((-1, 1)))

            return data_, data_encoder

        for i in range(0, len(col_list)):
            if option == 'train':
                data_, data_encoder_ = encode_and_transform(data=df[col_list[i]])
                data_encoder.append(data_encoder_)
            else:
                data_, data_encoder_ = encode_and_transform(data=df[col_list[i]], encoder=data_encoder[i])

            df_data_onthot.append(pd.DataFrame(data_.toarray(),
                                               index=df.index,
                                               columns=fun_concat(col_list[i], data_encoder_.categories_[0])
                                               )
                                  )

        return pd.concat([df, *df_data_onthot], axis=1).drop(columns=col_list), data_encoder


def get_corr(df, option=1, figsize=(14, 14)):
    """
    상관계수를 출력한다.
        :parameter
            option : 1(상관계수), 2(결정계수)
    """
    if option == 2:
        df_corr = df.corr() ** 2
    else:
        df_corr = df.corr()

    print(df_corr)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data=df_corr,
                annot=True,
                cmap='RdYlBu_r',
                linewidths=.5,  # 경계면 실선으로 구분하기
                cbar_kws={"shrink": .5},  # 컬러바 크기 절반으로 줄이기
                vmin=-1, vmax=1  # 컬러바 범위 -1 ~ 1
                )

    plt.show()


def get_column_list(df):
    """
    DataFrame의 컬럼 리스트를 Dictionary로 리턴
    """
    category_list = [column for column, dtype in df.dtypes.items() if str(dtype) in ["object", "category"]]
    numeric_list = [column for column, dtype in df.dtypes.items() if dtype in ["int", "float"]]

    column_lst_schema = {"numeric": numeric_list,
                         "category": category_list
                         }

    return column_lst_schema


##############################
# 모델 관련 Function
##############################
def get_statemodel_summary(feature, target):
    """
    StatsModel을 이용하여 Model Summary 출력
    """
    import statsmodels.api as sm

    train_featrue = sm.add_constant(feature)  # constant 값 추가
    model = sm.OLS(target, train_featrue).fit()  # 모델 학습

    print(model.summary2())


def get_vif(df, mode="single"):
    """
    다중공선성 체크를 위해 Variance inflation factor 리턴
        :parameter
            mode : single or all
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # 모든 feature 조합으로 VIF 계산
    if mode == "all":
        from itertools import combinations

        col_list = df.columns
        list_len = len(col_list)

        result = pd.DataFrame(columns=col_list)

        while True:
            for li in list(combinations((col_list), list_len)):
                df_temp = df[list(li)]

                df_vif = pd.DataFrame(
                    {"VIF": [variance_inflation_factor(df_temp.values, idx) for idx in range(df_temp.shape[-1])],
                     "features": df_temp.columns,
                     "i": 1
                     }).pivot(index="i", columns='features', values='VIF').reset_index(level=0).drop(columns=['i'])

                result = pd.concat([result, df_vif])

            list_len -= 1

            if list_len == 1:
                break

        result = result.reset_index(drop=True)
        result['MEAN'] = result.apply(np.mean, axis=1)
    else:
        result = pd.DataFrame({
            "VIF Factor": [variance_inflation_factor(df.values, idx) for idx in range(df.shape[1])],
            "features": df.columns,
        })

    return result


##############################
# AWS S3 관련 Function
##############################

def read_s3_csv_df(bucket_name, access_key, secret_key, path, dtype_dic=None):
    """
    S3의 CSV 파일을 읽어 Pandas DataFrame로 리턴
        :rtype: object
        :parameter
            bucket_name : S3 Bucket name
            access_key : S3 access key
            aws_secret_access_key : S3 secret access key
            path : s3 file path
            dtype_dic : pandas data type dictionary. ex) {"COL_NAME" : str}
        :return
            pandas DataFrame
    """
    import boto3

    # S3 접속을 위한 환경 설정
    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key
                          )

    response = client.list_objects(Bucket=bucket_name, Prefix=path)
    prefix_df = []
    for content in response['Contents']:
        if "_SUCCESS" not in content['Key']:
            obj_dict = client.get_object(Bucket=bucket_name, Key=content['Key'])
            body = obj_dict['Body']

    return pd.read_csv(body, dtype=dtype_dic)


def read_s3_csv_folder_df(bucket_name, access_key, secret_key, path, dtype_dic=None):
    """
    S3 특정 경로의 모든 CSV 파일을 읽어 Pandas DataFrame로 리턴
        :parameter
            bucket_name : S3 Bucket name
            access_key : S3 access key
            aws_secret_access_key : S3 secret access key
            path : s3 file path
            dtype_dic : pandas data type dictionary. ex) {"COL_NAME" : str}
        :return
            pandas DataFrame
    """
    import boto3

    # S3 접속을 위한 환경 설정
    session = boto3.Session(aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key
                            )

    s3 = session.resource('s3')

    bucket = s3.Bucket(bucket_name)

    df_list = []

    for obj in bucket.objects.filter(Prefix=path):
        key = obj.key
        if ('.csv' in key) & (path in key):
            df = read_s3_csv_df(bucket_name, access_key, secret_key, key, dtype_dic=dtype_dic)
            df_list.append(df)

    return pd.concat(df_list, axis=0, ignore_index=True)


def write_s3_csv_df(bucket_name, access_key, secret_key, path, df):
    """
    Pandas DataFrame을 S3에 CSV로 저장
        :parameter
            bucket_name : S3 Bucket name
            access_key : S3 access key
            aws_secret_access_key : S3 secret access key
            path : s3 file path
            df : pandas DataFrame
    """
    import boto3
    from io import StringIO

    # S3 접속을 위한 환경 설정
    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key
                          )

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3'
                                 ,aws_access_key_id=access_key
                                 ,aws_secret_access_key=secret_key
                                 )
    s3_resource.Object(bucket_name, path).put(Body=csv_buffer.getvalue())

def delete_s3_file(bucket_name, access_key, secret_key, path):
    """
    S3의 특정 폴더의 파일을 모두 삭제
        :parameter
            bucket_name : S3 Bucket name
            access_key : S3 access key
            aws_secret_access_key : S3 secret access key
            path : s3 file path
    """
    import boto3

    # S3 접속을 위한 환경 설정
    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key
                          )

    s3filelist = client.list_objects_v2(Bucket=bucket_name, Prefix=path)['Contents']
    filestodel = [filename['Key'] for filename in s3filelist]
    for file in filestodel:
        response = client.delete_object(Bucket=bucket_name, Key=file)



##############################
# Plot Function
##############################

def plot_train_history(history, title=None):
    import matplotlib.pyplot as plt
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def print_import():
    print("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
    """)

    
##############################
# MySql Function
##############################
def get_mysql_conn(connect=True):
    from sqlalchemy import create_engine
    import pymysql
    pymysql.install_as_MySQLdb()

    # credentials to create database connections
    db_driver = 'mysql+mysqldb'
    db_username = 'stock'
    db_password = 'stock'
    db_ipaddress = 'mariadb'
    db_port = '3306'
    db_dbname = 'stock'

    # database connection ... refresh last line before each dataframe read
    str_mariadb_con = f'{db_driver}://{db_username}:{db_password}@{db_ipaddress}:{db_port}/{db_dbname}'
    mariadb_engine = create_engine(str_mariadb_con)
    
    if connect:
        mariadb_connection = mariadb_engine.connect()
        return mariadb_connection
    else:
        return mariadb_engine

def get_mysql_data(sql):
    conn = get_mysql_conn()
    
    from sqlalchemy import text as sql_text
    
    df = pd.read_sql_query(sql_text(sql), conn)

    conn.close()
    
    return df

def save_mysql_data(df, table_name):
    engine = get_mysql_conn(connect=False)
    
    with engine.begin() as connection:
        df.to_sql(table_name, engine, if_exists='append', index=False)
        engine.dispose()
        
def delete_mysql(table_name):
    from sqlalchemy import text as sql_text
    
    engine = get_mysql_conn()
    engine.execute(sql_text("TRUNCATE TABLE " + table_name))

    engine.close()

def sql_mysql(sql):
    from sqlalchemy import text as sql_text
    
    engine = get_mysql_conn()
    engine.execute(sql_text(sql))

    engine.close()






##############################
# 기타 Function
##############################

import time
from functools import wraps

# retry decorator
def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param logger: logger to use. If None, print
    :type logger: logging.Logger instance
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry
