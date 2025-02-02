import pandas as pd


def getvaluecounts(df):

    return dict(df['subject'].value_counts())


def getlevelcount(df):

    return dict(list(df.groupby(['level'])['num_subscribers'].count().items())[1:])


def getsubjectsperlevel(df):

    ans = list(dict(df.groupby(['subject'])['level'].value_counts()).keys())
    alllabels = [ans[i][0]+'_'+ans[i][1] for i in range(len(ans))]
    ansvalues = list(dict(df.groupby(['subject'])[
                     'level'].value_counts()).values())

    completedict = dict(zip(alllabels, ansvalues))

    return completedict


def yearwiseprofit(df):
    df['price'] = df['price'].astype(str)  # Ensure all values are strings
    df['price'] = df['price'].str.replace('TRUE|Free', '0', regex=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)  # Convert to float safely
    df['profit'] = df['price'] * df['num_subscribers']

    # Converting timestamps to datetime format
    df['published_date'] = pd.to_datetime(df['published_timestamp'].str.split('T').str[0], errors='coerce')

    # Drop invalid index (ensure index 2066 exists first)
    if 2066 in df.index:
        df = df.drop(2066)

    df['Year'] = df['published_date'].dt.year
    df['Month'] = df['published_date'].dt.month
    df['Day'] = df['published_date'].dt.day
    df['Month_name'] = df['published_date'].dt.month_name()

    profitmap = df.groupby('Year')['profit'].sum().to_dict()
    subscribersmap = df.groupby('Year')['num_subscribers'].sum().to_dict()
    profitmonthwise = df.groupby('Month_name')['profit'].sum().to_dict()
    monthwisesub = df.groupby('Month_name')['num_subscribers'].sum().to_dict()

    return profitmap, subscribersmap, profitmonthwise, monthwisesub
