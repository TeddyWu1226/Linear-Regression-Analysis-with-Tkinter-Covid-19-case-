"""
ver.2
作者:Cheng hong Wu
資料來源:https://www.kaggle.com/code/therealcyberlord/coronavirus-covid-19-visualization-prediction/notebook
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import tkinter.messagebox as tkMessageBox
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
from tkinter import StringVar
from tkinter import LEFT, TOP, X, FLAT, RAISED

# 確認是否存在data 若無則下載下來
"""
print("time_series_covid19_confirmed_global.csv資料更新")
confirmed_df = pd.read_csv(
'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
confirmed_df.to_csv("time_series_covid19_confirmed_global.csv")

print("time_series_covid19_deaths_global.csv資料更新")
deaths_df = pd.read_csv(
'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
deaths_df.to_csv("time_series_covid19_deaths_global.csv")
"""

# 讀取資料
confirmed_df = pd.read_csv("time_series_covid19_confirmed_global.csv")
deaths_df = pd.read_csv("time_series_covid19_deaths_global.csv")
# 檢查資料讀取情況
# print(confirmed_df,"\n\n",deaths_df,"\n\n",latest_data)
# 資料設定變數
col_Name = confirmed_df.keys() # col的名稱
# ※[2]是國家  ※[3][4]是經緯度 ※[5]開始是日期
data_Confirmed = confirmed_df.loc[:, col_Name[5]:] # 取得 各國地區的確診人數
data_Deaths = deaths_df.loc[:, col_Name[5]:] # 取得 各國地區的死亡人數

# 全球情況-總確診與死亡人數統計(累計)
global_Confirmed_List = []
global_Deaths_List = []
for x in col_Name[5:]:
    global_Confirmed_List.append(data_Confirmed[x].sum())
    global_Deaths_List.append(data_Deaths[x].sum())
# 每日增新病例/死亡數量
def daily_increase(data):
    daily_Change_List = [0]
    for i in range(len(data)):
        if i == len(data)-1:
            break
        else:
            daily_Change_List.append(data[i+1]-data[i])
    return daily_Change_List
daily_Increase_Confirmed_List = daily_increase(global_Confirmed_List)
daily_Increase_Deaths_List = daily_increase(global_Deaths_List)

def moving_increase(data,breadth=1):
    daily_Change_List = []
    for i in range(len(data)):
        if i == len(data)-breadth+1:
            break
        else:
            daily_Change_List.append(sum(data[i:i+breadth])/breadth)
    return daily_Change_List
# print(df)
# print(f"確診案例每周平均增加數:{moving_increase(global_Confirmed_List,7)}")
# print(f"確診死亡案例每周平均增加數:{moving_increase(global_Deaths_List,7)}")

import numpy as np
# 調整維度(矩陣化)
npDates = np.array([i for i in range(len(col_Name[5:]))]).reshape(-1, 1) # 日期變編號
npConfirmedCase = np.array(global_Confirmed_List).reshape(-1, 1)
npDeathsCase = np.array(global_Deaths_List).reshape(-1, 1)
npIncreaseConfirmedCase = np.array(daily_Increase_Confirmed_List).reshape(-1, 1)
npIncreaseDeathsCase = np.array(daily_Increase_Deaths_List).reshape(-1, 1)

# 視窗化
win = tk.Tk()

# 視窗大小、能否縮放
win.geometry("600x600")
win.resizable(width=False, height=False)
win.wm_title("Covid-19 Linear-Regression model testing Application")


# 文字內容
url = tk.Label(win, text="Data from: \nhttps://www.kaggle.com/code/therealcyberlord/coronavirus-covid-19-visualization-prediction/notebook\n", font=("Arial", 10))
url.pack()

def downloadData():
    print("time_series_covid19_confirmed_global.csv資料更新")
    confirmed_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    confirmed_df.to_csv("time_series_covid19_confirmed_global.csv")

    print("time_series_covid19_deaths_global.csv資料更新")
    deaths_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    deaths_df.to_csv("time_series_covid19_deaths_global.csv")

    messagebox.showinfo("Finish!", "Downloaded Successfully")
Download = tk.Button(win,text="Download the new data",command=downloadData)
Download.pack()
DataUpateDates = tk.Label(win, text=f"Date of Data: From {col_Name[5]} to {col_Name[-1]}", font=("Arial", 15))
DataUpateDates.pack()
DataGlobalConfirmed = tk.Label(win, text=f"Global Confirmed Cases: {format(global_Confirmed_List[-1],',d')}(per)", font=("Arial", 15))
DataGlobalConfirmed.pack()
DataGlobalDeaths = tk.Label(win, text=f"Global Death Cases: {format(global_Deaths_List[-1],',d')}(per)", font=("Arial", 15))
DataGlobalDeaths.pack()
ModelSetting = tk.Label(win, text="\nPredicted Linear-Regression Model Setting", font=("Arial", 20))
ModelSetting.pack()

dataChosenBox = tk.Listbox(win,width=30,height=4,selectmode="browse")
dataChosenBox.insert(tk.END,"Daily Confirmed Cases(Cumulative)","Daily Death Cases(Cumulative)","Daily Confirmed Cases","Daily Death Cases")
dataChosenBox.pack()

toolbar = tk.Frame(win,bd=3,  relief=RAISED)             # relief樣式
toolbar.pack()

txt = tk.Label(toolbar, text="Predicted Model Power:", font=("Arial", 10))
txt.pack()
txtPower = StringVar(value="1")        # 變數
textPower = ttk.Spinbox(toolbar,textvariable=txtPower,width=20,from_=1,to=100)
textPower.pack()    # 位置

txt = tk.Label(toolbar, text="Test Size", font=("Arial", 10))
txt.pack()
txtTestSize = StringVar(value="0.01")        # 變數
textTestSize = ttk.Spinbox(toolbar,textvariable=txtTestSize,width=20,from_=0.01,to=1,increment=0.01)
textTestSize.pack()    # 位置

txt = tk.Label(toolbar, text="Future Prediction(days):", font=("Arial", 10))
txt.pack()
txtPrediction = StringVar(value="1")        # 變數
textPrediction = ttk.Spinbox(toolbar,textvariable=txtPrediction,width=20,from_=1,to=1000)
textPrediction.pack()    # 位置
ShuffleStatus = ttk.LabelFrame(toolbar, text="Shuffle Status",width=20,height=40)
ShuffleStatus.pack()
ShuffleValue = tk.StringVar()
ShuffleValue.set("False")
Shuffle0 = tk.Radiobutton(ShuffleStatus, text="True",variable=ShuffleValue, value="True")   # True
Shuffle1 = tk.Radiobutton(ShuffleStatus, text="False",variable=ShuffleValue, value="False")    # False
Shuffle0.pack()
Shuffle1.pack()
ChosenList = npConfirmedCase
dataName = "Daily Confirmed Cases(Cumulative)"
def plt_update_data(plot, xdata, ydata):
    plot.set_xdata(xdata)
    plot.set_ydata(ydata)
    plt.draw()

def Diagram():
    ChosenList = npConfirmedCase
    dataName = "Daily Confirmed Cases(Cumulative)"
    divide = 10000
    if not dataChosenBox.curselection():
        messagebox.showinfo("Data is empty!", "Please chose one kind of data!")
    else:
        if dataChosenBox.curselection()[0] == 1:
            ChosenList = npDeathsCase
            dataName = "Daily Death Cases(Cumulative)"
        elif dataChosenBox.curselection()[0] == 2:
            ChosenList = npIncreaseConfirmedCase
            dataName = "Daily Confirmed Cases"
            divide = 1000
        elif dataChosenBox.curselection()[0] == 3:
            ChosenList = npIncreaseDeathsCase
            dataName = "Daily Death Cases"
            divide = 1000

        else:
            pass
        x_train, x_test, y_train, y_test = \
            train_test_split(npDates, ChosenList, test_size=float(textTestSize.get()), shuffle=ShuffleValue.get())
        # test_size設定資料訓練規模 shuffle設定資料是否重組

        poly_model = make_pipeline(PolynomialFeatures(int(textPower.get())), LinearRegression())
        # PolynomialFeatures()中填入多少 就代表多少次方的方程式

        poly_model.fit(x_train, y_train)

        # 未來預估
        days_in_future = int(textPrediction.get())
        future_forcast = np.array([i for i in range(len(col_Name[5:]) + days_in_future)]).reshape(-1, 1)
        adjusted_dates = future_forcast[days_in_future:]

        xfit = adjusted_dates
        yfit = poly_model.predict(adjusted_dates)

        # 製圖
        plt.cla()
        plt.title(f"Test Model Diagram-{dataName}")
        plt1 = plt.scatter(x_train, y_train / divide, s=4)
        plt2 = plt.plot(xfit, yfit / divide, color="Red")
        plt.legend(["Test Data", "Predictions"])
        plt.xlabel("Days")
        plt.ylabel(f"Number(multiplier:{divide})")
        plt.show()

chartButton = tk.Button(toolbar,text="Diagram Show",command=Diagram)
chartButton.pack()

Author = tk.Label(win, text="Author:Cheng Hong,Wu", font=("Arial", 10))
Author.pack(side='bottom')

# END
def closePlt():
    plt.close('all')
    win.destroy()
win.protocol("WM_DELETE_WINDOW",closePlt)
win.mainloop()
