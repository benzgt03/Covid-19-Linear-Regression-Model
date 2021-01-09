import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd  # import libary ต่างๆที่จะใช้


covid_19 = pd.read_excel(r'C:\Users\Admin\Desktop\PortGT03\Covid19.xlsx')
# print (covid_19)
df = pd.DataFrame(covid_19)
# print(df) uncomment to show data
clean_df1 = df[df['age'] > 0]
df_b = df.groupby('announce_date').count()
df_s = df.groupby(['announce_date', 'sex']).count()
df_p = df.groupby('nationality').count()
list_s = df_s['no'].tolist()
list_b = df_b['no'].tolist()
list_p = df_p['no'].tolist()
list_idx = df_b.index.tolist()
list_idx1 = df_p.index.tolist()
list_idx.pop()
list_b.pop()
list_s.pop()
# ค่าสถิติในด้านต่างๆ
# print(df_s['no'])
print(df.head())
print(clean_df1.describe())
print('ค่าเฉลี่ยอายุคนที่ติดCovid19 =',
      round(clean_df1['age'].mean()))  # ค่าเฉลี่ยอายุคนที่ติดCovid ในไทย # ใช้function round ในการปัดเศษ
print('อายุสูงสุดที่ติดCovid19 =', round(clean_df1['age'].max()))  # คนที่อายุสูงที่สุดที่ติด Covid ในไทย
print('อายุน้อยสุดที่ติดCovid19 =', round(clean_df1['age'].min()))  # อายุน้อยที่สุด
print('SD อายุ =', round(clean_df1['age'].std()))  # ค่าส่วนเบี่ยงเบนมาตรฐาน
print('จำนวนผช =', df[df['sex'] == 'ชาย'].count()['sex'])
print('จำนวนผญ =', df[df['sex'] == 'หญิง'].count()['sex'])
print('จำนวนทั้งหมด =', df['sex'].count())
print('จำนวนวัน =', df['announce_date'].nunique())

# อธิบายข้อมูลในList
# print(list_b)
# print(list_idx)
# print(df_b)


print("วันที่ติดเชื้อเยอะที่สุด =", df_b.no.idxmax(), "\nจำนวนที่ติด =", df_b.no.max(), "คน")
print("วันที่ติดเชื้อน้อยที่สุด =", df_b.no.idxmin(), "\nจำนวนที่ติด =", df_b.no.min(), "คน")
print("วันที่มีคนติดเชื้อในไทยเป็นวันแรก =", df_b.no.first_valid_index())

# Insertplot อย่างเดียว Covid19 Uncomment เพื่อเรียกดู
# fig = plt.figure()
# axes = fig.add_axes([0.1,0.1,0.8,0.8]) # กำหนดขนาด
# axes.plot(list_idx,list_b,'g')
# axes.set_xlabel('Date / time')
# axes.set_ylabel('Amount')
# axes.set_title('Covid 19')
# axes1 = fig.add_axes([0.6,0.5,0.2,0.2])
# axes1.plot(list_b,list_idx,'r')
# axes1.set_ylabel('Date / time')
# axes1.set_xlabel('Amount')
# axes1.set_title('Covid 19')
# มองเป็นobject สร้างทั้งinsert และ subplot

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)  # row = 1 , column = 2 , figsize(กว้าง , ยาว)
axes[0].scatter(list_idx, list_b, )
axes[0].set_xlabel('Date / time')  # set x label's name
axes[0].set_ylabel('Amount')  # set y label'sname
axes[0].set_title('Covid 19 x-y (scatterplot)')  # set title
axes[0] = fig.add_axes([0.75, 0.5, 0.1, 0.2])  # สร้างplot ใน plot อีกที
axes[0].plot(list_b, list_idx, 'r')
axes[0].set_ylabel('Date / time')
axes[0].set_xlabel('Amount')
axes[0].set_title('Covid 19 plot y-x miniscreen')
axes[1].plot(list_idx, list_b, 'y')  # สร้าง multiplot
axes[1].set_ylabel('Date / time')
axes[1].set_xlabel('Amount')
axes[1].set_title('Covid 19 x-y fullscreen(normal graph)')



y = np.asarray(df_b['no'])
index = 296
y = np.delete(y, index)
# print(y)
df_b.index = pd.to_datetime(df_b.index)
df_b.index = df_b.index.map(dt.datetime.toordinal)
X = np.asarray(df_b.index)
X = np.delete(X, index)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
# print(len(X)) # Check length X
# print(len(y)) # Check length y


#Train Linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=60)

#Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
print('lm score =', lm.score(X_train, y_train))
y_pred = lm.predict(X_test)
print("Intercept =", lm.intercept_)
print("Coefficient =", lm.coef_)
print('Coefficient of determination: %.2f (The best case is 1)' % r2_score(y_test, y_pred))
print('Root Mean squared error: %.2f' % (np.sqrt(mean_squared_error(y_test, y_pred))))



#Linear Regression plot
figure1 = plt.figure()
axes5 = figure1.add_axes([0.1, 0.1, 0.8, 0.8])
axes5.scatter(X,y)
prd = lm.predict(X_test)
axes5.plot(X_test, prd, 'r')


#Poly Regression plot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=60)
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lm2 = LinearRegression()
lm2.fit(X_poly, y_train)

print("polyIntercept =", lm2.intercept_)
print("polyCoefficient =", lm2.coef_)
print('polylm score =', lm2.score(X_poly, y_train))
y_pred = lm2.predict(poly_reg.fit_transform(X))
print('polyCoefficient of determination: %.2f (The best case is 1)' % r2_score(y, y_pred))
print('polyRoot Mean squared error: %.2f' % (np.sqrt(mean_squared_error(y, y_pred))))

#Polyplot
fig2 = plt.figure()
prd1 = lm2 .predict(poly_reg.fit_transform(X))
axes3 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
axes3.plot(X, lm2.predict(poly_reg.fit_transform(X)))
axes3.set_title('Slope from PolyRegression Model')
axes3.scatter(X,y)

#Classify by Nationality
for i in range(2, 4):
    list_p.pop()
    list_idx1.pop()
list_idx2 = [w[:2] for w in list_idx1]
print("List of Fullname Country =",list_idx1)
fig3 = plt.figure()
axes4 = fig3.add_axes([0.1, 0.03, 0.8, 0.925])
axes4.barh(list_idx2, list_p)
axes4.set_title('Classify by nationality')
plt.show()
