# packages
import numpy as np
#used for dataframes (kinda like excell spreadsheets)
import pandas as pd

from sklearn import svm

#visualizing data
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale = 1.2)

#create dataframe, similar to R
recipes = pd.read_csv('Cupcakes vs Muffins.csv')
#print(recipes.head())

#format or pre-process data
type_label = np.where(recipes['Type']=='Muffin',0, 1)
recipe_features = recipes.columns.values[1:].tolist()
ingredients = recipes[['Flour', 'Sugar']].values

#fit model
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)

#w is two different coefficients
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30,60)
yy = a * xx - (model.intercept_[0]) / w[1] 

#plot the parallels seperating the hyperplane that pass through support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[1]
yy_up = a * xx + (b[1] - a * b[0])

#plot data with seaborn
sns.lmplot(x='Flour', y='Sugar', data=recipes, hue='Type', 
	palette='Set3', fit_reg=False, scatter_kws={"s": 70});
plt.plot(xx, yy, linewidth=2, color='orange')
plt.show()

def predict(flour, sugar):
	if(model.predict([[flour, sugar]]))==0:
		print("muffin recipe")
	else:
		print("cupcake recipe")

print(predict(20,20))