# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:20:34 2017

@author: a93701011
"""


import pandas as pd
import numpy as np

IDIR = "./input/"


print('loading prior..')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders..')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
"""
print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'product_name': np.str,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])
"""
print('data preparation..')

# merge priors & orders 
print('add order info to priors')
orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)
priors = priors[["order_id","product_id","reordered","user_id","order_number"]]

#count(order) group by user 
print("computing order info")
user = pd.DataFrame()
user['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
user["nb_orders"] = orders.groupby(orders.user_id).size().astype(np.int32)

users = pd.DataFrame()
users['total_items'] = priors.groupby(priors.user_id).size().astype(np.int16)
users = users.join(user)
del user
priors = priors.join(users, on = "user_id")

print('computing product f')
prods = pd.DataFrame()
prods['nb_product'] = priors.groupby(["user_id","product_id"]).size().astype(np.int32)
priors = priors.join(prods, on = ["user_id","product_id"], how ="left")
del prods

priors["product_order_rate"] = (priors.nb_product / priors.nb_orders *100).astype(np.int8)
priors.drop("nb_product", inplace=True, axis=1)

# where eval_set = 'test' and eval_set = 'train' 
print("split user train/test")
userset = orders[["user_id","eval_set"]].drop_duplicates()
testuser = userset[userset["eval_set"] == "test"]
trainuser = userset[userset["eval_set"] == "train"]
#where order eval_set = 'test'
predorder = orders[orders["eval_set"] == "test"]
predorder = predorder[["order_id","user_id"]]

print("feature and clearn..")
# get train feature
priortest = pd.merge(priors, testuser, on="user_id", how ="inner")
priortest.drop('eval_set', inplace=True, axis=1)

#fill one coulumn 
priortest["days_since_prior_order"].fillna(0,inplace=True) 
priortest["days_since_prior_order"] = priortest["days_since_prior_order"].astype(np.int8)
priortest["product_order_rate"].fillna(0,inplace=True) 
priortest["product_order_rate"] = priortest["product_order_rate"].astype(np.int)

print("splite data..")
#from sklearn.cross_validation import train_test_split

priortest["order_number_desc"] = (priortest.nb_orders -priortest.order_number).astype(np.int8)


X_train = priortest[["product_id","total_items","days_since_prior_order","product_order_rate"]].values
y_train = priortest["reordered"].values
datatest = priortest[priortest["order_number_desc"]<3]
X_test = datatest[["product_id","total_items","days_since_prior_order","product_order_rate"]].values
y_test = datatest["reordered"].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print("build model..")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)

#print("Accuracy {}".format(knn.score(X,y)))



print("predict reorder item..")
y_pred = knn.predict(X_test)
X_test["user_id"] = datatest.user_id
data = X_test[y_pred == 1]

print("result..")
feature = ["user_id","product_id","days_since_prior_order","product_order_rate"]
userresult = pd.DataFrame(data, columns = feature)
userresult = userresult[["user_id","product_id"]]

orderresult = pd.merge(userresult, predorder, how='inner',on='user_id')

print("transform format..")
d = dict()
for row in orderresult.itertuples():
    try:
        d[row.order_id] += ' ' + str(row.product_id)
    except:
        d[row.order_id] = str(row.product_id)
        
for order in predorder.order_id:
    if order not in d:
        d[order] = 'None'
        
sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv("D:\\Learning Note\\Python\\kaggle\\instacart\\result.csv", index=False)

#userresult.to_csv("D:\\Learning Note\\Python\\kaggle\\instacart\\result.csv", index=False)
       
