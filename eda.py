# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:40:27 2017

@author: a93701011
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
IDIR = "./input/"


print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'product_name': np.str,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])

print('loading orders..')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))
weekcount=orders.groupby('order_dow').size().reset_index()
weekcount.columns = ['week','total']

sns.set_color_codes("pastel")
sns.barplot(x="week", y="total", data=weekcount,
            label="Total", color="b")
sns.set_color_codes("muted")
plt.show()

