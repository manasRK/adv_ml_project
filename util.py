import json
import sqlite3
import datetime

data_path = 'data'
subset = 'arts'
if subset == 'arts':
    file_name = 'Arts.json'
if subset == 'video_games':
    file_name = 'Video_Games.json'
if subset == 'instant_videos':
    file_name = 'Amazon_Instant_Video.json'
if subset == 'electronics':
    file_name = 'Electronics.json'
if subset == 'patio':
    file_name = 'Patio.json'
if subset == 'watches':
    file_name = 'Watches.json'
if subset == 'shoes':
    file_name = 'Shoes.json'
if subset == 'tg':
    file_name = 'Tools_&_Home_Improvement.json'
if subset == 'toy_games':
    file_name = 'Toys_&_Games.json'
if subset == 'ind_sci':
    file_name = 'Industrial_&_Scientific.json'
if subset == 'so':
    file_name = 'Sports_&_Outdoors.json'
if subset == 'office_products':
    file_name = 'Office_Products.json'
if subset == 'kindle_store':
    file_name = 'Kindle_Store.json'
if subset == 'software':
    file_name = 'Software.json'
if subset == 'cell_phones':
    file_name = 'Cell_Phones_&_Accessories.json'


def read_data(full_file_name):
    raw_file = open(full_file_name, 'r')
    data = json.load(raw_file)
    return data


def insert_results(method, acc, recall, prec, f1):
    time_stamp = datetime.datetime.now()
    name = method + '_' + subset
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    c.execute('''INSERT INTO results (insert_date, dataset, acc, recall, prec, f1) VALUES (?,?,?,?,?,?)''',
              (str(time_stamp), name, acc, recall, prec, f1))
    conn.commit()
    conn.close()