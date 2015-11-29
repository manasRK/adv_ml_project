import json
import sqlite3
import time

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


def read_data(full_file_name):
    raw_file = open(full_file_name, 'r')
    data = json.load(raw_file)
    return data


def insert_results(method, acc, recall, prec, f1):
    time_stamp = time.time()
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    c.execute("INSERT INTO results VALUES (time_stamp, subset + _ + method, acc, recall, prec, f1)")
    conn.commit()
    conn.close()