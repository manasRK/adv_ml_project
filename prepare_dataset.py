import os
import gzip
import json
from collections import defaultdict

dataset_type = 'amazon'
file_path = '/home/raihan/Data/Amazon Data'

subset = 'cell_phones'
if subset == 'video_games':
    file_name = 'Video_Games.txt.gz'
if subset == 'instant_videos':
    file_name = 'Amazon_Instant_Video.txt.gz'
if subset == 'electronics':
    file_name = 'Electronics.txt.gz'
if subset == 'finefoods':
    file_name = 'finefoods.txt.gz'
if subset == 'books':
    file_name = 'Books.txt.gz'
if subset == 'home_kitchen':
    file_name = 'Home_&_Kitchen.txt.gz'
if subset == 'sports_outdoos':
    file_name = 'Sports_&_Outdoors.txt.gz'
if subset == 'health':
    file_name = 'Health.txt.gz'
if subset == 'arts':
    file_name = 'Arts.txt.gz'
if subset == 'automotive':
    file_name = 'Automotive.txt.gz'
if subset == 'beauty':
    file_name = 'Beauty.txt.gz'
if subset == 'cell_phones_accessories':
    file_name = 'Cell_Phones_&_Accessories.txt.gz'
if subset == 'clothing_accessories':
    file_name = 'Clothing_&_Accessories.txt.gz'
if subset == 'gourmet_foods':
    file_name = 'Gourmet_Foods.txt.gz'
if subset == 'industrial_scientific':
    file_name = 'Industrial_&_Scientific.txt.gz'
if subset == 'jewelry':
    file_name = 'Jewelry.txt.gz'
if subset == 'kindle_store':
    file_name = 'Kindle_Store.txt.gz'
if subset == 'musical_instruments':
    file_name = 'Musical_Instruments.txt.gz'
if subset == 'office_products':
    file_name = 'Office_Products.txt.gz'
if subset == 'patio':
    file_name = 'Patio.txt.gz'
if subset == 'pet_supplies':
    file_name = 'Pet_Supplies.txt.gz'
if subset == 'shoes':
    file_name = 'Shoes.txt.gz'
if subset == 'software':
    file_name = 'Software.txt.gz'
if subset == 'tools_home_improvement':
    file_name = 'Tools_&_Home_Improvement.txt.gz'
if subset == 'toys_games':
    file_name = 'Toys_&_Games.txt.gz'
if subset == 'watches':
    file_name = 'Watches.txt.gz'
if subset == 'music':
    file_name = 'Music.txt.gz'
if subset == 'movies_tv':
    file_name = 'Movies_&_TV.txt.gz'
if subset == 'cell_phones':
    file_name = 'Cell_Phones_&_Accessories.txt.gz'


pruned_f_name = file_name.split('.')[0] + '.json'
data_path = 'data'

def parse(filename):
    print filename
    f = gzip.open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos+2:]
        entry[eName] = rest
    yield entry


def create_dataset_amazon():
    '''i = 0
    review_per_user = {}
    for e in parse(os.path.join(file_path, file_name)):
        try:
            current_user_id = e['review/userId']
            current_product_id = e['product/productId']
            current_rating = e['review/score']
            current_review = e['review/text']
            if current_user_id != 'unknown' and current_user_id in review_per_user:
                review_per_user[current_user_id] += 1
            else:
                review_per_user[current_user_id] = 1
            i += 1
            if i % 1000 == 0:
                print i
        except:
            continue
    i = 0'''
    all_review_per_user = []
    i = 0
    out_file = open(os.path.join(data_path, pruned_f_name), 'w+')
    for e in parse(os.path.join(file_path, file_name)):
        i += 1
        if i % 1000 == 0:
            print i
        try:
            current_user_id = e['review/userId']
            current_product_id = e['product/productId']
            current_rating = e['review/score']
            current_review = e['review/text']
            if float(current_rating) >= 4 or float(current_rating) <= 2:
                if float(current_rating) >= 4:
                    rating = 1
                else:
                    rating = 0
                all_review_per_user.append({'uid':current_user_id, 'pid': current_product_id,
                                            'rat':str(rating), 'rev': current_review})
        except:
            continue
    json.dump(all_review_per_user, out_file)
    out_file.close()

if __name__ == '__main__':
    create_dataset_amazon()