# This .py file is used to generate the .info file required for training the cascade classifier

# cnt_pos & cnt_neg is the number of the image that goes in the path
cnt_pos = 1
cnt_neg = 1

# total_pos & total_neg is the total number of images in the positive folder
total_pos = 23709
total_neg = 7200

# creating positive .info file
with open('info.dat', 'w') as f:
    while cnt_pos <= total_pos:
        
        path = 'positive/pos-img ({}).jpg'.format(cnt_pos) 
        info = ' 1 0 0 200 200'

        text = path + info

        f.write(text)
        f.write('\n')
        
        cnt_pos += 1
        
# creating negative .info file
with open('bg.txt', 'w') as f:
    while cnt_neg <= total_neg:
    
        path = 'negative/neg-img({}).png'.format(cnt_neg) 
        
        text = path

        f.write(text)
        f.write('\n')
        
        cnt_neg += 1