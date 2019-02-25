import csv

with open('test.csv', 'a+') as f:
    mywrite = csv.writer(f)
    mywrite.writerow(['dis_loss',
                      'gen_gan_loss',
                      'gen_l1_loss',
                      'gen_style_loss',
                      'gen_content_loss'])
