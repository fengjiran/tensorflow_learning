import csv

with open('log.csv', 'a+') as f:
    mywrite = csv.writer(f)
    mywrite.writerow(['dis_loss',
                      'gen_gan_loss',
                      'gen_l1_loss',
                      'gen_style_loss',
                      'gen_content_loss'])
