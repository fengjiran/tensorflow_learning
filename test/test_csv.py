import csv

with open('log.csv', 'a+') as f:
    mywrite = csv.writer(f)
    mywrite.writerow(['dis_loss',
                      'gen_gan_loss',
                      'gen_l1_loss',
                      'gen_style_loss',
                      'gen_content_loss'])

for i in range(10):
    with open('log.csv', 'a+') as f:
        mywrite = csv.writer(f)
        mywrite.writerow([1, 2, 3, 4, 5])


# with open('log', 'a+') as f:
#     print >> f, 'The first time to pretrain!'
