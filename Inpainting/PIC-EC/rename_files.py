import os

path = 'E:\\model\\experiments\\exp2\\celebahq\\gt_images\\celebahq'

file_list = os.listdir(path)
total_file = len(file_list)

i = 1
for item in file_list:
    if item.endswith('.png'):
        src = os.path.join(os.path.abspath(path), item)
        dst = os.path.join(os.path.abspath(path), 'gt_img_1024_%03d' % i + '.png')
        os.rename(src, dst)
        i += 1
