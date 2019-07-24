import os

path = 'E:\\model\\experiments\\exp3\\psv\\results\\FMM\\128'
# path = 'E:\\model\\experiments\\exp2\\mask\\irregular_mask'

file_list = os.listdir(path)
total_file = len(file_list)

i = 1
for item in file_list:
    if item.endswith('.png'):
        src = os.path.join(os.path.abspath(path), item)
        dst = os.path.join(os.path.abspath(path), 'test_img_%04d_fake' % i + '.png')
        # dst = os.path.join(os.path.abspath(path), 'irregular_mask_%03d' % i + '.png')
        os.rename(src, dst)
        i += 1
