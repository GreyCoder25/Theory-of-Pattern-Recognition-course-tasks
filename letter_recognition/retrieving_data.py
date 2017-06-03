from PIL import Image


NUM_LETTERS_IN_ROW = 10
NUM_LETTERS_IN_COLUMN = 3
NUM_LETTERS_AT_ALL = 26


def binarize(image):
    return image.point(lambda i: 0 if i < 150 else 255)


data_dir = 'raw_data'
letters2 = binarize(Image.open(data_dir + '/letters2.png').convert('L'))
letters1 = binarize(Image.open(data_dir + '/letters1.png').convert('L'))

width, height = letters1.size

box = [0, 0, width // NUM_LETTERS_IN_ROW, height // NUM_LETTERS_IN_COLUMN]
x_step = width // NUM_LETTERS_IN_ROW
y_step = height // NUM_LETTERS_IN_COLUMN

i = 1
code = ord('A')
retr_data_dir = 'data'
while i <= NUM_LETTERS_AT_ALL:
    letters1.crop(box).save(retr_data_dir + '/' + chr(code) + '1' + '.png')
    letters2.crop(box).save(retr_data_dir + '/' + chr(code) + '2' + '.png')
    box[0] += x_step
    box[2] += x_step
    if i % 10 == 0:
        box[0] = 0
        box[2] = x_step
        box[1] += y_step
        box[3] += y_step
    i += 1
    code += 1


if __name__ == '__main__':
    pass