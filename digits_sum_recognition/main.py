from PIL import Image
import numpy as np
import random as rnd


NUM_DIGITS = 10
NUM_IMAGES = 20
NUM_SUM_VALS = NUM_IMAGES * (NUM_DIGITS - 1) + 1
IMAGE_WIDTH = 27
IMAGE_HEIGHT = 30

EPSILON = 0.49
NUM_TRIALS = 100

data_path = 'digits/'


digits = np.empty((NUM_DIGITS, IMAGE_HEIGHT, IMAGE_WIDTH))

naive_dev = 0
bayes_dev = 0


def noise(image, eps):
    noisy_image = image.copy()
    for i in range(noisy_image.shape[0]):
        for j in range(noisy_image.shape[1]):
            if rnd.random() < eps:
                val = noisy_image[i][j]
                if val == 0:
                    noisy_image[i][j] = 255
                elif val == 255:
                    noisy_image[i][j] = 0
    return noisy_image


def calc_p_x_k(noisy_im, im, eps):
    prob = 1
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if noisy_im[i][j] == im[i, j]:
                prob *= (1 - eps)
            else:
                prob *= eps
    return prob


def predict_naive(posterior_arr):
    sum = 0
    for digit_probs in posterior_arr:
        sum += digit_probs.argmax()

    return sum


def predict_bayes(posterior_arr):
    F = np.zeros((NUM_SUM_VALS, NUM_IMAGES))

    for i in range(NUM_DIGITS):
        F[i][0] = posterior_arr[0][i]

    for i in range(NUM_SUM_VALS):
        for j in range(max(1, (i-1) / (NUM_DIGITS - 1)), NUM_IMAGES):
            for k in range(0, min(NUM_DIGITS, i + 1)):
                F[i][j] += posterior_arr[j][k] * F[i - k][j - 1]

    return F[:, NUM_IMAGES - 1].argmax()


for i in range(NUM_DIGITS):
    digits[i] = np.array(Image.open(data_path + str(i) + '_thumb.png'))


for _ in range(NUM_TRIALS):

    noisy_images = np.empty((NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH))
    true_sum = 0

    for i in range(NUM_IMAGES):
        rand_image_ind = rnd.randint(0, NUM_DIGITS - 1)
        true_sum += rand_image_ind
        noisy_images[i] = noise(digits[rand_image_ind], EPSILON)

        Image.fromarray(noisy_images[i]).show()                               # testing noise-function
        Image.fromarray(digits[rand_image_ind]).show()


    p_k = 1 / float(NUM_DIGITS)
    p_x_k = np.zeros((NUM_IMAGES, NUM_DIGITS))                              # conditional probabilities
    p_k_x = np.zeros((NUM_IMAGES, NUM_DIGITS))                              # posterior probabilities

    for i in range(NUM_IMAGES):
        for j in range(NUM_DIGITS):
            p_x_k[i][j] = calc_p_x_k(noisy_images[i], digits[j], EPSILON)

    for i in range(NUM_IMAGES):
        denom = p_x_k[i].sum() * p_k
        for j in range(NUM_DIGITS):
            p_k_x[i][j] = p_x_k[i][j] * p_k / denom


    # print "true sum = %d" % true_sum

    predict_bayes_sum = predict_bayes(p_k_x)
    predict_naive_sum = predict_naive(p_k_x)

    # print "The sum of the digits in the images predicted by bayesian strategy is %d" % predict_bayes_sum
    # print "The sum of the digits in the images predicted by naive strategy is %d" % predict_naive_sum

    naive_dev += abs(true_sum - predict_naive_sum)
    bayes_dev += abs(true_sum - predict_bayes_sum)


print "naive prediction average deviation: %f" % (naive_dev / float(NUM_TRIALS))
print "bayes prediction average deviation: %f" % (bayes_dev / float(NUM_TRIALS))