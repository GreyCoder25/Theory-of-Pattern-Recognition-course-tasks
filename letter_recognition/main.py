from PIL import Image
import numpy as np
import random as rnd
from time_measure_tools import *

NUM_LETTERS = 2
NUM_MODELS = 2
# NUM_IMAGES = 1

IMAGE_WIDTH = 23
IMAGE_HEIGHT = 27

EPSILON = 0.3
NUM_TRIALS = 1
NUM_TRAINING_IMAGES = 3000
NUM_TEST_IMAGES = 3000

letters = np.empty((NUM_MODELS, NUM_LETTERS, IMAGE_HEIGHT, IMAGE_WIDTH))

training_images = np.empty((NUM_MODELS, NUM_TRAINING_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH))
training_answers = np.empty((NUM_MODELS, NUM_TRAINING_IMAGES), dtype='S1')


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


def generate_noisy_image(letters, theta):
    rand_image_coords = theta, rnd.randint(0, NUM_LETTERS - 1)
    # Image.fromarray(noisy_image).show()  # testing noise-function
    # Image.fromarray(letters[rand_image_coords]).show()
    true_letter = chr(ord('A') + rand_image_coords[1])

    return noise(letters[rand_image_coords], EPSILON), true_letter


def calc_p_x_k_theta(noisy_im, im, eps):
    prob = 1
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if noisy_im[i][j] == im[i][j]:
                prob *= (1 - eps)
            else:
                prob *= eps
    return prob


def calc_cond_arr(x):
    arr = np.zeros((NUM_MODELS, NUM_LETTERS))
    for i in range(NUM_MODELS):
        for j in range(NUM_LETTERS):
            arr[i][j] = calc_p_x_k_theta(x, letters[i][j], EPSILON)
    return arr


def q_naive(x):
    conditional_arr = calc_cond_arr(x)
    max_pos = conditional_arr.argmax()
    return chr(ord('A') + max_pos % NUM_LETTERS)

@measure_time
def R(q, theta, gamma0, gamma1):
    num_unrec_im = 0
    for i in range(NUM_TRAINING_IMAGES):
        x, true_k = training_images[theta][i], training_answers[theta][i]
        num_unrec_im += (q(x, gamma0, gamma1) != true_k)

    return float(num_unrec_im) / NUM_TRAINING_IMAGES


def q(x, gamma0, gamma1):
    conditional_arr = 10e100 * calc_cond_arr(x)
    max_k = 0
    for k in range(1, NUM_LETTERS):
        if (gamma0*conditional_arr[0][k] + gamma1*conditional_arr[1][k] >
            gamma0*conditional_arr[0][max_k] + gamma1*conditional_arr[1][max_k]
        ):
            max_k = k

    return chr(ord('A') + max_k)


data_dir = 'data/'
for i in range(NUM_MODELS):
    code = ord('A')
    for j in range(NUM_LETTERS):
        letters[i][j] = np.array(Image.open(data_dir + chr(code) + str(i + 1) + '.png'))
        code += 1


for i in range(NUM_TRAINING_IMAGES):
    training_images[0][i], training_answers[0][i] = generate_noisy_image(letters, 0)
    training_images[1][i], training_answers[1][i] = generate_noisy_image(letters, 1)


# if R(q, theta=0, gamma0=1, gamma1=0) > R(q, theta=1, gamma0=0, gamma1=1):
#     theta_start, theta_other, g0, g1 = 0, 1, 1, 0
# else:
#     theta_start, theta_other, g0, g1 = 1, 0, 0, 1
#
# step = 0.01
# alpha = 0.0025                                # threshold
#
# g0_start = g0
# Rs, Ro = R(q, theta_start, g0, g1), R(q, theta_other, g0, g1)
# if Ro > Rs:
#     while True:
#         # assert(g0 >= 0 and g0 <= 1 and g1 >= 0 and g1 <= 1, "g0 = %f, g1 = %f" % (g0, g1))
#         print 'R(theta_start) = %f, R(theta_other) = %f' % (Rs, Ro)
#         print 'g0 = %f, g1 = %f' % (g0, g1)
#         if g0_start == 1:
#             g0 -= step
#             g1 += step
#         elif g0_start == 0:
#             g0 += step
#             g1 -= step
#         print 'g0 = %f, g1 = %f' % (g0, g1)
#         Rs_next, Ro_next = R(q, theta_start, g0, g1), R(q, theta_other, g0, g1)
#         if abs(Rs_next - Ro_next) < alpha:
#             print '%.2f - %.2f < %.5f' % (Rs_next, Ro_next, alpha)
#             break
#         if np.sign(Rs_next - Ro_next) != np.sign(Rs - Ro) or abs(abs(g0 - g0_start) - 1) < 0.00000001:
#             print 'risks exchange'
#             if g0_start == 1:
#                 g0 += step
#                 g1 -= step
#             elif g0_start == 0:
#                 g0 -= step
#                 g1 += step
#             step /= 2
#             print 'step /= 2'
#         Rs, Ro = Rs_next, Ro_next
#
# print 'Finally, R(theta_start) = %f, R(theta_other) = %f' % (Rs_next, Ro_next)
#
# print "Bayesian strategy optimal parameters:\ngamma0 = %f\ngamma1 = %f\n" % (g0, g1)
#
# test_images = np.empty((NUM_TEST_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH))
# test_answers = np.empty((NUM_TEST_IMAGES), dtype='S1')
#
# for i in range(NUM_TEST_IMAGES):
#     test_images[i], test_answers[i] = generate_noisy_image(letters, rnd.randint(0, NUM_MODELS - 1))
#
# unrec_naive = 0
# unrec_bayes = 0
# for i in range(NUM_TEST_IMAGES):
#     unrec_naive += (q_naive(test_images[i]) != test_answers[i])
#     unrec_bayes += (q(test_images[i], g0, g1) != test_answers[i])
#     print '%d of %d images tested' % (i, NUM_TEST_IMAGES)
#
#
# print "Naive strategy recognises letters with risk %f" % (float(unrec_naive) / NUM_TEST_IMAGES)
# print "Bayes strategy recognises letters with risk %f" % (float(unrec_bayes) / NUM_TEST_IMAGES)
# noisy_image, true_letter = generate_noisy_image(letters, rnd.randint(0, NUM_MODELS - 1))
#
# print "True letter is '%s'" % true_letter
#
# naive_prediction = q_naive(noisy_image)
# my_prediction1 = q(noisy_image, gamma0=0.9, gamma1=0.1)
# my_prediction2 = q(noisy_image, gamma0=0.1, gamma1=0.9)
#
#
# print "Letter predicted by naive strategy is '%s'" % naive_prediction
# print "Letter predicted by my strategy 1 is '%s'" % my_prediction1
# print "Letter predicted by my strategy 2 is '%s'" % my_prediction2

print 'Risk1 is %f' % R(q, theta=0, gamma0=1, gamma1=0)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.95, gamma1=0.05)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.9, gamma1=0.1)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.85, gamma1=0.15)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.8, gamma1=0.2)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.75, gamma1=0.25)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.7, gamma1=0.3)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.65, gamma1=0.35)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.6, gamma1=0.4)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.55, gamma1=0.45)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.5, gamma1=0.5)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.45, gamma1=0.55)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.4, gamma1=0.6)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.35, gamma1=0.65)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.3, gamma1=0.7)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.25, gamma1=0.75)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.2, gamma1=0.8)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.15, gamma1=0.85)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.1, gamma1=0.9)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0.05, gamma1=0.95)
print 'Risk1 is %f' % R(q, theta=0, gamma0=0, gamma1=1)



