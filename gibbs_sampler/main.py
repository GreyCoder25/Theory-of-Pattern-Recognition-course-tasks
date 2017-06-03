import numpy as np
import random as rnd
import itertools

IMAGE_WIDTH = 5
IMAGE_HEIGHT = 5
EPSILON = 0.2
ALPHA = EPSILON / (1 - EPSILON)
NUM_COLORS = 2


def generate_image(blank_image, p):
    for i in range(blank_image.shape[0]):
        if rnd.random() <= p:
            blank_image[i] = 1

    for j in range(blank_image.shape[1]):
        if rnd.random() <= p:
            blank_image[:, j] = 1


def add_noise(image, eps):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if rnd.random() <= eps:
                image[i, j] ^= 1                                # xor for inverting pixel


def calculate_true_probabilities(image, row_post, col_post):
    for i in range(IMAGE_HEIGHT):
        for k_h_i in [0, 1]:
            column_sum = 0
            for k_v in itertools.product([0, 1], repeat=IMAGE_WIDTH):
                p1 = 1
                for j_ in range(IMAGE_WIDTH):
                    p1 *= ALPHA ** abs(image[i, j_] - (k_h_i or k_v[j_]))

                row_sum = 0
                for k_h in itertools.product([0, 1], repeat=IMAGE_HEIGHT):
                    if k_h[i] == k_h_i:
                        p2 = 1
                        for i_ in range(IMAGE_HEIGHT):
                            if i_ == i:
                                continue
                            for j_ in range(IMAGE_WIDTH):
                                p2 *= ALPHA ** abs(image[i_, j_] - (k_h[i_] or k_v[j_]))
                        row_sum += p2
                column_sum += p1 * row_sum
            row_post[k_h_i, i] = column_sum

    C = 1 / (row_post[0] + row_post[1])
    row_post *= C

    print "row posteriors:\n", row_post

    for j in range(IMAGE_WIDTH):
        for k_v_j in [0, 1]:
            row_sum = 0
            for k_h in itertools.product([0, 1], repeat=IMAGE_HEIGHT):
                p1 = 1
                for i_ in range(IMAGE_HEIGHT):
                    p1 *= ALPHA ** abs(image[i_, j] - (k_v_j or k_h[i_]))

                column_sum = 0
                for k_v in itertools.product([0, 1], repeat=IMAGE_WIDTH):
                    if k_v[j] == k_v_j:
                        p2 = 1
                        for j_ in range(IMAGE_WIDTH):
                            if j_ == j:
                                continue
                            for i_ in range(IMAGE_HEIGHT):
                                p2 *= ALPHA ** abs(image[i_, j_] - (k_h[i_] or k_v[j_]))
                        column_sum += p2
                row_sum += p1 * column_sum
            col_post[k_v_j, j] = row_sum

    C = 1 / (col_post[0] + col_post[1])
    col_post *= C

    print "column_posteriors:\n", col_post, "\n"


def gibbs_sampling(image, row_post, col_post, L):
    row_states = [0 for _ in range(IMAGE_HEIGHT)]
    column_states = [rnd.randint(0, 1) for _ in range(IMAGE_WIDTH)]

    rowscount = np.zeros(IMAGE_HEIGHT)
    colscount = np.zeros(IMAGE_WIDTH)

    stop_points = set([L / 1000, L / 100, L / 10, L / 2, L - 1])

    for t in range(L):
        for i in range(IMAGE_HEIGHT):
            for k_h_i in [0, 1]:
                p = 1
                for j_ in range(IMAGE_WIDTH):
                    p *= ALPHA ** abs(image[i, j_] - (k_h_i or column_states[j_]))
                row_post[k_h_i, i] = p


        C = 1 / (row_post[0] + row_post[1])
        row_post *= C

        for i in range(IMAGE_HEIGHT):
            row_states[i] = np.random.choice([0, 1], size=1, p=row_post[:, i])[0]
        rowscount += row_states

        for j in range(IMAGE_WIDTH):
            for k_v_j in [0, 1]:
                p = 1
                for i_ in range(IMAGE_HEIGHT):
                    p *= ALPHA ** abs(image[i_, j] - (k_v_j or row_states[i_]))
                col_post[k_v_j, j] = p

        C = 1 / (col_post[0] + col_post[1])
        col_post *= C

        for j in range(IMAGE_WIDTH):
            column_states[j] = np.random.choice([0, 1], size=1, p=col_post[:, j])[0]
        colscount += column_states

        if t in stop_points:
            p_h = rowscount / float(L)
            p_v = colscount / float(L)

            print "predicted_row_posteriors: (%d iterations)\n" % t, 1 - p_h, "\n", p_h
            print "predicted column posteriors: (%d iterations)\n" % t, 1 - p_v, "\n", p_v
            print "\n"


image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype='int32')
generate_image(image, 0.5)
print image, "\n"
add_noise(image, EPSILON)
print image, "\n"

row_posteriors = np.zeros((NUM_COLORS, IMAGE_HEIGHT))
column_posteriors = np.zeros((NUM_COLORS, IMAGE_WIDTH))

row_predicted_posteriors = np.zeros((NUM_COLORS, IMAGE_HEIGHT))
column_predicted_posteriors = np.zeros((NUM_COLORS, IMAGE_WIDTH))

calculate_true_probabilities(image, row_posteriors, column_posteriors)
gibbs_sampling(image, row_predicted_posteriors, column_predicted_posteriors, 10000)