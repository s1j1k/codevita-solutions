import numpy as np
from math import pi

from my_linalg import vector_proj_plane

# TODO plot the output at each stage to make the beetle's journey clear

d = 3  # 3 dimensions

def opposite_distance(w, n_ft, n_a_b, A, B):
    u = abs(A - B)

    # only 10 if the n_ft is 10 - the distance to the face. w is the weight of the face.
    d_a_b_ft = abs(2*w - A - B)*n_ft

    # combined distance across the (intermediate) face of travel (10cm) and the distance to get to the face of travel
    x1 = d_a_b_ft + 10*n_ft

    # now there is only one dimension remaining to get the distance, and then use pythagoras
    y1 = u - u * n_ft - u * n_a_b

    print(f'x1={x1}, y1={y1}')
    x1 = np.sum(x1)
    y1 = np.sum(y1)

    return np.sum(x1*x1 + y1*y1)**0.5


def distance(A, B):
    print(f'A={A},B={B}')

    dist = 0

    i_face_a = np.asscalar(np.where((A == 10) | (A == 0))[0])  # dimension
    i_face_b = np.asscalar(np.where((B == 10) | (B == 0))[0])  # dimension

    face_a = (i_face_a, A[i_face_a])
    face_b = (i_face_b, B[i_face_b])

    print(f'face_a={face_a},face_b={face_b}')
    # find a vector orthogonal to the face
    n_a = np.zeros((1, d)).T
    n_b = np.zeros((1, d)).T

    n_a[i_face_a] = (A[i_face_a] + 1) % 10
    n_b[i_face_b] = (B[i_face_b] + 1) % 10

    n_a = n_a.T
    n_b = n_b.T

    print(f'orthogonal vectors n_a={n_a}, n_b={n_b}')

    # CASE 1: SAME FACE
    if face_a == face_b:  # same face
        print('on the same face')
        ed = np.sum((A - B) * (A - B)) ** 0.5  # euclidean distance
        dist = round(2.0 * pi * ed / 6, 2)  # round to two decimal place

    else:
        print('not on same face')


        # CASE 2: OPPOSITE FACE
        if i_face_a == i_face_b:
            print('opposite sides')
            # compare different ways of collapsing cube depending on distance

            # 3 possible ways of adjacent travel on 3 faces - unfolding
            n_a_b = n_a  # same normal vector

            # only check the three normals that are NOT (the current face, the opposite face (where we need to go), the bottom)
            # check the top, the adjacent side to the left, the adjacent side to the right

            # CASE 2.1 TOP
            n_ft = np.array([[0, 0, 1]]) # get a normal vector for the face of travel, here it is the top face z = 10

            dist0 = opposite_distance(10, n_ft, n_a_b, A, B)
            print(f'top n_ft={n_ft}, dist={dist0}')

            # CASE 2.2 ADJACENT LEFT
            n_ft = np.ones((1,d)) - n_ft - n_a_b

            dist1 = opposite_distance(0, n_ft, n_a_b, A, B)
            print(f'adjacent 1 n_ft={n_ft}, w={0} dist={dist1}')

            # CASE 2.3 ADJACENT RIGHT

            dist2 = opposite_distance(10, n_ft, n_a_b, A, B)
            print(f'adjacent 2 n_ft={n_ft}, w={10}, dist={dist2}')

            print(f'dist0={dist0}, dist1={dist1}, dist2={dist2}')

            dist = min(dist0, dist1, dist2)

        # CASE 3: ADJACENT FACE
        else:
            print('adjacent')
            # simply collapse the cube


            # get a vector between the two points
            u = np.array([abs(A - B)])
            print(f'u={u}')


            # get a vector between the two points
            u = np.array([abs(A - B)])
            print(f'u={u}')

            # get the distance that can only be travelled across the faces individually, summed overall
            x1 = u * n_b + u * n_a

            # remove those elements which are on the normal (constant distance)
            y1 = u - x1

            x1 = np.sum(x1)
            y1 = np.sum(y1)

            print(f'x1={x1}, y1={y1}')

            dist = round((x1 * x1 + y1 * y1) ** 0.5, 2)

    print('dist between {} and {} {:.2f}'.format(A, B, dist))
    return dist


if __name__ == '__main__':
    # USER INPUT (commented out for testing)
    #n = int(input()) # N in [2,10]
    #pts = np.array(input().split(',')).reshape((n, d)).astype('int32')
    # each coordinate in pts in [0,10]

    #n = 3
    # example 1
    #pts = np.array(['1','1','10','2','1','10','0','1','9']).reshape((n, d)).astype('int32')
    # example 2
    # pts = np.array(['1', '1', '10', '2', '1', '10', '0', '5', '9']).reshape((n, d)).astype('int32')

    n = 2
    pts = [0, 1, 5, 10, 3, 8]
    pts = np.array(pts).reshape(n,d)

    print(f'pts: {pts}')  # n x d array

    total_distance = 0
    for i in range(n-1):
        total_distance += distance(pts[i],pts[i+1])

    print(f'total_distance={total_distance}')