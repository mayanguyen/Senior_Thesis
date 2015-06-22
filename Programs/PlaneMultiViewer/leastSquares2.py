import numpy as np
import scipy as sp
from scipy.sparse.linalg import lsqr
from sympy import *
import bigfloat as bf
import mpmath as mp
import cv2

# numpy.linalg.lstsq https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=numpy%20lstsq


# Least Squares Chessboard
def LSChessboard(p, nrows, ncols):
    if (p.shape[0] != nrows*ncols or p.shape[1] != 2):
        print('Incorrect dimensions.')
        return -1
    
    # total number of points
    n = nrows * ncols

    sum1, sum2, sum3, b1, b2, b3 = [0, 0, 0, 0, 0, 0]
    for i in range(n):
        sum1 += i%ncols
        sum2 += i/ncols
        sum3 += (i%ncols)*(i%ncols) + (i/ncols)*(i/ncols)
        b1 += p[i][0]
        b2 += p[i][1]
        b3 += (i%ncols)*p[i][0] + (i/ncols)*p[i][1]
    
    A = np.array([
        [n, 0, sum1],
        [0, n, sum2],
        [sum1, sum2, sum3]], np.float64)

    b = np.array([[b1], [b2], [b3]], np.float64)

    AInv = np.linalg.inv(A)
    x, y, u = AInv.dot(b)

    q = np.zeros((nrows*ncols, 2), np.float64)
    for i in range(n):
        q[i] = [x+(i%ncols)*u, y+(i/ncols)*u]

    return np.around(q)



# Homogenize the given list of 2D points
# param: p = list of points, each with 2 coordinates
# return: new list with homogeneous points created by appending 1 as the homoge.
#   coordinate to each point
def homogenize(p):
    n = p.shape[0]
    ones = np.ones((n,1))
    p = np.hstack([p, ones])
    
    return p

def unhomogenize(p):
    #print("last column = \n"+str(p[:,2]))
    p[:,0] = p[:,0]/p[:, 2]
    p[:,1] = p[:,1]/p[:, 2]
    p[:,2] = p[:,2]/p[:, 2]

    return p


# Least Squares Optimal Homography for 2 cameras
# param: A = H1
# param: B = H2
# param: p = points on orig image
# param: v = q1 = target points on cam1
# param: w = q2 = target points on cam2
# Return: H_hat (the optimal homography)
def HhatInv(A, B, p, v, w):
    if (p.shape != v.shape or p.shape != w.shape):
        print('p.shape = '+str(p.shape))
        print('v.shape = '+str(v.shape))
        print('w.shape = '+str(w.shape))
        print('The number of points is not matching.')
        return -1
    if (A.shape != (3,3) or B.shape != (3,3)):
        print('A.shape = '+str(A.shape))
        print('B.shape = '+str(B.shape))
        print('Matrix dimensions not matching.')
        return -1
    
    if (p.shape[1] < 3):
        p, v, w = homogenize(p), homogenize(v), homogenize(w)
        print('new p = '+str(p))
        print('new v = \n'+str(v))
        print('new w = \n'+str(w))

    #v, w = unhomogenize(v), unhomogenize(w)

    
    M, b = partialDerivs(A, B, p, v, w)
    print('M: h coefficients')
    for i in range(8 + 2*p.shape[0]):
        print(str(M[i,0:4])+ str(M[i,4:8]) + '\n' )
    print('lambda coefficients')
    for i in range(8 + 2*p.shape[0]):
        print(str(M[i,8:(8+p.shape[0])]) +'\n')
    print('mu coefficients')
    for i in range(8 + 2*p.shape[0]):
        print(str(M[i,(8+p.shape[0]):]) + '\n')


    '''for i in range(8 + 2*p.shape[0]):
        print(str(M[i,0:8]) + str(M[i,8:(8+p.shape[0])]) + str(M[i,(8+p.shape[0]):]) + '\n' )
    '''

    print('b = \n'+str(b))

    
    H_hatInv, res, rank, singular = np.linalg.lstsq(M, b)
    H_hatInv = H_hatInv[:8]
    H_hatInv = np.vstack([H_hatInv, [1]])
    #H_hatInv = H_hatInv.flatten().reshape((-1, 3))

    '''MInv = np.linalg.inv(M)
    print('solution: \n'+str(MInv.dot(b)))
    H_hatInv = MInv.dot(b)[:8]
    H_hatInv = np.vstack([H_hatInv, [1]])
    H_hatInv = H_hatInv.flatten().reshape((-1, 3))'''

    '''All = np.hstack((M,b))
    print('All'+str(All))
    All2 = Matrix(All)
    print('all2'+str(All2))
    All3 = All2.rref()
    print(str(All3))

    H_hatInv = np.ones((9,1), np.float64)
    for i in range(8):
        print(str( All3[0][ (8 + 2*p.shape[0]) + (i * (9 + 2*p.shape[0])) ] ))
        H_hatInv[i][0] = All3[0][(8 + 2*p.shape[0]) + (i * (9 + 2*p.shape[0])) ] #20+i*21]

    
    '''
    
    '''M = M/10000
    b = b/10000
    print(str(M))
    print(str(b))'''

    
    #solution, a2, a3, a4, a5, a6, a7, a8, a9, a10 = sp.sparse.linalg.lsqr(M,b)
    #print("All solutions: "+str(solution.shape)+'condition = '+str(a7)+"\n"+ str(solution))

    #H_hatInv = np.linalg.solve(M,b)
    
    '''print(np.allclose(np.dot(M, H_hatInv), b))
    H = np.array([
        [  1.0, 2.0, 5.0],
        [  2.0, 3.0, -1.0],
        [  1.0, 0.0, 1.0]]).flatten().transpose()[:8]
    print(np.allclose(np.dot(M[:8,:8], H), b[:8]))

    H_hatInv = H_hatInv[:8]
    print(np.allclose(np.dot(M[:8,:8], H_hatInv), b[:8]))'''

    '''H_hatInv = solution[:8]
    H_hatInv = np.vstack([H_hatInv, [1]])'''

    H_hatInv = H_hatInv.flatten().reshape((-1, 3))
    
    return H_hatInv


'''pivoting
singular value decomposition

A = BDC
A-1 = C-1 D-1 B-1

QR decomposition'''


# Return: list of coefficients of h_i and the const. term
# of the partial derivatives of e
def partialDerivs(A, B, p, v, w):
    size = p.shape[0]  # number of points correspondences
    M = np.zeros((2*size+8, 2*size+8), np.float64)
    b = np.zeros((2*size+8, 1), np.float64)
    
    # derivative with respect to h_mn
    for m in range(3):
        for n in range(3):
            if (m+n != 4):
                # Coefficient of h_ij
                for i in range(3):
                    for j in range(3):
                        if (i+j != 4):
                            # Coefficient of h_ij
                            h_ij = 0
                            for k in range(size):
                                for l in range(3):
                                    h_ij += p[k][j]*p[k][n] * (A[l][i]*A[l][m] + B[l][i]*B[l][m])
                            M[m*3+n][i*3+j] = h_ij
                # Coefficient of lamb_k and mu_k
                for k in range(size):
                    # Coefficient of lamb_k and mu_k
                    lamb = 0
                    mu = 0
                    for l in range(3):
                        lamb -= v[k][l]*A[l][m]*p[k][n]
                        mu -= w[k][l]*B[l][m]*p[k][n]
                    M[m*3+n][8+k] = lamb
                    M[m*3+n][size+8+k] = mu
                # Coefficient of h_2,2
                h_22 = 0
                for k in range(size):
                    for l in range(3):
                        h_22 -= p[k][2]*p[k][n] * (A[l][2]*A[l][m] + B[l][2]*B[l][m])
                b[m*3+n] = h_22
    
    # derivative with respect to lambda_m
    for m in range(size):
        # Coefficient of h_ij
        for i in range(3):
            for j in range(3):
                if (i+j != 4):
                    # Coefficient of h_ij
                    h_ij = 0
                    for l in range(3):
                        h_ij += p[m][j] * A[l][i] * v[m][l]
                    M[8+m][i*3+j] = h_ij
        # Coefficient of lamb_m
        lamb_m = 0
        for l in range(3):
            lamb_m -= v[m][l]*v[m][l]
        M[8+m][8+m] = lamb_m
        # Coefficient of h_2,2
        h_22 = 0
        for l in range(3):
            h_22 -= p[k][2] * A[l][2] * v[m][l]
        b[8+m] = h_22

    # derivative with respect to mu_m
    for m in range(size):
        # Coefficient of h_ij
        for i in range(3):
            for j in range(3):
                if (i+j != 4):
                    # Coefficient of h_ij
                    h_ij = 0
                    for l in range(3):
                        h_ij += p[m][j] * B[l][i] * w[m][l]
                    M[8+size+m][i*3+j] = h_ij
        # Coefficient of mu_m
        mu_m = 0
        for l in range(3):
            mu_m -= w[m][l]*w[m][l]
        M[8+size+m][8+size+m] = mu_m
        # Coefficient of h_2,2
        h_22 = 0
        for l in range(3):
            h_22 -= p[k][2] * B[l][2] * w[m][l]
        b[8+size+m] = h_22

    return M, b


def HhatInvTest2():
    '''H1 = np.array([
        [ 1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 1.0]], np.float64)
    H2 = np.array([
        [ 1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 1.0]], np.float64)'''

    '''H1 = np.array([
        [  0.2138,  -0.018484,   67.34774],
        [ -0.21618,   0.2113,   173.827],
        [ -8.029*0.0001,  -1.825*0.00001,   1]], np.float64)
    H2 = np.array([
        [  0.3186,  -1.2517*0.001,   99.27943],
        [ -0.17187,   0.260943843,   169.74],
        [ -5.552*0.0001,  -3.21849576*0.00001,   1]], np.float64)'''

    H1 = np.array([
        [  0.2138,  -0.018484,   67.34774],
        [ -0.21618,   0.2113,   173.827],
        [ -0.0001,  0.0,   1.0]], BigFloat)# np.float64)
    H2 = np.array([
        [  0.3186,  -0.0024,   99.27943],
        [ -0.17187,   0.260943843,   169.74],
        [ -0.0005,  -0.00003,   1.0]], BigFloat)# np.float64)

    '''H1 = np.array([
        [ 3.0, 4.0, 9.0],
        [ 4.0, 1.0, 2.5],
        [ 6.0, 2.0, 1]], np.float64)
    H2 = np.array([
        [ 4.0, 0.5, 7.0],
        [ 0.1, 1.0, 0.0],
        [ 2.0, 4.0, 1.0]], np.float64)'''

    p = np.array([
        [100.0, 100.0],
        [200.0, 100.0],
        [300.0, 100.0],
        [100.0, 200.0],
        [200.0, 200.0],
        [300.0, 200.0]], BigFloat)# np.float64)
    
    '''H = np.array([
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]], np.float64)'''
    H = np.array([
        [  1.0, 2.0, 5.0],
        [  2.0, 3.0, -1.0],
        [  1.0, 0.0, 1.0]], BigFloat)# np.float64)

    p = homogenize(p)
    v = np.transpose(H1.dot(H).dot(np.transpose(p)))
    w = np.transpose(H2.dot(H).dot(np.transpose(p)))
    print('p = \n'+str(p))
    print('v = \n'+str(v))
    print('w = \n'+str(w))
    H = HhatInv(H1, H2, p, v, w)
    
    print('H = \n'+str(H))


def HhatInvTest():
    H1 = np.array([
        [  0.2138,  -0.018484,   67.34774],
        [ -0.21618,   0.2113,   173.827],
        [ -8.029*0.0001,  -1.825*0.00001,   1]], np.float64)
    H2 = np.array([
        [  0.3186,  -1.2517*0.001,   99.27943],
        [ -0.17187,   0.260943843,   169.74],
        [ -5.552*0.0001,  -3.21849576*0.00001,   1]], np.float64)
    p = np.array([
        [100, 100],
        [200, 100],
        [300, 100],
        [100, 200],
        [200, 200],
        [300, 200]], np.float64)
    '''H1 = np.array([
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]], np.float64)
    H2 = np.array([
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]], np.float64)
    p = np.array([
        [100, 100],
        [200, 100],
        [100, 200],
        [200, 200]])'''
    v, w = LSChessboardTest()
    print('p = \n'+str(p))
    print('q1 = \n'+str(v))
    print('q2 = \n'+str(w))
    H_hatInv = HhatInv(H1, H2, p, v, w)
    
    print('H_hatInv = \n'+str(H_hatInv))


def LSChessboardTest():
    p1 = np.array([
        [100, 200],
        [220, 150],
        [340, 100],
        [100, 300],
        [220, 310],
        [340, 320]], np.float64)
    p2 = np.array([
        [110, 100],
        [230, 150],
        [350, 200],
        [90,  430],
        [230, 390],
        [370, 350]], np.float64)
    '''p1 = np.array([
        [100, 200],
        [220, 150],
        [100, 300],
        [220, 310]], np.float64)
    p2 = np.array([
        [110, 100],
        [230, 150],
        [90,  430],
        [230, 390]], np.float64)'''

    nrows = 2
    ncols = 3
    '''nrows, ncols = [2, 2]'''
    q1 = LSChessboard(p1, nrows, ncols)
    q2 = LSChessboard(p2, nrows, ncols)
    
    '''img1 = np.zeros((500, 500, 3), np.float64)
    img2 = np.zeros((500, 500, 3), np.float64)
    for i in range(nrows*ncols):
        cv2.circle(img1, (int(p1[i][0]),int(p1[i][1])), 10, (255,0,0), -1)
        cv2.circle(img1, (int(q1[i][0]),int(q1[i][1])), 10, (0,255,0), -1)
        cv2.circle(img2, (int(p2[i][0]),int(p2[i][1])), 10, (255,0,0), -1)
        cv2.circle(img2, (int(q2[i][0]),int(q2[i][1])), 10, (0,255,0), -1)
    cv2.imshow('image 1', img1)
    cv2.imshow('image 2', img2)
    cv2.imwrite('./data/leastSqChess/LSchess2.jpg', img2)

    k = cv2.waitKey(0) & 0xFF
    if k == 27 or k == ord('q'):         # wait for ESC or 'q' key to exit
        cv2.destroyAllWindows()'''
    return q1, q2



#LSChessboardTest()
HhatInvTest2()




 
 
 
 
 
 
