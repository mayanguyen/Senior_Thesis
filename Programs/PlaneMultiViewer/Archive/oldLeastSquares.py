import numpy as np
import cv2


# Least Squares Chessboard
def LSChessboard(p, nrows, ncols):
    if (p.shape[0] != nrows*ncols or p.shape[1] != 2):
        print("Incorrect dimensions.")
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
        [sum1, sum2, sum3]], np.float32)

    b = np.array([[b1], [b2], [b3]], np.float32)

    AInv = np.linalg.inv(A)
    x, y, u = AInv.dot(b)

    q = np.zeros((nrows*ncols, 2), np.float32)
    for i in range(n):
        q[i] = [x+(i%ncols)*u, y+(i/ncols)*u]

    return np.around(q)


# Least Squares Optimal Homography for 2 cameras
# Return: H_hat (the optimal homography)
def HhatInv(H1, H2, p, q1, q2):
    if (q1.shape != q2.shape or q1.shape != p.shape
        or q2.shape != p.shape):
        print("The number of points is not matching.")
        return -1
    
    n = p.shape[0]
    
    # List of coefficients of h_i and constant term in the error function,
    # where h_i are entries of H_hat, and 0 <= i <= 8.
    e = np.zeros((n, 6, 9), np.float32)
    
    for i in range(n):
        for j in range(6):
            for k in range(9):
                a, b = [1, 0]
                if (k%3 < 2):
                    a = p[i][k%3]
                if (j < 3):
                    b = H1[j][k/3]
                    if (k == 8):
                        if (j != 2):
                            e[i][j][k] -= q1[i][j]
                        else:
                            e[i][j][k] -= 1
                else:
                    b = H2[j%3][k/3]
                    if (k == 8):
                        if (j != 5):
                            e[i][j][k] -= q2[i][j%3]
                        else:
                            e[i][j][k] -= 1
                e[i][j][k] += a*b

    #print('e = \n'+str(e))

    deriv = partialDeriv(e)
    #print('deriv = \n'+str(d))

    A = deriv[:,:8]
    AInv = np.linalg.inv(A)
    b = deriv[:,8:9]
    b *= (-1)

    H_hatInv = AInv.dot(b)
    H_hatInv = np.vstack([H_hatInv, [1]])
    H_hatInv = H_hatInv.flatten().reshape((-1, 3))

    print("A = \n"+str(A))
    print("b = \n"+str(b))
    
    return H_hatInv


# Return: list of coefficients of h_i and the const. term
# of the partial derivatives of e
def partialDeriv(e):
    n = e.shape[0]  # number of points, and also of iterations of the sigma
    d = np.zeros((8, 9), np.float32)
    
    # derivative with respect to h_v
    for v in range(8):
        for i in range(n):
            for j in range(6):
                coeff = e[i][j][v]
                for k in range(9):
                    d[v][k] += coeff*e[i][j][k]

    return d


def HhatInvTest():
    '''H1 = np.array([
        [  0.2138,  -0.018484,   67.34774],
        [ -0.21618,   0.2113,   173.827],
        [ -8.029*0.0001,  -1.825*0.00001,   1]], np.float32)
    H2 = np.array([
        [  0.3186,  -1.2517*0.001,   99.27943],
        [ -0.17187,   0.260943843,   169.74],
        [ -5.552*0.0001,  -3.21849576*0.00001,   1]], np.float32)
    p = np.array([
        [100, 100],
        [200, 100],
        [300, 100],
        [100, 200],
        [200, 200],
        [300, 200]])'''
    H1 = np.array([
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]], np.float32)
    H2 = np.array([
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]], np.float32)
    p = np.array([
        [100, 100],
        [200, 100],
        [100, 200],
        [200, 200]])
    q1, q2 = LSChessboardTest()
    print('p = \n'+str(p))
    print('q1 = \n'+str(q1))
    print('q2 = \n'+str(q2))
    H_hatInv = HhatInv(H1, H2, p, q1, q2)
    
    print('H_hatInv = \n'+str(H_hatInv))


def LSChessboardTest():
    '''p1 = np.array([
        [100, 200],
        [220, 150],
        [340, 100],
        [100, 300],
        [220, 310],
        [340, 320]], np.float32)
    p2 = np.array([
        [110, 100],
        [230, 150],
        [350, 200],
        [90,  430],
        [230, 390],
        [370, 350]], np.float32)'''
    p1 = np.array([
        [100, 200],
        [220, 150],
        [100, 300],
        [220, 310]], np.float32)
    p2 = np.array([
        [110, 100],
        [230, 150],
        [90,  430],
        [230, 390]], np.float32)

    '''nrows = 2
    ncols = 3'''
    nrows, ncols = [2, 2]
    q1 = LSChessboard(p1, nrows, ncols)
    q2 = LSChessboard(p2, nrows, ncols)
    
    img1 = np.zeros((500, 500, 3), np.float32)
    img2 = np.zeros((500, 500, 3), np.float32)
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
        cv2.destroyAllWindows()
    return q1, q2



#LSChessboardTest()
HhatInvTest()




 
 
 
 
 
 
