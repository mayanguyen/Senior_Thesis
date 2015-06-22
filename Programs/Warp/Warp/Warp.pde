/*
 * Exploring Homography Estimation
 * Keith J. O'Hara <kohara@bard.edu>
 * October 2014
 * 
 * This sketch relates points in two images by a 3x3 homography matrix. 
 * You can snap pictures by pressing a key. The homography is then estimated using 
 * four points (really eight, four in each image) in the images by clicking on 
 * matching locations.   The resulting image shows both images in the reference 
 * frame of the first image using backward warping.
 * 
 * (1) First, understand how this code works.
 *
 * (2) Create a function that warps the image using forward mapping rather than
 *     backward mapping and compare the two methods. Which works better? Why?
 * 
 * (3) Create a modified sketch that allows the user to insert the webcam
 *     onto an arbitrary plane in a static image. You will need to:
 * 
 *   - use video and a static image from a file, rather than two static 
 *     images from the webcam
 *  
 *   OR
 * 
 * (3) stitch together more than two images 
 *     - either multiple pictures of a planar surface or 
 *       pictures taken from a rotated about a central point
 * 
 * * * * * * * * * * 
 * Lab submitted by:
 * Van Mai Nguyen Thi <vn4720@bard.edu>
 * October 2014
 * 
 * Lab 6, task (3)a: Ad placement
 *
 * I used forward warping to warp the camera image (displayed on the left half
 * of the window) and place it in a still image (displayed on the right half)
 * in a trapezoid area with four corners chosen by the user.
 * The points on the still image will correlate to the four corners of the camera
 * image and will be used to generate the homography matrix. The first (0th) point 
 * corresponds to the top-left corner of the camera image, and each next point
 * corresponds to the next corner in the clockwise manner.
 *
 * I chose to use forward as opposed to backward warping because most ad placements
 * involve scaling down the original image, so it is not likely that warping will create
 * gaps between pixels. Forward warping is also convenient in this case because it
 * is more convenient to loop through the pixels of the camera image, which are in a
 * perfect rectangle, without determining which pixels of the still image lie within
 * the chosen area. 
 *
 * In case the chosen area stretches out the original image in some direction so much 
 * that gaps will be created between pixels in the final image, I have left two lines
 * of code that are now commented out, which will make the program draw ellipses
 * instead of just pixels. (search: #bigAd)
 * However, the ellipse approach slows down the program significantly, so I am just
 * using the pixel method.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * *
 * Van Mai Nguyen Thi
 * 
 * This lab has been adapted to make a program to warp images.
 * 
 */

import Jama.*;
import processing.video.*;

boolean debug = false;

PImage orig;

//PImage img;

// points in our background image
PVector[] p1 = new PVector[4];

// corners in our original image
PVector[] p2;

int np = 0;

Matrix homography;

void setup() {
  orig = loadImage("black.jpg");
  size(orig.width, orig.height);
  
  // load background image
  //img = loadImage("img.jpg");
  //img.resize(orig.width, orig.height);

  // start top left, then clockwise
  p2 = new PVector[] {
    new PVector(0, 0, 1),
    new PVector(orig.width, 0, 1),
    new PVector(orig.width, orig.height, 1),
    new PVector(0, orig.height, 1)
  };

  noStroke();
  background(255);

  smooth();
  homography = Matrix.identity(3, 3);
  textFont(createFont("FFScala", 32));
}

PVector applyTransformation(Matrix h, PVector v) {
  Matrix u  = new Matrix(3, 1);
  u.set(0, 0, v.x);
  u.set(1, 0, v.y);
  u.set(2, 0, 1);
  Matrix t = h.times(u);
  return new PVector((float)(t.get(0, 0)/t.get(2, 0)), (float)(t.get(1, 0)/ (t.get(2, 0))));
} 

void mousePressed() {
  if (np < 4) {
    int i = int(mouseX / (orig.width)); 

    if (i == 0) {
      p1[np] = new PVector(mouseX % (orig.width), mouseY, 1);
      np++;
      println("bg " + p1[np-1]);
    }

    if (np == 4) {
      estimate();
    }
  }
}

void estimate() {
  if (np == 4) {

    double[][] a = new double[2*np][];
    
    // Creates the estimation matrix                                                                                                                                         
    for (int i = 0; i < np; i++) {
      double l1 [] = {
        p2[i].x, p2[i].y, p2[i].z, 0, 0, 0, -p2[i].x*p1[i].x, -p2[i].y*p1[i].x, -p1[i].x
      };
      double l2 [] = {
        0, 0, 0, p2[i].x, p2[i].y, p2[i].z, -p2[i].x*p1[i].y, -p2[i].y*p1[i].y, -p1[i].y
      };
      a[2*i] = l1;
      a[2*i+1] = l2;
    }

    Matrix A = new Matrix(a);
    Matrix X = A.transpose().times(A);

    // assumes the eigenvalues/vectors are sorted in increasing order
    EigenvalueDecomposition E = X.eig();  
    Matrix v = E.getV();

    if (debug) {
      double[] eigenvalues = E.getRealEigenvalues();
      println(eigenvalues);
      v.print(9, 9);
    }

    // create the homography matrix from the smallest eigenvector                                                                                                                
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        homography.set(i, j, v.get(i*3+j, 0));
      }
    }
  }

  println("Estimated H");
  homography.print(3, 3);
}

void forWarp() {
  // perform forward warping
  for (int x = 0; x < orig.width; x++) {
    for (int y = 0; y < orig.height; y++) {
      
      //grap pixel from original image
      color pix = color(255);
      float w0 = 1;
      int idx0 = y* orig.width + x;
      pix = orig.pixels[idx0];

      // get position to draw that pixel on the background image
      PVector np = applyTransformation(homography, new PVector(x, y, 1));
      int x1 = (int)np.x;
      int y1 = (int)np.y;

      if (x1 > 0 && x1< width && y1 > 0 && y1 < height) {
        int idx = y1* width + x1;
        pixels[idx] = pix;
        // #bigAd
        // Use these next 2 lines (and remove the 2 previous lines)
        // if you plan to place the camera image on an large (or not small) area.
        //fill(pix);
        //ellipse(x1, y1, 2, 2);
      }
    }
  }
}

void draw() {
  background(255);
  // draw the live image
  //image(orig, 0, 0);

  // draw the still image
  //image(img, img.width, 0);
  
  // draw the selected points on the bg image
  if (np < 4) {
    for (int i = 0; i < np; i++) {
      fill(255, 0, 0);
      ellipse(p1[i].x, p1[i].y, 2, 2);
    }
  }
  
  if (np == 4) {
    loadPixels();
    forWarp();
    updatePixels();
    saveFrame("warped_image.jpg");
    noLoop();
  }
  
}
