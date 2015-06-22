/**
 * Author: Van Mai Nguyen Thi
 * Senior Project
 * October 2014
 * 
 * 1. Calibrate: 
 *    a. Project 4 bright dots
 *    b. Camera: detect dots 
 *    c. Generate homography H relating camera image and projector image
 * 2. Using the homography matrices, warp the image to create anamorphosis
 * 3. Projector: display the resulting image
 * 
 */

import gab.opencv.*;
import processing.video.*;

OpenCV opencv;

PImage img;
PImage[] calibpts;


void setup() {
  size(640, 480);
  opencv = new OpenCV(this, width, height);
  
  //opencv.startBackgroundSubtraction(5, 3, 0.5);
  
  img = loadImage("test.jpg");
  
  calibrate();
}


void calibrate() {
  // TODO stub
  calibpts = new PImage[4];
  calibpts[0] = loadImage("TL.jpg");
  calibpts[1] = loadImage("TR.jpg");
  calibpts[2] = loadImage("BL.jpg");
  calibpts[3] = loadImage("BR.jpg");
  
  /*for (int i=0; i<calibpts.length; i++) {
    calibpts[i].filter(GRAY);
  }*/
  
  opencv.loadImage("TL.jpg",GRAY);
  opencv.
  
}

void draw() {
  //image(img, 0, 0);
  
  image(calibpts[0], 0, 0, width/2, height/2);
  image(calibpts[1], width/2, 0, width/2, height/2);
  image(calibpts[2], 0, height/2, width/2, height/2);
  image(calibpts[3], width/2, height/2, width/2, height/2);
  
  

  noFill();
  stroke(255, 0, 0);
  strokeWeight(3);
  //for (Contour contour : opencv.findContours()) {
    //contour.draw();
  //}
}
