/**
 * Van Mai Nguyen Thi <vn4720@bard.edu>
 * October 24, 2014
 *
 * This is a simple program that creates a 640x480 .jpg image of a black
 * rectangle in the center and a white border. The thickness of the border
 * is controlled by variables XGAP and YGAP.
 */


void setup() {
  size(640, 480);
  background(200);
}

void draw() {
  //loadPixels();
  /*for (int x=XGAP; x<640; x++){
    for (int y=YGAP; y<480; y++) {
      int idx = y*width+x;
      pixels[idx] = color();
    }
  }*/
  //updatePixels();
  saveFrame("black.jpg");
  noLoop();
}
