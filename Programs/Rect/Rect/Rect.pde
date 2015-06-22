/**
 * Van Mai Nguyen Thi <vn4720@bard.edu>
 * October 24, 2014
 *
 * This is a simple program that creates a 640x480 .jpg image of a black
 * rectangle in the center and a white border. The thickness of the border
 * is controlled by variables XGAP and YGAP.
 */

int WIDTH = 848; //640;
int HEIGHT = 480;

int XGAP = WIDTH*2/5;
int YGAP = HEIGHT*2/5;

void setup() {
  size(WIDTH, HEIGHT);
  background(0,255,0);
}

void draw() {
  loadPixels();
  for (int x=XGAP; x<WIDTH-XGAP; x++){
    for (int y=YGAP; y<HEIGHT-YGAP; y++) {
      int idx = y*width+x;
      pixels[idx] = color(255);
    }
  }
  updatePixels();
  saveFrame("small_rect.jpg");
  noLoop();
}
