/**
 * Van Mai Nguyen Thi <vn4720@bard.edu>
 * October 24, 2014
 *
 * This is a simple program that creates a 640x480 .jpg image of a black
 * rectangle in the center and a white border. The thickness of the border
 * is controlled by variables XGAP and YGAP.
 */

int XGAP;
int YGAP;

void setup() {
  size(displayWidth, displayHeight);
  
  XGAP = width*2/5;
  YGAP = height*2/5;
}

void rectangle(int col) {
  if (col < 0 || col > 255) return;
  loadPixels();
  for (int x=XGAP; x<width-XGAP; x++){
    for (int y=YGAP; y<height-YGAP; y++) {
      int idx = y*width+x;
      pixels[idx] = color(col);
    }
  }
  updatePixels();
}

// Make a 6x9 chessboard (5x8 inner corners)
// with white border around
void chessBoard() {
  background(255);
  
  int w = min(width/11, height/8);
  println("w = "+w);
  
  loadPixels();
  for (int ix=1; ix<10; ix+=2){
    for (int iy=1; iy<7; iy+=2) {
      for (int x=ix*w; x<(ix+1)*w; x++) {
        for (int y=iy*w; y<(iy+1)*w; y++) {
          int idx = y*width+x;
          pixels[idx] = color(0);
        }
      }
    }
  }
  for (int ix=2; ix<10; ix+=2){
    for (int iy=2; iy<7; iy+=2) {
      for (int x=ix*w; x<(ix+1)*w; x++) {
        for (int y=iy*w; y<(iy+1)*w; y++) {
          int idx = y*width+x;
          pixels[idx] = color(0);
        }
      }
    }
  }
  updatePixels();
}

void draw() {
  if (keyPressed) {
    if (key == ' ') {
      background(255);
      saveFrame("blank.jpg");
    /*} else if (key == 's') { // small blank rectangle
      size(displayWidth/5, displayHeight/5);
      background(255);
      saveFrame("small_blank.jpg");*/
    } else if (key == 'r') {
      background(0);
      rectangle(255);
      saveFrame("small_rect.jpg");
    } else if (key == 'a') {
      background(0);
      PImage img = loadImage("anamorph.jpg");
      image(img, 0, 0);
    } else if (key == 'c') {
      chessBoard();
      saveFrame("chessboard.jpg");
    }
    noLoop();
  }
  
  //saveFrame("small_rect.jpg");
  //noLoop();
}
