class Point {
 
  PVector pos;
  float size;
  color col;
  String label;
  
  Point(float x, float y, String label) {
   
    this.pos = new PVector(x, y);
    this.label = label.toUpperCase();
    
    size = 20;
    col = color(255);
  }
  
  Point(String label) {
   
    this(random( - width / 2, width / 2), random( - height / 2, height / 2), label);
  }
  
  void show() {
   
    push();
    
    translate(width / 2, height / 2);
    scale(1, -1);
    
    noFill();
    stroke(col);
    circle(pos.x, pos.y, size * 2);
    
    pop();
    
    fill(col);
    textAlign(CENTER);
    textSize(size);
    text(label, pos.x + width / 2, height / 2 - pos.y);
   
  }
}
