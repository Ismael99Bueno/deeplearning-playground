class DataSet {

  ArrayList<Point> points;
  ArrayList<String> labelCount;
  ArrayList<Color> colorCount;

  Vector[] trainSet;
  Vector[] labelSet;

  boolean isPrepared;
  int prepDataLabelCount;

  DataSet() {

    points = new ArrayList<Point>();
    labelCount = new ArrayList<String>();
    colorCount = new ArrayList<Color>();

    isPrepared = false;
    prepDataLabelCount = 0;
  }

  void add(Point pt) {

    points.add(pt);
    if (!labelCount.contains(pt.label)) {

      labelCount.add(pt.label);

      Color c = new Color(random(255), random(255), random(255));
      colorCount.add(c);
      pt.col = c.col;
    } else
      pt.col = toColor(pt.label);

    isPrepared = false;
  }

  color toColor(String label) {

    for (int i = 0; i < getLabelCount(); i++)
      if (labelCount.get(i).equals(label))
        return colorCount.get(i).col;

    throw new RuntimeException("Could not find any color");
  }

  void show() {

    for (Point pt : points)
      pt.show();
  }

  int getLabelCount() {

    return labelCount.size();
  }
  
  int getPrepDataLabelCount() {
   
    return prepDataLabelCount;
  }

  void erase() {

    points.clear();
    labelCount.clear();
    colorCount.clear();
    isPrepared = false;
  }

  void prepare() {
    
    prepDataLabelCount = labelCount.size();

    trainSet = new Vector[points.size()];
    labelSet = new Vector[points.size()];

    for (int i = 0; i < points.size(); i++) {

      Point pt = points.get(i);

      Vector sample = new Vector(new float[] {2 * pt.pos.x / width, 2 * pt.pos.y / height});
      Vector label = new Vector(getLabelCount());

      for (int j = 0; j < getLabelCount(); j++)
        if (labelCount.get(j).equals(pt.label)) {
          label.set(j, 1.0);
          break;
        }

      trainSet[i] = sample;
      labelSet[i] = label;
    }

    if (!isEmpty())
      isPrepared = true;
  }

  Vector[] getTrainSet() {

    return trainSet;
  }

  Vector[] getLabelSet() {

    return labelSet;
  }
  
  boolean isEmpty() {
   
    return getPrepDataLabelCount() == 0 || trainSet == null || trainSet.length == 0;
  }
}

class Color {

  color col;
  Color(float r, float g, float b) {

    col = color(r, g, b);
  }

  Color(color col) {

    this.col = col;
  }
}
