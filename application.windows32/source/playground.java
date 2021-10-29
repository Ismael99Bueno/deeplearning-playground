import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import checkBox.*; 
import tensors.Float.*; 
import deepLearning.utilities.*; 
import grafica.*; 
import java.util.List; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class playground extends PApplet {







PlotWindow graphs;

GPlot lossPlot;
Network myModel;
DataSet data;

CheckBox prepModelToggle, prepDataToggle, trainToggle, predictToggle, showDataToggle, fillCanvasToggle, erasePlotToggle, editModelToggle, discreteToggle;
CheckBox[] boxes;

PGraphics fillCanvas;

boolean showData = true;
boolean isFilled = false;
boolean isEditing = false;
boolean isDiscrete = true;

int nEpochs = 500;

float learningRate = 0.001f;

public void settings() {
  
  data = new DataSet();
  myModel = new Network(data);
  
  size(1280, 720);
}

public void setup() {
  
  runGraphWindow();
  initCheckBoxes();
  key = 'a';
  
  fillCanvas = createGraphics(width, height);
  
  initModelDraw();
}

public void draw() {
  background(0);
  
  if (!isEditing) {
    image(fillCanvas, 0, 0);
    
    if (showData)
      data.show();
      
    showCheckBoxes();
    
    textSize(20);
    fill(255);
    textAlign(CORNER);
    text("Epochs: " + nEpochs, 10, height - 60);
  } else {
    drawModel();
    textSize(20);
    fill(255);
    textAlign(CORNER);
    text("Learning Rate: " + round(learningRate * 1e8f) / 1e8f, 10, height - 60);
  }
}
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

  public void add(Point pt) {

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

  public int toColor(String label) {

    for (int i = 0; i < getLabelCount(); i++)
      if (labelCount.get(i).equals(label))
        return colorCount.get(i).col;

    throw new RuntimeException("Could not find any color");
  }

  public void show() {

    for (Point pt : points)
      pt.show();
  }

  public int getLabelCount() {

    return labelCount.size();
  }
  
  public int getPrepDataLabelCount() {
   
    return prepDataLabelCount;
  }

  public void erase() {

    points.clear();
    labelCount.clear();
    colorCount.clear();
    isPrepared = false;
  }

  public void prepare() {
    
    prepDataLabelCount = labelCount.size();

    trainSet = new Vector[points.size()];
    labelSet = new Vector[points.size()];

    for (int i = 0; i < points.size(); i++) {

      Point pt = points.get(i);

      Vector sample = new Vector(new float[] {2 * pt.pos.x / width, 2 * pt.pos.y / height});
      Vector label = new Vector(getLabelCount());

      for (int j = 0; j < getLabelCount(); j++)
        if (labelCount.get(j).equals(pt.label)) {
          label.set(j, 1.0f);
          break;
        }

      trainSet[i] = sample;
      labelSet[i] = label;
    }

    if (!isEmpty())
      isPrepared = true;
  }

  public Vector[] getTrainSet() {

    return trainSet;
  }

  public Vector[] getLabelSet() {

    return labelSet;
  }
  
  public boolean isEmpty() {
   
    return getPrepDataLabelCount() == 0 || trainSet == null || trainSet.length == 0;
  }
}

class Color {

  int col;
  Color(float r, float g, float b) {

    col = color(r, g, b);
  }

  Color(int col) {

    this.col = col;
  }
}
class Network {

  Sequential model;
  Sequential.Options opt;

  ArrayList<Float> epochTrainLoss, epochValLoss;
  ArrayList<Integer> totEpochs;
  int epochCount;

  boolean predicting, training;
  float predProbability;

  DataSet myData;

  Network(DataSet data) {

    myData = data;

    epochTrainLoss = new ArrayList<Float>();
    epochValLoss = new ArrayList<Float>();
    totEpochs = new ArrayList<Integer>();
    
    epochCount = 1;

    predicting = false;
    training = false;
    
    model = new Sequential();
  }

  public void prepareModel() {
    
    model = getModel();
    model.optimizer(Optimizer.LOSS.CROSSENTROPY);

    eraseData();
  }

  public void trainModel() {
    
    opt = model.new Options(model) {

      @Override
        public void tweak() {

        lr = learningRate;
        shuffle = true;
        epochs = nEpochs;
        valSplit = 0.2f;
      }

      @Override
        public void onEpochEnd(int epoch, float tLoss, float vLoss, String log) {

        epochTrainLoss.add(tLoss);
        epochValLoss.add(vLoss);
        totEpochs.add(epochCount++);
      }
    };
    
    training = true;
    validateModel();

    Vector[] trainSet = myData.getTrainSet();
    Vector[] labelSet = myData.getLabelSet();

    model.fit(trainSet, labelSet, opt);
    training = false;
  }

  public String predict(float x, float y) {

    validateModel();
    
    Vector guess = model.feedForward(new Vector(new float[] {2 * x / width, 2 * y / height}));
    
    int selection = guess.indexMax();
    predProbability = guess.get(selection);
    
    return myData.labelCount.get(selection);
  }

  public void eraseData() {

    epochTrainLoss.clear();
    epochValLoss.clear();
    totEpochs.clear();
    epochCount = 1;
  }

  public float[] getTotEpochs() {

    float[] result = new float[totEpochs.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = totEpochs.get(i);

    return result;
  }

  public float[] getTLoss() {

    float[] result = new float[epochTrainLoss.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = epochTrainLoss.get(i);

    return result;
  }

  public float[] getVLoss() {

    float[] result = new float[epochValLoss.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = epochValLoss.get(i);

    return result;
  }

  public void validateModel() {

    if (!isPrepared())
      throw new RuntimeException("Model is not prepared");

    if (!isUpdated())
      throw new RuntimeException("You have introduced new labels: you have to prepare the model again, or the model is empty");
  }

  public boolean isPrepared() {

    return model != null;
  }

  public boolean isUpdated() {

    if (model.isEmpty())
      return false;

    if (model.hasDense())
      return model.getLastDense().getNeurons() == myData.getPrepDataLabelCount();

    return model.getLastConv().getPooledOutSize() == myData.getPrepDataLabelCount();
  }

  public boolean isPredicting() {

    return predicting;
  }

  public boolean isTraining() {

    return training;
  }
  
  public float getProbability() {
   
    return predProbability;
  }

  public void predicting(boolean p) {

    predicting = p;
  }
}
class PlotWindow extends PApplet {

  public void settings() {
    size(640, 360);
  }

  public void setup() {

    lossPlot = new GPlot(this, 0, 0, width, height);
    lossPlot.addLayer("Train Loss", new GPointsArray());
    lossPlot.addLayer("Val Loss", new GPointsArray());

    lossPlot.getXAxis().setAxisLabelText("Epochs");
    lossPlot.getYAxis().setAxisLabelText("Model Loss");

    lossPlot.getLayer("Train Loss").setLineColor(color(255, 0, 0));
    lossPlot.getLayer("Train Loss").setPointColor(color(255, 0, 0));
    lossPlot.getLayer("Train Loss").setPointSize(1);

    lossPlot.getLayer("Val Loss").setLineColor(color(0, 0, 255));
    lossPlot.getLayer("Val Loss").setPointColor(color(0, 0, 255));
    lossPlot.getLayer("Val Loss").setPointSize(1);
  }

  int threshold = 0;
  public void draw() {

    if (threshold > myModel.totEpochs.size()) {

      lossPlot.setPoints(new GPointsArray(), "Train Loss");
      lossPlot.setPoints(new GPointsArray(), "Val Loss");

      threshold = 0;
    } else
      for (int i = threshold; i < myModel.totEpochs.size(); i++) {

        int epoch = myModel.totEpochs.get(i);

        float tLoss = myModel.epochTrainLoss.get(i);
        float vLoss = myModel.epochValLoss.get(i);

        lossPlot.addPoint(epoch, tLoss, "Train Loss", "Train Loss");
        lossPlot.addPoint(epoch, vLoss, "Val Loss", "Val Loss");
      }

    threshold = myModel.totEpochs.size();    
    lossPlot.defaultDraw();
    lossPlot.drawLines();
  }
}
class Point {
 
  PVector pos;
  float size;
  int col;
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
  
  public void show() {
   
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
public void keyPressed() {

  if (keyCode == BACKSPACE)
    data.erase();
}

public void mouseDragged() {

  if (!overlapsAnyCheckBox(mouseX, mouseY)) {

    float px = mouseX - width / 2;
    float py = height / 2 - mouseY;

    String label;
    if (myModel.isPredicting())
      label = myModel.predict(px, py);
    else
      label = String.valueOf(key);

    data.add(new Point(px, py, label));
  }
}

public void mouseWheel(MouseEvent event) {

  if (!isEditing)
    nEpochs -= event.getCount() * 100;
  else
    learningRate -= event.getCount() * learningRate * 0.1f;
}

public void mouseClicked() {

  if (!isEditing) {
    for (CheckBox cb : boxes)
      if (cb.overlaps(mouseX, mouseY) && !cb.cannotClick())
        cb.action();
  } else
    executeVisBoxes(mouseX, mouseY);
}
public void runGraphWindow() {

  graphs = new PlotWindow();
  String[] args = {graphs.getClass().getSimpleName()};
  runSketch(args, graphs);
}

public void trainModel() {

  myModel.trainModel();
}

public void showCheckBoxes() {

  for (CheckBox cb : boxes)
    cb.show();
}

public void fillCanvas() {

  myModel.validateModel();

  fillCanvas.beginDraw();

  for (int i = 0; i < fillCanvas.width; i++) {
    float x = i - width / 2;
    for (int j = 0; j < fillCanvas.height; j++) {
      float y = height / 2 - j;

      if (isDiscrete)
        fillCanvas.set(i, j, data.toColor(myModel.predict(x, y)));
      else {
       
        float factor = 1.0f / myModel.myData.getPrepDataLabelCount();
        if (factor == 1.0f)
          factor = 0;
        
        int rough = data.toColor(myModel.predict(x, y));
        int smooth = color(red(rough), green(rough), blue(rough), (myModel.getProbability() - factor) / (1.0f - factor) * 255);
        
        fillCanvas.set(i, j, smooth);
      }
    }
  }

  fillCanvas.endDraw();
}

public void unFillCanvas() {

  fillCanvas.beginDraw();
  fillCanvas.clear();
  fillCanvas.endDraw();
}

public boolean overlapsAnyCheckBox(float x, float y) {

  for (CheckBox cb : boxes)
    if (cb.overlaps(x, y))
      return true;

  return false;
}

public void initCheckBoxes() {

  int w = 150;
  int h = 50;

  int x = 0;
  int y = height - h;

  prepModelToggle = new CheckBox(this, x, y, w, h, "Prepare model") {

    @Override public void action() {
      myModel.prepareModel();
    }

    @Override public boolean hasToClick() {
      return !myModel.isPrepared() || !myModel.isUpdated();
    }

    @Override public boolean canClick() {
      return false; //No lo voy a usar
    }

    @Override public boolean cannotClick() {
      return myModel.isTraining();
    }

    @Override public boolean isDone() {
      return myModel.isPrepared() && myModel.isUpdated();
    }
  };

  prepDataToggle = new CheckBox(this, x + w, y, w, h, "Prepare data") {

    @Override public void action() {
      data.prepare();
    }

    @Override public boolean hasToClick() {
      return !data.isPrepared;
    }

    @Override public boolean canClick() {
      return false;
    }

    @Override public boolean cannotClick() {
      return myModel.isTraining();
    }

    @Override public boolean isDone() {
      return data.isPrepared;
    }
  };

  trainToggle = new CheckBox(this, x + 2 * w, y, w, h, "Train model") {

    @Override public void action() {
      thread("trainModel");
    }

    @Override public boolean hasToClick() {
      return false;
    }

    @Override public boolean canClick() {
      return myModel.isUpdated() && myModel.isPrepared();
    }

    @Override public boolean cannotClick() {
      return !myModel.isUpdated() || !myModel.isPrepared() || myModel.isTraining() || myModel.myData.isEmpty();
    }

    @Override public boolean isDone() {
      return false;
    }
  };

  predictToggle = new CheckBox(this, x + 3 * w, y, w, h, "Predict") {

    @Override public void action() {
      myModel.predicting(!myModel.isPredicting());
    }

    @Override public boolean hasToClick() {
      return false;
    }

    @Override public boolean canClick() {
      return myModel.isUpdated() && myModel.isPrepared();
    }

    @Override public boolean cannotClick() {
      return !myModel.isUpdated() || !myModel.isPrepared() || myModel.isTraining() || myModel.myData.isEmpty();
    }

    @Override public boolean isDone() {
      return myModel.isPredicting();
    }
  };

  showDataToggle = new CheckBox(this, x + 4 * w, y, w, h, "Show data") {

    @Override public void action() {
      showData = !showData;
    }

    @Override public boolean hasToClick() {
      return false;
    }

    @Override public boolean canClick() {
      return true;
    }

    @Override public boolean cannotClick() {
      return false;
    }

    @Override public boolean isDone() {
      return showData;
    }
  };

  fillCanvasToggle = new CheckBox(this, x + 5 * w, y, w, h, "Fill canvas") {

    @Override public void action() {
      if (isFilled)
        unFillCanvas();
      else
        fillCanvas();

      isFilled = !isFilled;
    }

    @Override public boolean hasToClick() {
      return false;
    }

    @Override public boolean canClick() {
      return myModel.isPrepared() && myModel.isUpdated();
    }

    @Override public boolean cannotClick() {
      return !myModel.isPrepared() || !myModel.isUpdated() || myModel.isTraining() || myModel.myData.isEmpty();
    }

    @Override public boolean isDone() {
      return isFilled;
    }
  };

  erasePlotToggle = new CheckBox(this, x + 6 * w, y, w, h, "Erase plot") {

    @Override public void action() {

      myModel.eraseData();
    }

    @Override public boolean isDone() {

      return myModel.totEpochs.isEmpty();
    }

    @Override public boolean cannotClick() {

      return myModel.isTraining();
    }
  };

  editModelToggle = new CheckBox(this, x + 7 * w, y, w, h, "Edit model") {

    @Override public void action() {

      if (!isEditing)
        setModel();

      isEditing = !isEditing;
    }

    @Override public boolean cannotClick() {

      return myModel.isTraining() || !myModel.isPrepared();
    }
  };

  discreteToggle = new CheckBox(this, x + 5 * w, y - 2 * h / 3, w, 2 * h / 3, "Discretize") {

    @Override public void action() {

      isDiscrete = !isDiscrete;
      if (isFilled)
        fillCanvas();
    }

    @Override public boolean isDone() {

      return isDiscrete;
    }
  };

  boxes = new CheckBox[] {prepModelToggle, prepDataToggle, trainToggle, predictToggle, showDataToggle, fillCanvasToggle, erasePlotToggle, editModelToggle, discreteToggle};

  for (CheckBox cb : boxes)
    cb.tweak();
}
ArrayList<CheckBox> add, remove, switchAct;

List<Sequential.ACTIVATION> actLabels;

ArrayList<Integer> count, labelIndexes;
int index;

int maxN = 16;
int maxL = 8;

Visualizer vis;

CheckBox useThisToggle, addL, removeL;
CheckBox[] visBoxes;

public void initModelDraw() {

  Sequential initial = new Sequential();

  int out = myModel.myData.getPrepDataLabelCount();
  initial.add(new Dense(2, out > 0 ? out : 1, Sequential.ACTIVATION.SOFTMAX));
  initial.optimizer(Optimizer.LOSS.CROSSENTROPY);

  vis = new Visualizer(this, initial);

  initLists();
  initCB();

  setModel();
}

public Sequential getModel() {

  setModel();
  return new Sequential(vis.getModel());
}

public void setModel() {

  add.clear();
  remove.clear();
  switchAct.clear();
  
  Sequential toBeSet = new Sequential();
  toBeSet.optimizer(Optimizer.LOSS.CROSSENTROPY);

  for (int i = 0; i < count.size(); i++) {

    if (i == 0)
      toBeSet.add(new Dense(2, count.get(i), actLabels.get(labelIndexes.get(i))));
    else
      toBeSet.add(new Dense(count.get(i), actLabels.get(labelIndexes.get(i))));
  }

  int out = myModel.myData.getPrepDataLabelCount();

  if (count.size() > 0)
    toBeSet.add(new Dense(out > 0 ? out : 1, Sequential.ACTIVATION.SOFTMAX));
  else
    toBeSet.add(new Dense(2, out > 0 ? out : 1, Sequential.ACTIVATION.SOFTMAX));

  vis.setModel(toBeSet);
  vis.updateAsModel();

  index = 0;
  for (int i = 0; i < count.size(); i++) {

    initCB(i);

    add.get(i).setFloat(count.get(i));
    remove.get(i).setFloat(count.get(i));
    switchAct.get(i).setFloat(labelIndexes.get(i));

    add.get(i).setInt(i);
    remove.get(i).setInt(i);
    switchAct.get(i).setInt(i);

    index++;
  }
  
  if (out == toBeSet.getLastDense().getNeurons() && out == myModel.myData.getLabelCount() && out > 0)
    vis.setOutputLabels(myModel.myData.labelCount);
    
  vis.setInputLabels(new String[] {"X", "Y"});
}

public void drawModel() {

  vis.draw();
  showVisBoxes();
}

public void showVisBoxes() {

  for (CheckBox cb : visBoxes)
    cb.show();

  for (CheckBox cb : switchAct)
    cb.show();

  for (CheckBox cb : add)
    cb.show();

  for (CheckBox cb : remove)
    cb.show();
}

public void executeVisBoxes(float x, float y) {

  boolean hasClicked = false;

  for (CheckBox cb : visBoxes)
    if (cb.overlaps(x, y) && !cb.cannotClick()) {
      cb.action();
      hasClicked = true;
    }

  for (CheckBox cb : switchAct)
    if (cb.overlaps(x, y) && !cb.cannotClick()) {
      cb.action();
      hasClicked = true;
    }

  for (CheckBox cb : add)
    if (cb.overlaps(x, y) && !cb.cannotClick()) {
      cb.action();
      hasClicked = true;
    }

  for (CheckBox cb : remove)
    if (cb.overlaps(x, y) && !cb.cannotClick()) {
      cb.action();
      hasClicked = true;
    }

  if (hasClicked)
    setModel();    
}

public void initLists() {

  add = new ArrayList<CheckBox>();
  remove = new ArrayList<CheckBox>();
  switchAct = new ArrayList<CheckBox>();

  actLabels = Sequential.getActLabels();

  labelIndexes = new ArrayList<Integer>();
  count = new ArrayList<Integer>();

  for (int i = 0; i < myModel.model.getDenseCount() - 1; i++) {
    count.add(vis.getModel().getDense().get(i).getNeurons());
    labelIndexes.add(1);
  }
}

public void initCB(int i) { 

  int w = 30;
  int h = 30;

  float y = height - 20;
  float x = vis.getPos(i + 1).get(0).x;
  add.add(new CheckBox(this, x, y, w, h, "+") {

    @Override public void action() {

      count.set(getInt(), round(getFloat()) + 1);
    }

    @Override public boolean cannotClick() {

      return count.get(getInt()) >= maxN;
    }
  }
  );

  remove.add(new CheckBox(this, x - 30, y, w, h, "-") {

    @Override public void action() {

      count.set(getInt(), round(getFloat()) - 1);
    }

    @Override public boolean cannotClick() {

      return count.get(getInt()) <= 1;
    }
  }
  );

  switchAct.add(new CheckBox(this, x - (w + 10) / 2, y - 3 * h / 2, w + 10, h, "Act") {

    @Override public void action() {
      
      if (round(getFloat()) + 1 >= actLabels.size())
        setFloat(- 1);
        
      labelIndexes.set(getInt(), round(getFloat()) + 1);
    }

  }
  );
}

public void initCB() {

  useThisToggle = new CheckBox(this, 0, height - 50, 150, 50, "Use model") {
    @Override public void action() {
      isEditing = !isEditing;
    }
  };

  int w = 30;
  int h = 30;

  int x = 0;
  int y = 0;

  addL = new CheckBox(this, x, y, w, h, "+") {

    @Override public void action() {

      count.add(1);
      labelIndexes.add(1);
    }

    @Override public boolean cannotClick() {

      return count.size() >= maxL;
    }
  };

  removeL = new CheckBox(this, x + w, y, w, h, "-") {

    @Override public void action() {

      count.remove(count.get(count.size() - 1));
      labelIndexes.remove(labelIndexes.get(labelIndexes.size() - 1));
    }

    @Override public boolean cannotClick() {

      return count.isEmpty();
    }
  };

  visBoxes = new CheckBox[] {useThisToggle, addL, removeL};
}
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "playground" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
