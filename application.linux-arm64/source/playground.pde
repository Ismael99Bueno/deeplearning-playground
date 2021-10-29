import checkBox.*;
import tensors.Float.*;
import deepLearning.utilities.*;
import grafica.*;
import java.util.List;

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

float learningRate = 0.001;

void settings() {
  
  data = new DataSet();
  myModel = new Network(data);
  
  size(1280, 720);
}

void setup() {
  
  runGraphWindow();
  initCheckBoxes();
  key = 'a';
  
  fillCanvas = createGraphics(width, height);
  
  initModelDraw();
}

void draw() {
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
    text("Learning Rate: " + round(learningRate * 1e8) / 1e8, 10, height - 60);
  }
}
