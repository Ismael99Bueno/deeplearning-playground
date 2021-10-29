class PlotWindow extends PApplet {

  void settings() {
    size(640, 360);
  }

  void setup() {

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
  void draw() {

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
