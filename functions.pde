void runGraphWindow() {

  graphs = new PlotWindow();
  String[] args = {graphs.getClass().getSimpleName()};
  runSketch(args, graphs);
}

void trainModel() {

  myModel.trainModel();
}

void showCheckBoxes() {

  for (CheckBox cb : boxes)
    cb.show();
}

void fillCanvas() {

  myModel.validateModel();

  fillCanvas.beginDraw();

  for (int i = 0; i < fillCanvas.width; i++) {
    float x = i - width / 2;
    for (int j = 0; j < fillCanvas.height; j++) {
      float y = height / 2 - j;

      if (isDiscrete)
        fillCanvas.set(i, j, data.toColor(myModel.predict(x, y)));
      else {
       
        float factor = 1.0 / myModel.myData.getPrepDataLabelCount();
        if (factor == 1.0)
          factor = 0;
        
        color rough = data.toColor(myModel.predict(x, y));
        color smooth = color(red(rough), green(rough), blue(rough), (myModel.getProbability() - factor) / (1.0 - factor) * 255);
        
        fillCanvas.set(i, j, smooth);
      }
    }
  }

  fillCanvas.endDraw();
}

void unFillCanvas() {

  fillCanvas.beginDraw();
  fillCanvas.clear();
  fillCanvas.endDraw();
}

boolean overlapsAnyCheckBox(float x, float y) {

  for (CheckBox cb : boxes)
    if (cb.overlaps(x, y))
      return true;

  return false;
}

void initCheckBoxes() {

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
