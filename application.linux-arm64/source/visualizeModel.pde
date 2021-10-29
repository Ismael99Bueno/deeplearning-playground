ArrayList<CheckBox> add, remove, switchAct;

List<Sequential.ACTIVATION> actLabels;

ArrayList<Integer> count, labelIndexes;
int index;

int maxN = 16;
int maxL = 8;

Visualizer vis;

CheckBox useThisToggle, addL, removeL;
CheckBox[] visBoxes;

void initModelDraw() {

  Sequential initial = new Sequential();

  int out = myModel.myData.getPrepDataLabelCount();
  initial.add(new Dense(2, out > 0 ? out : 1, Sequential.ACTIVATION.SOFTMAX));
  initial.optimizer(Optimizer.LOSS.CROSSENTROPY);

  vis = new Visualizer(this, initial);

  initLists();
  initCB();

  setModel();
}

Sequential getModel() {

  setModel();
  return new Sequential(vis.getModel());
}

void setModel() {

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

void drawModel() {

  vis.draw();
  showVisBoxes();
}

void showVisBoxes() {

  for (CheckBox cb : visBoxes)
    cb.show();

  for (CheckBox cb : switchAct)
    cb.show();

  for (CheckBox cb : add)
    cb.show();

  for (CheckBox cb : remove)
    cb.show();
}

void executeVisBoxes(float x, float y) {

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

void initLists() {

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

void initCB(int i) { 

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

void initCB() {

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
