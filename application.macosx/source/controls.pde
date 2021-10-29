void keyPressed() {

  if (keyCode == BACKSPACE)
    data.erase();
}

void mouseDragged() {

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

void mouseWheel(MouseEvent event) {

  if (!isEditing)
    nEpochs -= event.getCount() * 100;
  else
    learningRate -= event.getCount() * learningRate * 0.1;
}

void mouseClicked() {

  if (!isEditing) {
    for (CheckBox cb : boxes)
      if (cb.overlaps(mouseX, mouseY) && !cb.cannotClick())
        cb.action();
  } else
    executeVisBoxes(mouseX, mouseY);
}
