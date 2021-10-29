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

  void prepareModel() {
    
    model = getModel();
    model.optimizer(Optimizer.LOSS.CROSSENTROPY);

    eraseData();
  }

  void trainModel() {
    
    opt = model.new Options(model) {

      @Override
        public void tweak() {

        lr = learningRate;
        shuffle = true;
        epochs = nEpochs;
        valSplit = 0.2;
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

  String predict(float x, float y) {

    validateModel();
    
    Vector guess = model.feedForward(new Vector(new float[] {2 * x / width, 2 * y / height}));
    
    int selection = guess.indexMax();
    predProbability = guess.get(selection);
    
    return myData.labelCount.get(selection);
  }

  void eraseData() {

    epochTrainLoss.clear();
    epochValLoss.clear();
    totEpochs.clear();
    epochCount = 1;
  }

  float[] getTotEpochs() {

    float[] result = new float[totEpochs.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = totEpochs.get(i);

    return result;
  }

  float[] getTLoss() {

    float[] result = new float[epochTrainLoss.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = epochTrainLoss.get(i);

    return result;
  }

  float[] getVLoss() {

    float[] result = new float[epochValLoss.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = epochValLoss.get(i);

    return result;
  }

  void validateModel() {

    if (!isPrepared())
      throw new RuntimeException("Model is not prepared");

    if (!isUpdated())
      throw new RuntimeException("You have introduced new labels: you have to prepare the model again, or the model is empty");
  }

  boolean isPrepared() {

    return model != null;
  }

  boolean isUpdated() {

    if (model.isEmpty())
      return false;

    if (model.hasDense())
      return model.getLastDense().getNeurons() == myData.getPrepDataLabelCount();

    return model.getLastConv().getPooledOutSize() == myData.getPrepDataLabelCount();
  }

  boolean isPredicting() {

    return predicting;
  }

  boolean isTraining() {

    return training;
  }
  
  float getProbability() {
   
    return predProbability;
  }

  void predicting(boolean p) {

    predicting = p;
  }
}
