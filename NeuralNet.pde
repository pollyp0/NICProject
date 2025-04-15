class NeuralNet {
  
  int iNodes, hNodes, oNodes, hLayers;
  Matrix[] weights;

  NeuralNet(int input, int hidden, int output, int hiddenLayers) {
    iNodes = input;
    hNodes = hidden;
    oNodes = output;
    hLayers = hiddenLayers;
    
    weights = new Matrix[hLayers + 1];
    weights[0] = new Matrix(hNodes, iNodes + 1);
    for (int i = 1; i < hLayers; i++) {
      weights[i] = new Matrix(hNodes, hNodes + 1);
    }
    weights[weights.length - 1] = new Matrix(oNodes, hNodes + 1);
    
    for (Matrix w : weights) {
      w.randomize();
    }
  }

  void mutate(float mr) {
    for (Matrix w : weights) {
      w.mutate(mr);
    }
  }

  // [MODIFIED] Обновлённая output-функция с Leaky ReLU
  float[] output(float[] inputsArr) {
    Matrix inputs = weights[0].singleColumnMatrixFromArray(inputsArr);
    Matrix curr_bias = inputs.addBias();
    
    for (int i = 0; i < hLayers; i++) {
      Matrix hidden_ip = weights[i].dot(curr_bias);
      Matrix hidden_op = activationFunction(hidden_ip);  // [MODIFIED]
      curr_bias = hidden_op.addBias();
    }
    
    Matrix output_ip = weights[weights.length - 1].dot(curr_bias);
    Matrix output = activationFunction(output_ip);  // [MODIFIED]
    
    return output.toArray();
  }

  // [MODIFIED] Универсальная функция активации (сейчас Leaky ReLU)
  Matrix activationFunction(Matrix input) {
    Matrix out = new Matrix(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
        float val = input.matrix[i][j];
        // Leaky ReLU вместо sigmoid
        out.matrix[i][j] = val >= 0 ? val : 0.01f * val;
        // Чтобы вернуть sigmoid: закомментируй строку выше и раскомментируй:
        // out.matrix[i][j] = 1.0 / (1.0 + exp(-val));
      }
    }
    return out;
  }

  NeuralNet crossover(NeuralNet partner) {
    NeuralNet child = new NeuralNet(iNodes, hNodes, oNodes, hLayers);
    for (int i = 0; i < weights.length; i++) {
      child.weights[i] = weights[i].crossover(partner.weights[i]);
    }
    return child;
  }

  NeuralNet clone() {
    NeuralNet clone = new NeuralNet(iNodes, hNodes, oNodes, hLayers);
    for (int i = 0; i < weights.length; i++) {
      clone.weights[i] = weights[i].clone();
    }
    return clone;
  }

  void load(Matrix[] weight) {
    for (int i = 0; i < weights.length; i++) {
      weights[i] = weight[i];
    }
  }

  Matrix[] pull() {
    return weights.clone();
  }

  void show(float x, float y, float w, float h, float[] vision, float[] decision) {
    float space = 5;
    float nSize = (h - (space * (iNodes - 2))) / iNodes;
    float nSpace = (w - (weights.length * nSize)) / weights.length;
    float hBuff = (h - (space * (hNodes - 1)) - (nSize * hNodes)) / 2;
    float oBuff = (h - (space * (oNodes - 1)) - (nSize * oNodes)) / 2;
    
    int maxIndex = 0;
    for (int i = 1; i < decision.length; i++) {
      if (decision[i] > decision[maxIndex]) {
        maxIndex = i;
      }
    }
    
    int lc = 0;

    // DRAW INPUT NODES
    for (int i = 0; i < iNodes; i++) {
      fill(vision[i] != 0 ? color(0, 255, 0) : 255);
      stroke(0);
      ellipseMode(CORNER);
      ellipse(x, y + (i * (nSize + space)), nSize, nSize);
      textSize(nSize / 2);
      textAlign(CENTER, CENTER);
      fill(0);
      text(i, x + (nSize / 2), y + (nSize / 2) + (i * (nSize + space)));
    }

    lc++;

    // DRAW HIDDEN NODES
    for (int a = 0; a < hLayers; a++) {
      for (int i = 0; i < hNodes; i++) {
        fill(255);
        stroke(0);
        ellipseMode(CORNER);
        ellipse(x + (lc * nSize) + (lc * nSpace), y + hBuff + (i * (nSize + space)), nSize, nSize);
      }
      lc++;
    }

    // DRAW OUTPUT NODES
    for (int i = 0; i < oNodes; i++) {
      fill(i == maxIndex ? color(0, 255, 0) : 255);
      stroke(0);
      ellipseMode(CORNER);
      ellipse(x + (lc * nSpace) + (lc * nSize), y + oBuff + (i * (nSize + space)), nSize, nSize);
    }

    lc = 1;

    // DRAW WEIGHTS
    for (int i = 0; i < weights[0].rows; i++) {
      for (int j = 0; j < weights[0].cols - 1; j++) {
        stroke(weights[0].matrix[i][j] < 0 ? color(255, 0, 0) : color(0, 0, 255));
        line(x + nSize, y + (nSize / 2) + (j * (space + nSize)),
             x + nSize + nSpace, y + hBuff + (nSize / 2) + (i * (space + nSize)));
      }
    }

    lc++;

    for (int a = 1; a < hLayers; a++) {
      for (int i = 0; i < weights[a].rows; i++) {
        for (int j = 0; j < weights[a].cols - 1; j++) {
          stroke(weights[a].matrix[i][j] < 0 ? color(255, 0, 0) : color(0, 0, 255));
          line(x + ((lc - 1) * nSpace) + ((lc - 1) * nSize) + nSize,
               y + hBuff + (nSize / 2) + (j * (space + nSize)),
               x + (lc * nSpace) + (lc * nSize),
               y + hBuff + (nSize / 2) + (i * (space + nSize)));
        }
      }
      lc++;
    }
  }
}
