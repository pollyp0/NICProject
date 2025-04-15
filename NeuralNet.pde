class NeuralNet {
  
  // Структура сети
  private final int iNodes, hNodes, oNodes, hLayers;
  private Matrix[] weights;
  
  // Параметры для различных функций активации
  private static final int ACTIVATION_SIGMOID = 0;
  private static final int ACTIVATION_RELU = 1;
  private static final int ACTIVATION_LEAKY_RELU = 2;
  private int activationType = ACTIVATION_SIGMOID; // По умолчанию сигмоида
  private float leakyReluAlpha = 0.01f; // Параметр для Leaky ReLU
  
  /**
   * Конструктор нейронной сети
   */
  NeuralNet(int input, int hidden, int output, int hiddenLayers) {
    this.iNodes = input;
    this.hNodes = hidden;
    this.oNodes = output;
    this.hLayers = hiddenLayers;
    
    // Инициализация весов с оптимизированной схемой Xavier/He
    initializeWeights();
  }
  
  /**
   * Инициализирует веса с использованием Xavier/He инициализации
   * для лучшей сходимости при обучении
   */
  private void initializeWeights() {
    weights = new Matrix[hLayers + 1];
    weights[0] = new Matrix(hNodes, iNodes + 1);
    
    for(int i = 1; i < hLayers; i++) {
       weights[i] = new Matrix(hNodes, hNodes + 1); 
    }
    weights[weights.length - 1] = new Matrix(oNodes, hNodes + 1);
    
    // Xavier/He инициализация
    for(int i = 0; i < weights.length; i++) {
      int fanIn = (i == 0) ? iNodes : hNodes;
      float scaleFactor = (float) Math.sqrt(2.0 / fanIn);
      weights[i].randomize(scaleFactor);
    }
  }
  
  /**
   * Устанавливает тип функции активации
   */
  void setActivationType(int type) {
    if (type >= ACTIVATION_SIGMOID && type <= ACTIVATION_LEAKY_RELU) {
      this.activationType = type;
    }
  }
  
  /**
   * Мутация весов с заданной вероятностью
   * Добавлена возможность адаптивной мутации
   */
  void mutate(float mr) {
    mutate(mr, 1.0f);
  }
  
  /**
   * Расширенная мутация с контролем амплитуды
   * @param mr вероятность мутации
   * @param amplitude амплитуда мутации (сила изменения)
   */
  void mutate(float mr, float amplitude) {
    for(Matrix w : weights) {
      w.mutate(mr, amplitude); 
    }
  }
  
  /**
   * Применяет выбранную функцию активации к матрице
   * @param matrix входная матрица
   * @return активированная матрица
   */
  private Matrix applyActivation(Matrix matrix) {
    switch(activationType) {
      case ACTIVATION_RELU:
        return matrix.applyReLU();
      case ACTIVATION_LEAKY_RELU:
        return matrix.applyLeakyReLU(leakyReluAlpha);
      case ACTIVATION_SIGMOID:
      default:
        return matrix.activate(); // Стандартная сигмоида
    }
  }
  
  /**
   * Прямой проход через нейронную сеть
   */
  float[] output(float[] inputsArr) {
    // Проверка входных данных
    if (inputsArr.length != iNodes) {
      throw new IllegalArgumentException("Input array length doesn't match expected input nodes");
    }
    
    Matrix inputs = weights[0].singleColumnMatrixFromArray(inputsArr);
    Matrix curr_bias = inputs.addBias();
    
    // Прохождение через скрытые слои
    for(int i = 0; i < hLayers; i++) {
      Matrix hidden_ip = weights[i].dot(curr_bias); 
      Matrix hidden_op = applyActivation(hidden_ip);
      curr_bias = hidden_op.addBias();
    }
    
    // Выходной слой
    Matrix output_ip = weights[weights.length - 1].dot(curr_bias);
    Matrix output = applyActivation(output_ip);
    
    // Применение softmax для выходных значений, чтобы получить распределение вероятностей
    return softmax(output.toArray());
  }
  
  /**
   * Применяет функцию softmax к выходным данным для получения вероятностей
   */
  private float[] softmax(float[] output) {
    float sum = 0;
    float[] softmaxOutput = new float[output.length];
    
    // Находим максимальное значение для численной стабильности
    float max = output[0];
    for(int i = 1; i < output.length; i++) {
      if(output[i] > max) max = output[i];
    }
    
    // Вычисляем exp(x - max) для каждого выхода и суммируем
    for(int i = 0; i < output.length; i++) {
      softmaxOutput[i] = (float)Math.exp(output[i] - max);
      sum += softmaxOutput[i];
    }
    
    // Нормализуем, чтобы получить вероятности
    for(int i = 0; i < softmaxOutput.length; i++) {
      softmaxOutput[i] /= sum;
    }
    
    return softmaxOutput;
  }
  
  /**
   * Создает потомка путем скрещивания с другой сетью
   * Улучшенный алгоритм кроссовера с элементами интерполяции
   */
  NeuralNet crossover(NeuralNet partner) {
    NeuralNet child = new NeuralNet(iNodes, hNodes, oNodes, hLayers);
    child.setActivationType(this.activationType);
    
    for(int i = 0; i < weights.length; i++) {
      // Используем продвинутый кроссовер с интерполяцией
      child.weights[i] = weights[i].crossover(partner.weights[i], 0.5f);
    }
    
    return child;
  }
  
  /**
   * Улучшенный кроссовер с параметром смешивания
   * @param partner партнер для скрещивания
   * @param mixRate коэффициент смешивания (0.0 - полностью первый родитель, 
   *                1.0 - полностью второй родитель)
   */
  NeuralNet crossover(NeuralNet partner, float mixRate) {
    NeuralNet child = new NeuralNet(iNodes, hNodes, oNodes, hLayers);
    child.setActivationType(this.activationType);
    
    for(int i = 0; i < weights.length; i++) {
      child.weights[i] = weights[i].crossover(partner.weights[i], mixRate);
    }
    
    return child;
  }
  
  /**
   * Создает полную копию нейронной сети
   */
  NeuralNet clone() {
    NeuralNet clone = new NeuralNet(iNodes, hNodes, oNodes, hLayers);
    clone.setActivationType(this.activationType);
    
    for(int i = 0; i < weights.length; i++) {
      clone.weights[i] = weights[i].clone(); 
    }
    
    return clone;
  }
  
  /**
   * Загружает веса из существующей модели
   */
  void load(Matrix[] weight) {
    if (weight.length != weights.length) {
      throw new IllegalArgumentException("Weight arrays don't match in size");
    }
      
    for(int i = 0; i < weights.length; i++) {
      weights[i] = weight[i]; 
    }
  }
  
  /**
   * Возвращает копию весов сети
   */
  Matrix[] pull() {
    Matrix[] model = new Matrix[weights.length];
    for(int i = 0; i < weights.length; i++) {
      model[i] = weights[i].clone();
    }
    return model;
  }
  
  /**
   * Сохраняет модель в файл
   */
  void saveToFile(String filename) {
    // Код для сохранения модели в файл
    // Зависит от конкретной реализации и библиотек
  }
  
  /**
   * Загружает модель из файла
   */
  static NeuralNet loadFromFile(String filename) {
    // Код для загрузки модели из файла
    // Зависит от конкретной реализации и библиотек
    return null;
  }
  
 /**
 * Визуализирует нейронную сеть
 */
void show(float x, float y, float w, float h, float[] vision, float[] decision) {

  float space = 5;

  float nSize = (h - (space * (iNodes - 2))) / iNodes;

  float nSpace = (w - (weights.length * nSize)) / weights.length;

  float hBuff = (h - (space * (hNodes - 1)) - (nSize * hNodes)) / 2;

  float oBuff = (h - (space * (oNodes - 1)) - (nSize * oNodes)) / 2;

   

  // Находим индекс максимального решения

  int maxIndex = 0;

  for(int i = 1; i < decision.length; i++) {

    if(decision[i] > decision[maxIndex]) {

      maxIndex = i; 

    }

  }

   

  int lc = 0;  // Layer Count

   

  // ОТРИСОВКА УЗЛОВ

  // Рисуем входные узлы

  for(int i = 0; i < iNodes; i++) {

    if(vision[i] != 0) {

      fill(0, 255, 0);

    } else {

      fill(255); 

    }

    stroke(0);

    ellipseMode(CORNER);

    ellipse(x, y + (i * (nSize + space)), nSize, nSize);

    textSize(nSize / 2);

    textAlign(CENTER, CENTER);

    fill(0);

    text(i, x + (nSize / 2), y + (nSize / 2) + (i * (nSize + space)));

  }

  

  lc++;

  

  // Рисуем скрытые узлы

  for(int a = 0; a < hLayers; a++) {

    for(int i = 0; i < hNodes; i++) {

      fill(255);

      stroke(0);

      ellipseMode(CORNER);

      ellipse(x + (lc * nSize) + (lc * nSpace), y + hBuff + (i * (nSize + space)), nSize, nSize);

    }

    lc++;

  }

  

  // Рисуем выходные узлы

  for(int i = 0; i < oNodes; i++) {

    if(i == maxIndex) {

      fill(0, 255, 0);

    } else {

      fill(255); 

    }

    stroke(0);

    ellipseMode(CORNER);

    ellipse(x + (lc * nSpace) + (lc * nSize), y + oBuff + (i * (nSize + space)), nSize, nSize);

    

    // Добавляем процент уверенности

    fill(0);

    textSize(nSize / 3);

    textAlign(CENTER, CENTER);

    text(nf(decision[i] * 100, 0, 0) + "%", 

         x + (lc * nSpace) + (lc * nSize) + nSize/2, 

         y + oBuff + (nSize/2) + (i * (nSize + space)));

  }

  

  lc = 1;

  

  // ОТРИСОВКА ВЕСОВ

  // Веса от входного к первому скрытому слою

  for(int i = 0; i < weights[0].rows; i++) {

    for(int j = 0; j < weights[0].cols - 1; j++) {

      float weight = weights[0].matrix[i][j];

      float absWeight = abs(weight);

      

      // Меняем толщину линии в зависимости от силы веса

      strokeWeight(map(absWeight, 0, 1, 0.5, 2.5));

      

      // Прозрачность также зависит от силы веса

      int alpha = (int)map(absWeight, 0, 1, 50, 200);

      

      if(weight < 0) {

        stroke(255, 0, 0, alpha); // красный для отрицательных

      } else {

        stroke(0, 0, 255, alpha); // синий для положительных

      }

      

      line(x + nSize, 

           y + (nSize / 2) + (j * (space + nSize)),

           x + nSize + nSpace, 

           y + hBuff + (nSize / 2) + (i * (space + nSize)));

    }

  }

  

  lc++;

  

  // Веса между скрытыми слоями

  for(int a = 1; a < hLayers; a++) {

    for(int i = 0; i < weights[a].rows; i++) {

      for(int j = 0; j < weights[a].cols - 1; j++) {

        float weight = weights[a].matrix[i][j];

        float absWeight = abs(weight);

        

        strokeWeight(map(absWeight, 0, 1, 0.5, 2.5));

        int alpha = (int)map(absWeight, 0, 1, 50, 200);

        

        if(weight < 0) {

          stroke(255, 0, 0, alpha); 

        } else {

          stroke(0, 0, 255, alpha); 

        }

        

        line(x + (lc * nSize) + ((lc - 1) * nSpace),

             y + hBuff + (nSize / 2) + (j * (space + nSize)),

             x + (lc * nSize) + (lc * nSpace),

             y + hBuff + (nSize / 2) + (i * (space + nSize)));

      }

    }

    lc++;

  }

  

  // Веса от последнего скрытого слоя к выходному

  for(int i = 0; i < weights[weights.length - 1].rows; i++) {

    for(int j = 0; j < weights[weights.length - 1].cols - 1; j++) {

      float weight = weights[weights.length - 1].matrix[i][j];

      float absWeight = abs(weight);

      

      strokeWeight(map(absWeight, 0, 1, 0.5, 2.5));

      int alpha = (int)map(absWeight, 0, 1, 50, 200);

      

      if(weight < 0) {

        stroke(255, 0, 0, alpha); 

      } else {

        stroke(0, 0, 255, alpha); 

      }

      

      line(x + (lc * nSize) + ((lc - 1) * nSpace),

           y + hBuff + (nSize / 2) + (j * (space + nSize)),

           x + (lc * nSize) + (lc * nSpace),

           y + oBuff + (nSize / 2) + (i * (space + nSpace)));

    }

  }

  

  // Сбрасываем толщину линии

  strokeWeight(1);

  

  // Добавляем подписи для выходных узлов

  fill(0);

  textSize(15);

  textAlign(CENTER, CENTER);

  text("U", x + (lc * nSize) + (lc * nSpace) + nSize / 2, y + oBuff + (nSize / 2));

  text("D", x + (lc * nSize) + (lc * nSpace) + nSize / 2, y + oBuff + space + nSize + (nSize / 2));

  text("L", x + (lc * nSize) + (lc * nSpace) + nSize / 2, y + oBuff + (2 * space) + (2 * nSize) + (nSize / 2));

  text("R", x + (lc * nSize) + (lc * nSpace) + nSize / 2, y + oBuff + (3 * space) + (3 * nSize) + (nSize / 2));

}
