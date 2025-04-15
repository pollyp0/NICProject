class Snake {
  
  int score = 1;
  // [MODIFIED] Уменьшено начальное время жизни для стимулирования активности
  int lifeLeft = 100;
  int lifetime = 0;
  int xVel, yVel;
  int foodItterate = 0;

  float fitness = 0;
  
  boolean dead = false;
  boolean replay = false;

  float[] vision;
  float[] decision;

  PVector head;

  ArrayList<PVector> body;
  ArrayList<Food> foodList;

  Food food;
  NeuralNet brain;

  Snake() {
    this(hidden_layers);
  }

  Snake(int layers) {
    head = new PVector(800, height / 2);
    food = new Food();
    body = new ArrayList<PVector>();
    if (!humanPlaying) {
      vision = new float[24];
      decision = new float[4];
      foodList = new ArrayList<Food>();
      foodList.add(food.clone());
      brain = new NeuralNet(24, hidden_nodes, 4, layers);
      body.add(new PVector(800, (height / 2) + SIZE));
      body.add(new PVector(800, (height / 2) + (2 * SIZE)));
      score += 2;
    }
  }

  Snake(ArrayList<Food> foods) {
    replay = true;
    vision = new float[24];
    decision = new float[4];
    body = new ArrayList<PVector>();
    foodList = new ArrayList<Food>(foods.size());
    for (Food f : foods) {
      foodList.add(f.clone());
    }
    food = foodList.get(foodItterate);
    foodItterate++;
    head = new PVector(800, height / 2);
    body.add(new PVector(800, (height / 2) + SIZE));
    body.add(new PVector(800, (height / 2) + (2 * SIZE)));
    score += 2;
  }

  boolean bodyCollide(float x, float y) {
    for (int i = 0; i < body.size(); i++) {
      if (x == body.get(i).x && y == body.get(i).y) {
        return true;
      }
    }
    return false;
  }

  boolean foodCollide(float x, float y) {
    return x == food.pos.x && y == food.pos.y;
  }

  boolean wallCollide(float x, float y) {
    return x >= width - SIZE || x < 400 + SIZE || y >= height - SIZE || y < SIZE;
  }

  void show() {
    food.show();
    fill(255);
    stroke(0);
    for (int i = 0; i < body.size(); i++) {
      rect(body.get(i).x, body.get(i).y, SIZE, SIZE);
    }
    fill(dead ? 150 : 255);
    rect(head.x, head.y, SIZE, SIZE);
  }

  void move() {
    if (!dead) {
      if (!humanPlaying && !modelLoaded) {
        lifetime++;
        lifeLeft--;
      }
      if (foodCollide(head.x, head.y)) {
        eat();
      }
      shiftBody();
      if (wallCollide(head.x, head.y) || bodyCollide(head.x, head.y) || (lifeLeft <= 0 && !humanPlaying)) {
        dead = true;
      }
    }
  }

  void eat() {
    int len = body.size() - 1;
    score++;
    // [MODIFIED] Новая логика увеличения lifeLeft после еды
    if (!humanPlaying && !modelLoaded) {
      lifeLeft += 80;
      if (lifeLeft > 400) lifeLeft = 400;
    }
    if (len >= 0) {
      body.add(new PVector(body.get(len).x, body.get(len).y));
    } else {
      body.add(new PVector(head.x, head.y));
    }
    if (!replay) {
      food = new Food();
      while (bodyCollide(food.pos.x, food.pos.y)) {
        food = new Food();
      }
      if (!humanPlaying) {
        foodList.add(food);
      }
    } else {
      food = foodList.get(foodItterate);
      foodItterate++;
    }
  }

  void shiftBody() {
    float tempx = head.x;
    float tempy = head.y;
    head.x += xVel;
    head.y += yVel;
    float temp2x, temp2y;
    for (int i = 0; i < body.size(); i++) {
      temp2x = body.get(i).x;
      temp2y = body.get(i).y;
      body.get(i).x = tempx;
      body.get(i).y = tempy;
      tempx = temp2x;
      tempy = temp2y;
    }
  }

  Snake cloneForReplay() {
    Snake clone = new Snake(foodList);
    clone.brain = brain.clone();
    return clone;
  }

  Snake clone() {
    Snake clone = new Snake(hidden_layers);
    clone.brain = brain.clone();
    return clone;
  }

  Snake crossover(Snake parent) {
    Snake child = new Snake(hidden_layers);
    child.brain = brain.crossover(parent.brain);
    return child;
  }

  void mutate() {
    brain.mutate(mutationRate);
  }

  void calculateFitness() {
    // [MODIFIED] Новая формула: поощряет сбор еды, активность, штрафует пассивность
    fitness = score * 1000 + lifetime * 5 - (200 - lifeLeft) * 2;
  }

  void look() {
    vision = new float[24];
    float[] temp;
    PVector[] directions = {
      new PVector(-SIZE, 0), new PVector(-SIZE, -SIZE), new PVector(0, -SIZE),
      new PVector(SIZE, -SIZE), new PVector(SIZE, 0), new PVector(SIZE, SIZE),
      new PVector(0, SIZE), new PVector(-SIZE, SIZE)
    };
    for (int i = 0; i < directions.length; i++) {
      temp = lookInDirection(directions[i]);
      vision[i * 3] = temp[0];
      vision[i * 3 + 1] = temp[1];
      vision[i * 3 + 2] = temp[2];
    }
  }

  float[] lookInDirection(PVector direction) {
    float[] look = new float[3];
    PVector pos = new PVector(head.x, head.y);
    float distance = 0;
    boolean foodFound = false, bodyFound = false;
    pos.add(direction);
    distance += 1;
    while (!wallCollide(pos.x, pos.y)) {
      if (!foodFound && foodCollide(pos.x, pos.y)) {
        foodFound = true;
        look[0] = 1;
      }
      if (!bodyFound && bodyCollide(pos.x, pos.y)) {
        bodyFound = true;
        look[1] = 1;
      }
      if (replay && seeVision) {
        stroke(0, 255, 0);
        point(pos.x, pos.y);
        if (foodFound) {
          noStroke();
          fill(255, 255, 51);
          ellipse(pos.x, pos.y, 5, 5);
        }
        if (bodyFound) {
          noStroke();
          fill(102, 0, 102);
          ellipse(pos.x, pos.y, 5, 5);
        }
      }
      pos.add(direction);
      distance += 1;
    }
    if (replay && seeVision) {
      noStroke();
      fill(0, 255, 0);
      ellipse(pos.x, pos.y, 5, 5);
    }
    look[2] = 1 / distance;
    return look;
  }

  void think() {
    decision = brain.output(vision);
    // [MODIFIED] Инициализация max с NEGATIVE_INFINITY для корректной работы с любыми значениями
    int maxIndex = 0;
    float max = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < decision.length; i++) {
      if (decision[i] > max) {
        max = decision[i];
        maxIndex = i;
      }
    }

    switch (maxIndex) {
      case 0: moveUp(); break;
      case 1: moveDown(); break;
      case 2: moveLeft(); break;
      case 3: moveRight(); break;
    }
  }

  void moveUp() {
    if (yVel != SIZE) {
      xVel = 0;
      yVel = -SIZE;
    }
  }

  void moveDown() {
    if (yVel != -SIZE) {
      xVel = 0;
      yVel = SIZE;
    }
  }

  void moveLeft() {
    if (xVel != SIZE) {
      xVel = -SIZE;
      yVel = 0;
    }
  }

  void moveRight() {
    if (xVel != -SIZE) {
      xVel = SIZE;
      yVel = 0;
    }
  }
}
