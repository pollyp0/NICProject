class Population {
   
   Snake[] snakes;
   Snake bestSnake;
   int bestSnakeScore = 0;
   int gen = 0;
   int samebest = 0;
   float bestFitness = 0;
   float fitnessSum = 0;
   
   Population(int size) {
      snakes = new Snake[size];
      for(int i = 0; i < snakes.length; i++) {
         snakes[i] = new Snake();
      }
      bestSnake = snakes[0].clone();
      bestSnake.replay = true;
   }
   
   boolean done() {
      for(int i = 0; i < snakes.length; i++) {
         if(!snakes[i].dead) return false;
      }
      if(!bestSnake.dead) {
         return false;
      }
      return true;
   }
   
   void update() {
      if(!bestSnake.dead) {
         bestSnake.look();
         bestSnake.think();
         bestSnake.move();
      }
      for(int i = 0; i < snakes.length; i++) {
         if(!snakes[i].dead) {
            snakes[i].look();
            snakes[i].think();
            snakes[i].move();
         }
      }
   }
   
   void show() {
      if(replayBest) {
        bestSnake.show();
        bestSnake.brain.show(0,0,360,790, bestSnake.vision, bestSnake.decision);
      } else {
         for(int i = 0; i < snakes.length; i++) {
            snakes[i].show();
         }
      }
   }
   
   void setBestSnake() {
       float max = 0;
       int maxIndex = 0;
       for(int i = 0; i < snakes.length; i++) {
          if(snakes[i].fitness > max) {
             max = snakes[i].fitness;
             maxIndex = i;
          }
       }
       if(max > bestFitness) {
         bestFitness = max;
         bestSnake = snakes[maxIndex].cloneForReplay();
         bestSnakeScore = snakes[maxIndex].score;
       } else {
         bestSnake = bestSnake.cloneForReplay();
       }
   }
   
   Snake getSecondBestSnake(){
     float secondMax = 0;
     Snake second = null;
     float theBest = bestSnake.fitness;
     for(int i = 0; i < snakes.length; i++){
       if(snakes[i].fitness > secondMax && snakes[i].fitness < theBest){
         secondMax = snakes[i].fitness;
         second = snakes[i];
       }
     }
     return second;
   }
   
   Snake selectParent() {
      float randVal = random(fitnessSum);
      float summation = 0;
      for(int i = 0; i < snakes.length; i++) {
         summation += snakes[i].fitness;
         if(summation > randVal) {
           return snakes[i];
         }
      }
      return snakes[0];
   }
   
   void naturalSelection() {
      Snake[] newSnakes = new Snake[snakes.length];
      setBestSnake();
      calculateFitnessSum();
      newSnakes[0] = bestSnake.clone();
      Snake secondBest = getSecondBestSnake();
      if(secondBest != null){
        newSnakes[1] = secondBest.clone();
      } else {
        newSnakes[1] = bestSnake.clone();
      }
      for(int i = 2; i < snakes.length; i++) {
         Snake child = selectParent().crossover(selectParent());
         child.mutate();
         newSnakes[i] = child;
      }
      snakes = newSnakes.clone();
      evolution.add(bestSnakeScore);
      gen++;
   }
   
   void mutate() {
       for(int i = 1; i < snakes.length; i++) {
          snakes[i].mutate();
       }
   }
   
   void calculateFitness() {
      for(int i = 0; i < snakes.length; i++) {
         snakes[i].calculateFitness();
      }
   }
   
   void calculateFitnessSum() {
       fitnessSum = 0;
       for(int i = 0; i < snakes.length; i++) {
         fitnessSum += snakes[i].fitness;
       }
   }
}
