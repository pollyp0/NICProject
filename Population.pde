class Population {
    Snake[] snakes;
    Snake bestSnake; // Лучшая змея в популяции
    int bestSnakeScore = 0; // Лучший счет среди змей
    int gen = 0; // Текущее поколение
    float bestFitness = 0; // Лучшая приспособленность
    float fitnessSum = 0; // Сумма приспособленностей всех змей

    Population(int size) {
        snakes = new Snake[size];
        for (int i = 0; i < size; i++) {
            snakes[i] = new Snake(); // Инициализация каждой змеи
        }
        bestSnake = snakes[0].clone(); // Клонируем первую змею как лучшую
        bestSnake.replay = true; // Устанавливаем флаг для воспроизведения
    }

    // Проверяем, завершили ли все змеи свою жизнь
    boolean done() {
        for (Snake snake : snakes) {
            if (!snake.dead) return false; // Если хотя бы одна змея жива, возвращаем false
        }
        return bestSnake.dead; // Возвращаем true, если лучшая змея также мертва
    }

    // Обновляем состояние всех змей
    void update() {
        if (!bestSnake.dead) {
            bestSnake.look();
            bestSnake.think();
            bestSnake.move(); // Обновляем лучшую змею, если она жива
        }
        for (Snake snake : snakes) {
            if (!snake.dead) {
                snake.look();
                snake.think();
                snake.move(); // Обновляем каждую живую змею
            }
        }
    }

    // Отображаем змей на экране
    void show() {
        if (replayBest) {
            bestSnake.show();
            bestSnake.brain.show(0, 0, 360, 790, bestSnake.vision, bestSnake.decision); // Показываем мозг лучшей змеи
        } else {
            for (Snake snake : snakes) {
                snake.show(); // Показываем всех змей
            }
        }
    }

    // Устанавливаем лучшую змею в поколении
    void setBestSnake() {
        float maxFitness = 0;
        int maxIndex = 0;
        for (int i = 0; i < snakes.length; i++) {
            if (snakes[i].fitness > maxFitness) {
                maxFitness = snakes[i].fitness;
                maxIndex = i;
            }
        }
        if (maxFitness > bestFitness) {
            bestFitness = maxFitness;
            bestSnake = snakes[maxIndex].cloneForReplay(); // Клонируем лучшую змею для воспроизведения
            bestSnakeScore = snakes[maxIndex].score;
        } else {
            bestSnake = bestSnake.cloneForReplay(); // Если лучшая змея не изменилась, клонируем её снова
        }
    }

    // Выбираем родителя для скрещивания на основе приспособленности
    Snake selectParent() {
        float rand = random(fitnessSum);
        float summation = 0;
        for (Snake snake : snakes) {
            summation += snake.fitness;
            if (summation > rand) {
                return snake; // Возвращаем змею, если её приспособленность попадает в диапазон
            }
        }
        return snakes[0]; // Возвращаем первую змею, если ничего не найдено
    }

    // Процесс естественного отбора
    void naturalSelection() {
        Snake[] newSnakes = new Snake[snakes.length];

        setBestSnake();
        calculateFitnessSum();

        newSnakes[0] = bestSnake.clone(); // Добавляем лучшую змею в новое поколение
        for (int i = 1; i < snakes.length; i++) {
            Snake child = selectParent().crossover(selectParent()); // Скрещиваем двух родителей
            child.mutate(); // Мутируем потомка
            newSnakes[i] = child; // Добавляем потомка в новое поколение
        }
        snakes = newSnakes.clone(); // Обновляем популяцию
        evolution.add(bestSnakeScore); // Добавляем лучший счет в историю эволюции
        gen++; // Увеличиваем номер поколения
    }

    // Мутируем всех змей, кроме лучшей
    void mutate() {
        for (int i = 1; i < snakes.length; i++) {
            snakes[i].mutate(); // Мутируем каждую змею, начиная со второй
        }
    }

    // Вычисляем приспособленность для каждой змеи
    void calculateFitness() {
        for (Snake snake : snakes) {
            snake.calculateFitness();
        }
    }

    // Вычисляем сумму приспособленностей всех змей
    void calculateFitnessSum() {
        fitnessSum = 0;
        for (Snake snake : snakes) {
            fitnessSum += snake.fitness;
        }
    }
}
