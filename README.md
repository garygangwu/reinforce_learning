# Reinforce Learning
This repo use [gymnasium](https://www.gymlibrary.dev/) to implement basic reinforcement learning applications

## Simple applications using Q-learning algoritm

### 1. Frozen Lake ([code](q_learning_apps/frozen_lake.py))
Successfully solved 8X8 [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) in slippery mode. The task involves guiding the character from the top-left starting position to the bottom-right goal without falling into any holes. Due to the slippery ice, there’s only a 1/3 chance of moving in the intended direction, with a 2/3 chance of sliding in one of the perpendicular directions.

https://github.com/user-attachments/assets/e00c8169-d24f-4b91-b14a-9f912972ff0a

### 2. Play Blackjack ([code](q_learning_apps/blackjack.py))
Attempted to improve the win rate in [blackjack](https://www.gymlibrary.dev/environments/toy_text/blackjack/) using a simple Q-learning algorithm. Through basic training, the win rate reached approximately 45%.

https://github.com/user-attachments/assets/f72fc4e4-22a4-421b-85e2-b537d1cb7386

### 3. Mountain Car ([code](q_learning_apps/mountain_car.py))
The [Mountain Car](https://www.gymlibrary.dev/environments/classic_control/mountain_car/) allows for three discrete deterministic actions: accelerating left, accelerating right, and no acceleration. The goal is to reach the flag at the top of the right hill as quickly as possible by leveraging both the car’s acceleration and gravity. To solve this problem, the continuous observation space needs to be divided into multiple discrete buckets, enabling the Q-learning algorithm to train the weights on a finite set of inputs.

### 4. Mountain Car Continuous ([code](q_learning_apps/mountain_car_continuous.py))
In this version of [Mountain Car Environment](https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous/), the car can take continuous actions within the range of [-1, 1], representing the directional force applied to the car. Therefore, in addition to discretizing the continuous observation space, the action space must also be divided into discrete buckets for the Q-learning algorithm to train on.

One challenge with this problem is that Q-learning can sometimes lead to local infinite loops, where the highest-weighted options are not the best globally. To address this, instead of always selecting the action with the highest weight, the modified Q-learning algorithm introduces exploration by occasionally choosing sub-optimal actions, such as those with the 2nd or 3rd highest weights. With this adjustment, the car consistently reaches the final destination.

https://github.com/user-attachments/assets/d8b3d989-fa9d-435b-b6d3-e48479ea74a7

### 5. Cart Pole ([code](q_learning_apps/cart_pole.py))
The [Cart Pole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) tries balancing a pole on a cart by applying forces to move the cart left or right.

https://github.com/user-attachments/assets/3fa3d5b4-70cb-42a9-a683-95079f43b8ce

.
