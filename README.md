# Reinforce Learning
This repo use [gymnasium](https://www.gymlibrary.dev/) to implement basic reinforcement learning applications

## Simple applications using Q-learning algoritm

### 1. Frozen Lake ([code](q_learning_apps/frozen_lake.py))
Successfully solved 8X8 [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) in slippery mode. The task involves guiding the character from the top-left starting position to the bottom-right goal without falling into any holes. Due to the slippery ice, there’s only a 1/3 chance of moving in the intended direction, with a 2/3 chance of sliding in one of the perpendicular directions.

https://github.com/user-attachments/assets/e00c8169-d24f-4b91-b14a-9f912972ff0a

### 2. Play Black Jack ([code](q_learning_apps/blackjack.py))
Attempted to improve the win rate in [blackjack](https://www.gymlibrary.dev/environments/toy_text/blackjack/) using a simple Q-learning algorithm. Through basic training, the win rate reached approximately 45%.

https://github.com/user-attachments/assets/f72fc4e4-22a4-421b-85e2-b537d1cb7386

