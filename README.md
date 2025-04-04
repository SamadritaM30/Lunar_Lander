# 🚀 LunarLander-v3 AI Agent (PSO + PyTorch)

A PyTorch-based AI agent that plays the classic [LunarLander-v3](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment from OpenAI Gym. The agent is trained using **Particle Swarm Optimization (PSO)** to optimize the weights of a neural network policy — achieving an average best score of **284.75** 🎯.

---

## 🧱 Overview

- ✅ Reinforcement learning to convert the 8D input to 4D output using hidden layers of 256 neurons
- ✅ Uses **Particle Swarm Optimization (PSO)** 
- ✅ Built with **PyTorch** and **Gymnasium**
- ✅ Fully self-contained, CPU-compatible training
- ✅ Achieves competitive performance with average reward ~**284.75**

---

## 📦 Requirements

```bash
pip install torch gymnasium numpy
```

---
## 🎮 Game UI

![image](https://github.com/user-attachments/assets/8e2fc20f-2ccc-4147-8d12-ea9bc9072cc3)


---
## 🏁 How to Train

Run the following command to start training using PSO:

```bash
python your_script_name.py --train
for eg: python train_agent.py --train
```

> The best policy will be saved to `best_policy_pso.npy`.

---
## 🏁 How to make the agent Play

> Run the evaluate.bat file

## 📂 Files

| File                 | Description                                           |
|----------------------|-------------------------------------------------------|
| `your_script_name.py`| Main training script (PSO + Policy Network)           |
| `best_policy_pso.npy`| Saved weights of the best policy found by PSO         |

---

## 📊 Training Output (Sample)

```
Iteration 4999/5000 - Best Reward: 284.75
Iteration 5000/5000 - Best Reward: 284.75
Best policy saved to best_policy_pso.npy with reward: 284.75
```

---

## 🧠 Model Architecture

- Input: 8 state features
- Hidden layers: Two layers with 256 neurons each
- Output: 4 action probabilities (via Softmax)

---

## 📌 Notes

- The training is **completely gradient-free**, relying on swarm intelligence to find good solutions.
- The saved policy can be used later for inference or evaluation with minor modifications.

---

## 🚀 Future Ideas

- Add a testing mode to watch the agent play
- Visualize training rewards over time
- Switch to other classic control environments

---

## 📜 License

MIT License – use freely and modify as needed.
