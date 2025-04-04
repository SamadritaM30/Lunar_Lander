import gymnasium as gym  # Importing Gymnasium to create a reinforcement learning environment
import numpy as np       # Importing NumPy for numerical operations (arrays, randomness, etc.)
import torch             # Importing PyTorch for deep learning computations
import torch.nn as nn    # Importing PyTorch's neural network module
import argparse          # Importing argparse to handle command-line arguments

# Define the Policy Network (Neural Network that decides the agent’s actions)
class PolicyNetwork(nn.Module):  # Inherits from PyTorch's base neural network class (nn.Module)
    def __init__(self, input_dim=8, output_dim=4):  
        super(PolicyNetwork, self).__init__()  # Calls the parent class constructor
        hidden_size = 256  # Defines the number of neurons in hidden layers

        # The neural network structure: Input layer -> Hidden layers -> Output layer
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),  # Input layer: 8 state values → 256 neurons
            nn.ReLU(),  # Activation function to introduce non-linearity
            nn.Linear(hidden_size, hidden_size),  # Second layer: 256 neurons → 256 neurons
            nn.ReLU(),  # Another activation function
            nn.Linear(hidden_size, output_dim),  # Output layer: 256 neurons → 4 action probabilities
            nn.Softmax(dim=-1)  # Converts output into probabilities (ensuring they sum to 1)
        )

    def forward(self, x):  # Defines the forward pass (how input data flows through the network)
        return self.actor(x)  # Pass input `x` through the neural network and return action probabilities

# PSO (Particle Swarm Optimization) Hyperparameters
NUM_PARTICLES = 30  # Number of particles (solutions) in the swarm
MAX_ITERS = 5000  # Maximum number of training iterations
INERTIA_WEIGHT = 0.7  # Controls how much a particle retains its previous velocity
COGNITIVE = 1.5  # Controls how much a particle trusts its personal best position
SOCIAL = 1.5  # Controls how much a particle trusts the global best solution found by others

# Function to evaluate a given policy (weights) on the environment
def evaluate_policy(weights, env_name="LunarLander-v3", episodes=5):
    env = gym.make(env_name)  # Create the Gym environment
    model = PolicyNetwork()  # Create an instance of the policy network (neural network)
    set_weights(model, weights)  # Load the given weights into the neural network
    model.eval()  # Set the model to evaluation mode (prevents unnecessary gradient calculations)

    total_reward = 0  # Initialize total reward counter

    for _ in range(episodes):  # Run multiple episodes for evaluation
        state, _ = env.reset()  # Reset the environment to start a new episode
        done = False  # Track whether the episode has ended
        episode_reward = 0  # Initialize reward for the current episode

        while not done:  # Loop until the episode ends
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state into a PyTorch tensor (batch format)
            
            with torch.no_grad():  # Disable gradient tracking (since we're just evaluating)
                action_probs = model(state_tensor)  # Get action probabilities from the model
            
            action = torch.argmax(action_probs, dim=-1).item()  # Select the action with the highest probability
            
            state, reward, terminated, truncated, _ = env.step(action)  # Apply action to the environment
            episode_reward += reward  # Add received reward to the episode's total
            done = terminated or truncated  # Check if the episode is over

        total_reward += episode_reward  # Add episode reward to total reward

    env.close()  # Close the environment
    return total_reward / episodes  # Return the average reward over all test episodes

# Function to set model weights (converts a flat array of weights into the PyTorch model format)
def set_weights(model, flat_weights):
    idx = 0  # Initialize an index for slicing the flat_weights array

    with torch.no_grad():  # Disable gradient tracking (since we're setting values, not training)
        for param in model.parameters():  # Loop over model parameters
            num_params = param.numel()  # Get the total number of elements in the parameter
            param.copy_(torch.tensor(flat_weights[idx:idx + num_params]).view_as(param))  # Assign values
            idx += num_params  # Move to the next segment of the weight array

# Function to get all model weights as a flat NumPy array (used for PSO)
def get_flat_weights(model):
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

# Function to train the policy using Particle Swarm Optimization (PSO)
def train_pso(env_name="LunarLander-v3", save_path="best_policy_pso.npy"):
    env = gym.make(env_name)  # Create the Gym environment
    model = PolicyNetwork()  # Create an instance of the policy network
    num_params = sum(p.numel() for p in model.parameters())  # Count total number of parameters in the model

    # Initialize particle positions (solutions) randomly in the range [-1,1]
    particles = np.random.uniform(-1, 1, (NUM_PARTICLES, num_params))
    velocities = np.zeros((NUM_PARTICLES, num_params))  # Initialize velocities to zero
    
    # Initialize each particle’s personal best solution and scores
    personal_best_positions = particles.copy()
    personal_best_scores = np.full(NUM_PARTICLES, -np.inf)  # Set initial scores to negative infinity
    
    global_best_position = None  # Track the best solution found by any particle
    global_best_score = -np.inf  # Initialize the best score as negative infinity

    # Main training loop
    for iteration in range(MAX_ITERS):
        for i in range(NUM_PARTICLES):  # Loop through all particles
            score = evaluate_policy(particles[i])  # Evaluate the current particle's policy
            
            # Update personal best position if this score is better
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]
            
            # Update global best position if this score is better than any previous score
            if score > global_best_score:
                global_best_score = score
                global_best_position = particles[i]

        # Update each particle's velocity and position
        for i in range(NUM_PARTICLES):
            r1, r2 = np.random.rand(num_params), np.random.rand(num_params)  # Generate random factors
            
            velocities[i] = (INERTIA_WEIGHT * velocities[i] +  # Inertia component
                             COGNITIVE * r1 * (personal_best_positions[i] - particles[i]) +  # Personal influence
                             SOCIAL * r2 * (global_best_position - particles[i]))  # Social influence
            
            particles[i] += velocities[i]  # Update position using new velocity

        print(f"Iteration {iteration + 1}/{MAX_ITERS} - Best Reward: {global_best_score:.2f}")

    # Save the best found policy (weights)
    np.save(save_path, global_best_position)
    print(f"Best policy saved to {save_path} with reward: {global_best_score:.2f}")

# Entry point of the script (handles command-line arguments)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Create argument parser
    parser.add_argument("--train", action="store_true")  # Add `--train` flag to start training
    parser.add_argument("--filename", type=str, default="best_policy_pso.npy")  # Specify filename for saving weights
    args = parser.parse_args()  # Parse command-line arguments

    if args.train:
        train_pso(save_path=args.filename)  # Train using PSO if `--train` is specified
    else:
        print("Specify --train to train the agent.")  # Print message if no flag is given
