from agent import Agent
from game import Game
import torch

if __name__ == "__main__":
    game_env = Game()
    agent = Agent()

    best_model = agent.train(game_env)
    torch.save(best_model.state_dict(), 'best_model.pth')
    print("Best model saved to 'best_model.pth'")

