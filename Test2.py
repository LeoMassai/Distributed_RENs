import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from Models import REN, RNNModel
import numpy as np


def line_circle_intersection_penalty(point1, point2, obstacle_position, obstacle_radius, penalty):
    """
    Compute the penalty based on whether the line between two points intersects a circular obstacle.

    Args:
    - point1 (torch.Tensor): Tensor of shape (2,) representing the first point.
    - point2 (torch.Tensor): Tensor of shape (2,) representing the second point.
    - obstacle_position (torch.Tensor): Tensor of shape (2,) representing the position of the obstacle.
    - obstacle_radius (float): Radius of the obstacle.
    - penalty (float): Penalty coefficient when the line intersects the obstacle.

    Returns:
    - penalty (float): The penalty value.
    """
    # Compute the vector from point1 to point2
    line_vector = point2 - point1

    # Compute the vector from point1 to the obstacle position
    to_obstacle_vector = obstacle_position - point1

    # Compute the orthogonal projection of the to_obstacle_vector onto the line_vector
    projection = torch.dot(to_obstacle_vector, line_vector) / torch.dot(line_vector, line_vector)
    projection = torch.clamp(projection, 0, 1)
    closest_point_on_line = point1 + projection * line_vector

    # Compute the distance between the closest point on the line and the obstacle position
    distance = torch.norm(obstacle_position - closest_point_on_line)

    # Penalize if the distance is less than the obstacle radius
    penalty = penalty * torch.relu(obstacle_radius - distance)

    return penalty


class Robot:
    def __init__(self, initial_pos, obstacle_position, obstacle_radius, final_position):
        self.position = initial_pos  # Initial position [x, y]
        self.obstacle_position = obstacle_position
        self.obstacle_radius = obstacle_radius
        self.final_position = final_position
        self.history = [self.position.clone()]  # History of robot's positions

    def update_state(self, velocity_x, velocity_y):
        # Update position based on velocity components
        self.position[0] = self.position[0] + velocity_x
        self.position[1] = self.position[1] + velocity_y
        self.history.append(self.position.clone())  # Record new position in history

    def check_collision(self):
        # Check if robot collides with obstacle
        distance_to_obstacle = torch.norm(self.position - self.obstacle_position)
        return distance_to_obstacle <= self.obstacle_radius


# Define obstacle position and radius
obstacle_position = torch.tensor([5, 5])  # Example obstacle position [x, y]
obstacle_radius = 3.0
target = torch.tensor([10.6, 8.3])
# Create and run the robot with velocity control in 2D space
posi = torch.tensor([2.37, 3.34])
robot = Robot(posi, obstacle_position, obstacle_radius, target)

# Control horizon
T = 6
n = 8
m = 2
p = 2
l = 4

system = 'REN'

if system == 'REN':
    RENsys = REN(m, p, n, l)
else:
    RENsys = RNNModel(m, n, l, p)

model_parameters = filter(lambda p: p.requires_grad, RENsys.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

epochs = 3000
lossp = torch.zeros(epochs)
# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

uf = torch.zeros((2, T))
dist = nn.MSELoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    final_position_penalty = 0
    obstacle_penalty = 0
    u = torch.zeros(2)
    xi = torch.randn(n)
    pos = posi
    for t in range(T):
        poso = pos
        u, xi = RENsys(pos, xi, t)
        pos = pos + u
        uf[:, t] = u
        #distance_to_obstacle = torch.norm(pos - robot.obstacle_position)
        obstacle_penalty = obstacle_penalty + line_circle_intersection_penalty(poso, pos, obstacle_position,
                                                                               obstacle_radius, 1e12)
        final_position_penalty = final_position_penalty + dist(pos, target)
    loss = 1e2 * final_position_penalty + 0.1 * obstacle_penalty + 1e10 * dist(pos, target)
    loss.backward()
    optimizer.step()
    if system == 'REN':
        RENsys.set_param()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    lossp[epoch] = loss

# Simulate robot's movements for the next T time steps

with torch.no_grad():
    for t in range(T):
        velocity_x, velocity_y = uf[:, t]
        robot.update_state(velocity_x, velocity_y)

        # # Check for collision with obstacle
        # if robot.check_collision():
        #     print(f"Collision detected at time step {t}! Final position:", robot.position)
        #     break
    else:
        print("Obstacle avoided! Final position:", robot.position)

    # Plot robot's trajectory and obstacle
    history = torch.stack(robot.history)
    obstacle_circle = plt.Circle(obstacle_position, obstacle_radius, color='r', fill=False)

    plt.plot(history[:, 0], history[:, 1], marker='o', label='Robot trajectory')
    plt.gca().add_patch(obstacle_circle)
    plt.scatter(obstacle_position[0], obstacle_position[1], color='r', label='Obstacle center')
    plt.scatter(target[0], target[1], color='b', label='Target point')  # Add target point
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Movement and Obstacle')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(epochs), lossp.detach().numpy())
    plt.show()
