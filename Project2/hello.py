import math
import numpy as np
from itertools import product
from typing import List, Tuple
from utils import *
from grid import *
from particle import Particle
import setting

np.random.seed(setting.RANDOM_SEED)

def create_random(count: int, grid: CozGrid) -> List[Particle]:
    particles = [Particle(*grid.random_free_place()) for _ in range(count)]
    return particles

def motion_update(old_particles: List[Particle], odometry_measurement: Tuple, grid: CozGrid) -> List[Particle]:
    new_particles = []
    for particle in old_particles:
        x_g, y_g, h_g = particle.xyh
        dx_r, dy_r, dh_r = odometry_measurement
        dx_g, dy_g = rotate_point(dx_r, dy_r, h_g)
        dx_g = add_gaussian_noise(dx_g, setting.ODOM_TRANS_SIGMA)
        dy_g = add_gaussian_noise(dy_g, setting.ODOM_TRANS_SIGMA)
        dh_r = add_gaussian_noise(dh_r, setting.ODOM_HEAD_SIGMA)
        x, y, h = x_g + dx_g, y_g + dy_g, (h_g + dh_r) % 360
        new_particle = Particle(x, y, h)
        if grid.is_in(x, y) and grid.is_free(x, y):
            new_particles.append(new_particle)
    return new_particles

def generate_marker_pairs(robot_marker_list: List[Tuple], particle_marker_list: List[Tuple]) -> List[Tuple]:
    marker_pairs = []
    while robot_marker_list and particle_marker_list:
        pair = min(
            product(particle_marker_list, robot_marker_list),
            key=lambda pair: grid_distance(pair[0][0], pair[0][1], pair[1][0], pair[1][1]),
        )
        marker_pairs.append(pair)
        particle_marker_list.remove(pair[0])
        robot_marker_list.remove(pair[1])
    return marker_pairs

def marker_likelihood(robot_marker: Tuple, particle_marker: Tuple) -> float:
    x1, y1, t1 = robot_marker
    x2, y2, t2 = particle_marker
    distance = grid_distance(x1, y1, x2, y2)
    angle = diff_heading_deg(t1, t2)
    trans_prob = math.exp(- (distance ** 2) / (2 * setting.MARKER_TRANS_SIGMA ** 2))
    angle_prob = math.exp(- (angle ** 2) / (2 * setting.MARKER_HEAD_SIGMA ** 2))
    return trans_prob * angle_prob

def particle_likelihood(robot_marker_list: List[Tuple], particle_marker_list: List[Tuple]) -> float:
    marker_pairs = generate_marker_pairs(robot_marker_list, particle_marker_list)
    if not marker_pairs:
        return 0.0
    likelihood = 1.0
    for robot_marker, particle_marker in marker_pairs:
        likelihood *= marker_likelihood(robot_marker, particle_marker)
    return likelihood

def measurement_update(particles: List[Particle], measured_marker_list: List[Tuple], grid: CozGrid) -> List[Particle]:
    measured_particles = []
    particle_weights = []
    num_rand_particles = 25
    
    if measured_marker_list:
        for p in particles:
            x, y = p.xy
            if grid.is_in(x, y) and grid.is_free(x, y):
                robot_marker_list = measured_marker_list.copy()
                particle_marker_list = p.read_markers(grid)
                l = particle_likelihood(robot_marker_list, particle_marker_list)
            else:
                l = 0.0
            particle_weights.append(l)
    else:
        particle_weights = [1.0] * len(particles)
    
    total_weight = sum(particle_weights)
    if total_weight == 0:
        return create_random(setting.PARTICLE_COUNT, grid)
    
    particle_weights = [w / total_weight for w in particle_weights]
    resampled_particles = list(np.random.choice(particles, size=setting.PARTICLE_COUNT - num_rand_particles, p=particle_weights, replace=True))
    resampled_particles += create_random(num_rand_particles, grid)
    return resampled_particles
