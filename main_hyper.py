
"""
Most of the code in this file is similar to that
in the main_neat.py file. Please check that file
for more detailed explanatory comments.
"""

import gym
import ppaquette_gym_super_mario
import os
import sys
import neat
import visualize
import logging
import numpy as np
import pickle
import multiprocessing
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.hyperneat.hyperneat import create_phenotype_network

import time

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


def map_node_coordinates(n, y_coordinate):
    input_nodes = []
    for i in range(n):
        x = ((2*i)/(n-1))-1
        input_nodes.append((x, y_coordinate))
    return input_nodes

input_coordinates = map_node_coordinates(208,-1)
output_coordinates = map_node_coordinates(6,1)
hidden_coordinates = [map_node_coordinates(10,0), map_node_coordinates(10, 0.5)]

sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)
activations = len(hidden_coordinates) + 2

config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'hyperneat_mario_cppn')


def ini_pop(state, stats, config, output):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    files = [f for f in files if 'neat-checkpoint' in f]

    if files == []:
        print('No checkpoints found, creating new population.')
        pop = neat.population.Population(config, state)
    else:
        files = sorted([int(f.split('-')[-1]) for f in files])
        last_checkpoint_n = files[-1]
        print('Checkpoint found, resuming from checkpoint {}'.format(last_checkpoint_n))
        pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{}'.format(last_checkpoint_n))

    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def eval_fitness(genomes, config):
    activation = "sigmoid"
    i = 0
    for genome_id, genome in genomes:
        i += 1
        print('Testing population member ' + str(i))

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        net = create_phenotype_network(cppn, sub, activation)

        env = gym.make('ppaquette/meta-SuperMarioBros-Tiles-v0')
        env.reset()
        
        action = [0] * 6

        obs, reward, is_finished, info = env.step(action)
       
        distance = info['distance']
        ever_moved = False
        stuck_counter = 0

        nn_input = np.reshape(obs, (1, -1))[0]
        

        while True:
            distance_now = info['distance'] 
            output = net.activate(nn_input)
            output = [round(o) for o in output]
            
            action = output
            
            obs, reward, is_finished, info = env.step(action)
            nn_input = np.reshape(obs, (1, -1))[0]
            distance_later = info['distance']

            if is_finished:
                break

            delta_distance = info['distance'] - distance
            distance = info['distance']
            if delta_distance == 0:
                stuck_counter += 1
                if ever_moved:
                    if stuck_counter >= 70:
                        print('Stopped because he moved but then did not.')
                        break
                else:
                    if stuck_counter >= 10:
                        print('Stopped because he never moved')
                        break
            else:
                ever_moved = True
                stuck_counter = 0

            if distance_later - distance_now < -0.8:
                break 
        
        
        genome.fitness = distance
        print(genome.fitness)
        print(genome)

        env.close()

    visualize.plot_stats(stats, ylog=False, view=False)
    stats.save()

def load_stats():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    if 'stats.obj' in files:
        print('Loading pickled stats.')
        with open('stats.obj', 'rb') as stats_file:
            return pickle.load(stats_file)
    else:
        print('No pickled stats, creating new stats object.')
        return neat.statistics.StatisticsReporter()

stats = load_stats()


def eval_single(genome_tuple, config):
    cppn = neat.nn.FeedForwardNetwork.create(genome_tuple, config)
    net = create_phenotype_network(cppn, sub, "sigmoid")

    blockPrint()
    env = gym.make('ppaquette/meta-SuperMarioBros-Tiles-v0')
    env.reset()
    enablePrint()

    action = [0] * 6

    obs, reward, is_finished, info = env.step(action)
    
    distance = info['distance']
    ever_moved = False
    stuck_counter = 0

    nn_input = np.reshape(obs, (1, -1))[0]
    

    while True:
        distance_now = info['distance'] 
        output = net.activate(nn_input)
        output = [round(o) for o in output]
        
        action = output
        
        obs, reward, is_finished, info = env.step(action)
        nn_input = np.reshape(obs, (1, -1))[0]
        distance_later = info['distance']

        if is_finished:
            break

        delta_distance = info['distance'] - distance
        distance = info['distance']
        printed = False
        if delta_distance == 0:
            stuck_counter += 1
            if ever_moved:
                if stuck_counter >= 70:
                    sys.stdout.write('@')
                    printed = True
                    break
            else:
                if stuck_counter >= 10:
                    sys.stdout.write('#')
                    printed = True
                    break
        else:
            ever_moved = True
            stuck_counter = 0

        if distance_later - distance_now < -0.8:
            break
    
    if not printed: 
        sys.stdout.write('*')
    sys.stdout.flush()
    env.close()
    return distance

gen_counter = 1
def eval_parallel(genomes, config):
    global gen_counter

    print("Testing generation number {}...".format(gen_counter))

    evaluator = neat.parallel.ParallelEvaluator(12, eval_single, 2 * 60) # quit after 2 min if stuck
    evaluator.evaluate(genomes, config)

    with open('stats.obj', 'wb') as stats_file:
        pickle.dump(stats, stats_file)
    visualize.plot_stats(stats, ylog=False, view=False)
    if gen_counter > 2:
        stats.save()
    
    gen_counter += 1

def run_hyper(gens, config, substrate, activations, activation="sigmoid", output=True):
    trials = 1

    # Create population and train the network. Return winner of network running 100 episodes.
    pop = ini_pop(None, stats, config, output)
    pop.add_reporter(neat.Checkpointer(1))
    #winner = pop.run(eval_fitness, gens)
    try:
        winner = pop.run(eval_parallel, gens)
        return winner, stats
    except multiprocessing.context.TimeoutError:
        os.system('make die')
        os._exit()

    
    

def run(gens):
    winner, stats = run_hyper(gens, config, sub, activations)
    print("hyperneat_mario done") 
    return winner, stats


if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Run!
    winner = run(700)[0]

    # Save CPPN and draw it + winner to file.
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    net = create_phenotype_network(cppn, sub)
    draw_net(cppn, filename="hyperneat_mario_cppn")
    draw_net(net, filename="hyperneat_mario_winner")
    with open('hyperneat_mario_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

