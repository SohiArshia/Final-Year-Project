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
from pureples.es_hyperneat.es_hyperneat import ESNetwork

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

sub = Substrate(input_coordinates, output_coordinates)

# ES-HyperNEAT specific parameters.
params = {"initial_depth": 0, 
          "max_depth": 2, 
          "variance_threshold": 0.03, 
          "band_threshold": 0.3, 
          "iteration_level": 1,
          "division_threshold": 0.5, 
          "max_weight": 8.0, 
          "activation": "sigmoid"}

config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'es_hyperneat_mario_cppn')


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

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(sub, cppn, params)
        net = network.create_phenotype_network()

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
        genome.fitness = distance

    with open('stats.obj', 'wb') as stats_file:
        pickle.dump(stats, stats_file)
    visualize.plot_stats(stats, ylog=False, view=False)
    if gen_counter > 2:
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
    network = ESNetwork(sub, cppn, params)
    net = network.create_phenotype_network()


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

def run_es_hyper(gens, config, substrate, activation="sigmoid", output=True):
    trials = 1

    pop = ini_pop(None, stats, config, output)
    pop.add_reporter(neat.Checkpointer(1))

    try:
        winner = pop.run(eval_parallel, gens)
        return winner, stats
    except multiprocessing.context.TimeoutError:
        os.system('make die')
        os._exit()
    

def run(gens):
    winner, stats = run_es_hyper(gens, config, sub)
    print("es_hyperneat_mario done") 
    return winner, stats


if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Run!
    winner = run(700)[0]

    # Save CPPN if and draw it + winner to file.
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    network = ESNetwork(sub, cppn, params)
    net = network.create_phenotype_network()
    draw_net(cppn, filename="es_hyperneat_mario_cppn")
    draw_net(net, filename="es_hyperneat_mario_winner")
    with open('es_hyperneat_mario_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

