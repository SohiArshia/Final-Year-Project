
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

import time


"""
Functions for disabling and enabling output to console.
These are used to stop the NES emulator from logging to the
console every time it starts. 
"""
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


# Load the config file
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config-mario')


# Initialise the population 
def ini_pop(state, stats, config, output=True):
    global gen_counter

    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    for f in files:
        if 'gen_counter' in f:
            with open('gen_counter.txt') as gen_counter_f:
                gen_counter = int(gen_counter_f.readline())

    files = [f for f in files if 'neat-checkpoint' in f]

    if files == []:
        print('No checkpoints found, creating new population.')
        pop = neat.population.Population(config, state)
    else:
        files = sorted([int(f.split('-')[-1]) for f in files])
        last_checkpoint_n = files[-1]
        print('Checkpoint found, resuming from checkpoint {}'.format(last_checkpoint_n))
        pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-{}'.format(last_checkpoint_n))
        #pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-200')

    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop

"""
Function to evaluate the fitness of each population member.

This variant of the function is used when executing the program
in single core mode.
"""
def eval_fitness(genomes, config):
    i = 0
    
    # For each population member 
    for genome_id, genome in genomes:
        i += 1
        print('Testing population member ' + str(i))

        # Set up the ANN
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Load the GYM environment
        env = gym.make('ppaquette/meta-SuperMarioBros-Tiles-v0')
        env.reset()
        
        """
        For the first input, have there be no pressed buttons. 
        This is only used to retrieve the first batch of data
        from the GYM environemnt. 

        All subsequent inputs will be based on ANN output. 
        """
        action = [0] * 6


        obs, reward, is_finished, info = env.step(action)
       

        distance = info['distance']
        ever_moved = False
        stuck_counter = 0

        """
        Flatten array representing the tiles before it is
        passed as input to the ANN.
        """
        nn_input = np.reshape(obs, (1, -1))[0]
        
        # NES emulation loop
        while True:
            distance_now = info['distance'] 
            output = net.activate(nn_input)
            output = [round(o) for o in output]
            
            action = output
            
            obs, reward, is_finished, info = env.step(action)
            nn_input = np.reshape(obs, (1, -1))[0]
            distance_later = info['distance']

            # Stop emulation if the level is completed.
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

            # Code to prevent Mario from walking backwards for too long
            if distance_later - distance_now < -0.8:
                break
        
        
        genome.fitness = distance

        # Close the emulator/GYM environment 
        env.close()

    # Plot statistics
    visualize.plot_stats(stats, ylog=False, view=False)
    stats.save()

# Function used to load statistics data if resuming a previously started experiment
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

"""
Function to evaluate the fitness of each population member.

This variant of the function is used when executing the program
in multi-core mode.

Please check the single-core version of this function for 
explanatory comments. 
"""
def eval_single(genome_tuple, config):
    global current_level
    net = neat.nn.FeedForwardNetwork.create(genome_tuple, config)

    blockPrint()
    env = gym.make(current_level)
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

# Levels to train on
levels = ['ppaquette/SuperMarioBros-{}-1-Tiles-v0'.format(i) for i in [2, 3, 4]]
gen_counter = 1
current_level = None
"""
Function that controls multi-core execution.

This is where code that runs after each whole
generation is evaluated runs. 
"""
def eval_parallel(genomes, config):
    global current_level
    global gen_counter
    current_level = levels[gen_counter % len(levels)]

    print("Testing generation number {} on level {}...".format(gen_counter, current_level))

    evaluator = neat.parallel.ParallelEvaluator(12, eval_single, 2 * 60) # quit after 2 min if stuck
    evaluator.evaluate(genomes, config)

    with open('stats.obj', 'wb') as stats_file:
        pickle.dump(stats, stats_file)
    visualize.plot_stats(stats, ylog=False, view=False)

    if len(stats.best_genomes(1)) == 0:
        print("There is no best genome")
    else:
        visualize.draw_net(config, stats.best_genome())

    if gen_counter > 1:
        stats.save()
        with open('{}.csv'.format(current_level.replace('/', '-')), 'a') as f:
            print('{}, {}, {}, {}'.format(gen_counter, stats.get_fitness_stat(max)[-1], stats.get_fitness_mean()[-1], stats.get_fitness_stdev()[-1]), file=f)
    
    with open('gen_counter.txt', 'w') as f:
        f.write(str(gen_counter))

    gen_counter += 1

def run_neat(gens, config):
    trials = 1

    # Create population and train the network. Return winner of network running 100 episodes.
    pop = ini_pop(None, stats, config)
    pop.add_reporter(neat.Checkpointer(1))
    try:
        winner = pop.run(eval_parallel, gens)
        return winner, stats
    except multiprocessing.context.TimeoutError:
        print('A Timeout Exception occured. Restarting simulation.')
        os.system('make die')
        os._exit()

    
    

def run(gens):
    winner, stats = run_neat(gens, config)
    print("neat_mario done") 
    return winner, stats

# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Run!
    #winner = run(700)[0]
    winner = run(5000)[0]

    # Save net if wished reused and draw it + winner to file.
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('neat_mario_net.pkl', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
