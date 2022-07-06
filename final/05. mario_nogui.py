# 05. mario_nogui.py
import retro
import numpy as np
import sys
import random
import os

relu = lambda X: np.maximum(0, X)
sigmoid = lambda X: 1.0 / (1.0 + np.exp(-X))


class Chromosome:
    def __init__(self):
        self.w1 = np.random.uniform(low=-1, high=1, size=(80, 9))
        self.b1 = np.random.uniform(low=-1, high=1, size=(9,))

        self.w2 = np.random.uniform(low=-1, high=1, size=(9, 6))
        self.b2 = np.random.uniform(low=-1, high=1, size=(6,))

        self.l1 = None

        self.distance = 0
        self.max_distance = 0
        self.frames = 0
        self.stop_frames = 0
        self.win = 0

    def predict(self, data):
        self.l1 = relu(np.matmul(data, self.w1) + self.b1)
        output = sigmoid(np.matmul(self.l1, self.w2) + self.b2)
        result = (output > 0.5).astype(np.int)
        return result

    def fitness(self):
        return int(max(self.distance ** 1.8 - self.frames ** 1.5 + min(max(self.distance - 50, 0), 1) * 2500 + self.win * 1000000, 1))


class GeneticAlgorithm:
    def __init__(self):
        self.generation = 0
        self.chromosomes = [Chromosome() for _ in range(10)]
        self.current_chromosome_index = 0

    def elitist_preserve_selection(self):
        sort_chromosomes = sorted(self.chromosomes, key=lambda x: x.fitness(), reverse=True)
        return sort_chromosomes[:2]

    def roulette_wheel_selection(self):
        result = []
        fitness_sum = sum(c.fitness() for c in self.chromosomes)
        for _ in range(2):
            pick = random.uniform(0, fitness_sum)
            current = 0
            for chromosome in self.chromosomes:
                current += chromosome.fitness()
                if current > pick:
                    result.append(chromosome)
                    break
        return result

    def SBX(self, p1, p2):
        rand = np.random.random(p1.shape)
        gamma = np.empty(p1.shape)
        gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (100 + 1))
        gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (100 + 1))
        c1 = 0.5 * ((1 + gamma) * p1 + (1 - gamma) * p2)
        c2 = 0.5 * ((1 - gamma) * p1 + (1 + gamma) * p2)
        return c1, c2

    def crossover(self, chromosome1, chromosome2):
        child1 = Chromosome()
        child2 = Chromosome()

        child1.w1, child2.w1 = self.SBX(chromosome1.w1, chromosome2.w1)
        child1.b1, child2.b1 = self.SBX(chromosome1.b1, chromosome2.b1)
        child1.w2, child2.w2 = self.SBX(chromosome1.w2, chromosome2.w2)
        child1.b2, child2.b2 = self.SBX(chromosome1.b2, chromosome2.b2)

        return child1, child2

    def static_mutation(self, data):
        mutation_array = np.random.random(data.shape) < 0.05
        gaussian_mutation = np.random.normal(size=data.shape)
        data[mutation_array] += gaussian_mutation[mutation_array]

    def mutation(self, chromosome):
        self.static_mutation(chromosome.w1)
        self.static_mutation(chromosome.b1)
        self.static_mutation(chromosome.w2)
        self.static_mutation(chromosome.b2)

    def next_generation(self):
        if not os.path.exists('../data'):
            os.mkdir('../data')
        if not os.path.exists('../data/' + str(self.generation)):
            os.mkdir('../data/' + str(self.generation))
        for i in range(10):
            if not os.path.exists('../data/' + str(self.generation) + '/' + str(i)):
                os.mkdir('../data/' + str(self.generation) + '/' + str(i))
            np.save('../data/' + str(self.generation) + '/' + str(i) + '/w1.npy', self.chromosomes[i].w1)
            np.save('../data/' + str(self.generation) + '/' + str(i) + '/w2.npy', self.chromosomes[i].w2)
            np.save('../data/' + str(self.generation) + '/' + str(i) + '/b1.npy', self.chromosomes[i].b1)
            np.save('../data/' + str(self.generation) + '/' + str(i) + '/b2.npy', self.chromosomes[i].b2)
            np.save('../data/' + str(self.generation) + '/' + str(i) + '/fitness.npy', np.array([self.chromosomes[i].fitness()]))

        print(f'{self.generation}세대 시뮬레이션 완료.')

        next_chromosomes = []
        next_chromosomes.extend(self.elitist_preserve_selection())
        print(f'엘리트 적합도: {next_chromosomes[0].fitness()}')

        for i in range(4):
            selected_chromosome = self.roulette_wheel_selection()

            child_chromosome1, child_chromosome2 = self.crossover(selected_chromosome[0], selected_chromosome[1])
            self.mutation(child_chromosome1)
            self.mutation(child_chromosome2)

            next_chromosomes.append(child_chromosome1)
            next_chromosomes.append(child_chromosome2)

        self.chromosomes = next_chromosomes
        for c in self.chromosomes:
            c.distance = 0
            c.max_distance = 0
            c.frames = 0
            c.stop_frames = 0
            c.win = 0

        self.generation += 1
        self.current_chromosome_index = 0


class Mario:
    def __init__(self):
        super().__init__()
        self.env = retro.make(game='SuperMarioBros-Nes', state=f'Level1-1')
        self.env.reset()

        self.ga = GeneticAlgorithm()

        self.nogui()

    def nogui(self):
        while True:
            ram = self.env.get_ram()

            full_screen_tiles = ram[0x0500:0x069F + 1]
            full_screen_tile_count = full_screen_tiles.shape[0]

            full_screen_page1_tiles = full_screen_tiles[:full_screen_tile_count // 2].reshape((-1, 16))
            full_screen_page2_tiles = full_screen_tiles[full_screen_tile_count // 2:].reshape((-1, 16))

            full_screen_tiles = np.concatenate((full_screen_page1_tiles, full_screen_page2_tiles), axis=1).astype(
                np.int)

            enemy_drawn = ram[0x000F:0x0014]
            enemy_horizontal_position_in_level = ram[0x006E:0x0072 + 1]
            enemy_x_position_on_screen = ram[0x0087:0x008B + 1]
            enemy_y_position_on_screen = ram[0x00CF:0x00D3 + 1]

            for i in range(5):
                if enemy_drawn[i] == 1:
                    ex = (((enemy_horizontal_position_in_level[i] * 256) + enemy_x_position_on_screen[
                        i]) % 512 + 8) // 16
                    ey = (enemy_y_position_on_screen[i] - 8) // 16 - 1
                    if 0 <= ex < full_screen_tiles.shape[1] and 0 <= ey < full_screen_tiles.shape[0]:
                        full_screen_tiles[ey][ex] = -1

            current_screen_in_level = ram[0x071A]
            screen_x_position_in_level = ram[0x071C]
            screen_x_position_offset = (256 * current_screen_in_level + screen_x_position_in_level) % 512
            sx = screen_x_position_offset // 16

            screen_tiles = np.concatenate((full_screen_tiles, full_screen_tiles), axis=1)[:, sx:sx + 16]

            for i in range(screen_tiles.shape[0]):
                for j in range(screen_tiles.shape[1]):
                    if screen_tiles[i][j] > 0:
                        screen_tiles[i][j] = 1
                    if screen_tiles[i][j] == -1:
                        screen_tiles[i][j] = 2

            player_x_position_current_screen_offset = ram[0x03AD]
            player_y_position_current_screen_offset = ram[0x03B8]
            px = (player_x_position_current_screen_offset + 8) // 16
            py = (player_y_position_current_screen_offset + 8) // 16 - 1

            ix = px
            iy = 2

            input_data = screen_tiles[iy:iy + 10, ix:ix + 8]

            if 2 <= py <= 11:
                input_data[py - 2][0] = 2

            input_data = input_data.flatten()

            current_chromosome = self.ga.chromosomes[self.ga.current_chromosome_index]
            current_chromosome.frames += 1
            current_chromosome.distance = ram[0x006D] * 256 + ram[0x0086]

            if current_chromosome.max_distance < current_chromosome.distance:
                current_chromosome.max_distance = current_chromosome.distance
                current_chromosome.stop_frame = 0
            else:
                current_chromosome.stop_frame += 1

            if ram[0x001D] == 3 or ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2 or current_chromosome.stop_frame > 180:
                if ram[0x001D] == 3:
                    current_chromosome.win = 1

                print(f'{self.ga.current_chromosome_index + 1}번 마리오: {current_chromosome.fitness()}')

                self.ga.current_chromosome_index += 1

                if self.ga.current_chromosome_index == 10:
                    self.ga.next_generation()
                    print(f'== {self.ga.generation}세대 ==')

                self.env.reset()
            else:
                predict = current_chromosome.predict(input_data)
                press_buttons = np.array(
                    [predict[5], 0, 0, 0, predict[0], predict[1], predict[2], predict[3], predict[4]])
                self.env.step(press_buttons)


if __name__ == '__main__':
    mario = Mario()
