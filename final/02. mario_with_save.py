# 02. mario_with_save.py
import retro
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
import numpy as np
import random
import os


relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))


class Chromosome:
    def __init__(self):
        self.w1 = np.random.uniform(low=-1, high=1, size=(80, 9))
        self.b1 = np.random.uniform(low=-1, high=1, size=(9,))

        self.w2 = np.random.uniform(low=-1, high=1, size=(9, 6))
        self.b2 = np.random.uniform(low=-1, high=1, size=(6,))

        self.distance = 0
        self.max_distance = 0
        self.frames = 0
        self.stop_frames = 0
        self.win = 0

    def predict(self, data):
        l1 = relu(np.matmul(data, self.w1) + self.b1)
        output = sigmoid(np.matmul(l1, self.w2) + self.b2)
        result = (output > 0.5).astype(np.int)
        return result

    def fitness(self):
        # # 1. 시간을 반영한 거리 점수
        # fit = self.distance ** 1.8 - self.frames ** 1.5
        # # 2. 조금이라도 움직인 경우 보너스 +2500
        # # if self.distance > 50:
        # #     fit += 2500
        # fit += min(max(self.distance - 50, 0), 1) * 2500
        # # 3. 목적지에 도달한 경우 보너스 +1000000
        # fit += self.win * 1000000
        # # 4. 아무리 못해도 기본점수 1점
        # # if fit < 1:
        # #     fit = 1
        # fit = max(fit, 1)
        # fit = int(fit)
        # return fit
        return int(max(self.distance ** 1.8 - self.frames ** 1.5 + min(max(self.distance - 50, 0), 1) * 2500 + self.win * 1000000, 1))


class GeneticAlgorithm:
    def __init__(self):
        self.chromosomes = [Chromosome() for _ in range(10)]
        self.generation = 0
        self.current_chromosome_index = 0

    def roulette_wheel_selection(self):
        result = []
        fitness_sum = 0
        for chromosome in self.chromosomes:
            fitness_sum += chromosome.fitness()

        for _ in range(2):
            pick = random.uniform(0, fitness_sum)
            current = 0
            for chromosome in self.chromosomes:
                current += chromosome.fitness()
                if current > pick:
                    result.append(chromosome)
                    break

        return result

    def elitist_preserve_selection(self):
        sorted_chromosomes = sorted(self.chromosomes, key=lambda x: x.fitness(), reverse=True)
        return sorted_chromosomes[:2]

    def selection(self):
        result = self.roulette_wheel_selection()
        return result

    def simulated_binary_crossover(self, parent_chromosome1, parent_chromosome2):
        rand = np.random.random(parent_chromosome1.shape)
        gamma = np.empty(parent_chromosome1.shape)
        gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (100 + 1))
        gamma[rand > 0.5] = (2 * rand[rand > 0.5]) ** (1.0 / (100 + 1))
        child_chromosome1 = 0.5 * ((1 + gamma) * parent_chromosome1 + (1 - gamma) * parent_chromosome2)
        child_chromosome2 = 0.5 * ((1 - gamma) * parent_chromosome1 + (1 + gamma) * parent_chromosome2)
        return child_chromosome1, child_chromosome2

    def crossover(self, chromosome1, chromosome2):
        child1 = Chromosome()
        child2 = Chromosome()

        child1.w1, child2.w1 = self.simulated_binary_crossover(chromosome1.w1, chromosome2.w1)
        child1.b1, child2.b1 = self.simulated_binary_crossover(chromosome1.b1, chromosome2.b1)
        child1.w2, child2.w2 = self.simulated_binary_crossover(chromosome1.w2, chromosome2.w2)
        child1.b2, child2.b2 = self.simulated_binary_crossover(chromosome1.b2, chromosome2.b2)

        return child1, child2

    def static_mutation(self, chromosome):
        mutation_array = np.random.random(chromosome.shape) < 0.05
        gaussian_mutation = np.random.normal(size=chromosome.shape)
        chromosome[mutation_array] += gaussian_mutation[mutation_array]

    def mutation(self, chromosome):
        self.static_mutation(chromosome.w1)
        self.static_mutation(chromosome.b1)
        self.static_mutation(chromosome.w2)
        self.static_mutation(chromosome.b2)

    def next_generation(self):
        # 저장
        if not os.path.exists('../data'):
            os.mkdir('../data')
        if not os.path.exists(f'../data/{self.generation}'):
            os.mkdir(f'../data/{self.generation}')
        for i in range(10):
            if not os.path.exists(f'../data/{self.generation}/{i}'):
                os.mkdir(f'../data/{self.generation}/{i}')
            np.save(f'../data/{self.generation}/{i}/w1.npy', self.chromosomes[i].w1)
            np.save('../data/' + str(self.generation) + '/' + str(i) + '/b1.npy', self.chromosomes[i].b1)
            np.save(f'../data/{self.generation}/{i}/w2.npy', self.chromosomes[i].w2)
            np.save(f'../data/{self.generation}/{i}/b2.npy', self.chromosomes[i].b2)

        next_chromosomes = []
        next_chromosomes.extend(self.elitist_preserve_selection())
        # for c in self.elitist_preserve_selection():
        #     next_chromosomes.append(c)

        for i in range(4):
            selected_chromosome = self.selection()

            child_chromosome1, child_chromosome2 = self.crossover(
                selected_chromosome[0],
                selected_chromosome[1])

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


class Mario(QWidget):
    def __init__(self):
        super().__init__()

        self.env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
        self.env.reset()

        screen = self.env.get_screen()
        self.screen_width = screen.shape[0] * 2
        self.screen_height = screen.shape[1] * 2

        self.setFixedSize(self.screen_width + 600, self.screen_height + 100)
        self.setWindowTitle('GA-Mario')

        self.screen_label = QLabel(self)
        self.screen_label.setGeometry(0, 0, self.screen_width, self.screen_height)

        self.ga = GeneticAlgorithm()

        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_game)
        self.game_timer.start(1000 // 60)

        self.show()

    def update_game(self):
        screen = self.env.get_screen()
        original = QImage(screen, screen.shape[1], screen.shape[0], QImage.Format_RGB888)
        qimage = QImage(original)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(self.screen_width, self.screen_height, Qt.IgnoreAspectRatio)
        self.screen_label.setPixmap(pixmap)
        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        ram = self.env.get_ram()

        full_screen_tiles = ram[0x0500:0x069F + 1]

        full_screen_tile_count = full_screen_tiles.shape[0]

        full_screen_page1_tile = full_screen_tiles[:full_screen_tile_count // 2].reshape((13, 16))
        full_screen_page2_tile = full_screen_tiles[full_screen_tile_count // 2:].reshape((13, 16))

        full_screen_tiles = np.concatenate((full_screen_page1_tile, full_screen_page2_tile), axis=1).astype(np.int)

        enemy_drawn = ram[0x000F:0x0013 + 1]

        enemy_horizon_position = ram[0x006E:0x0072 + 1]
        enemy_screen_position_x = ram[0x0087:0x008B + 1]
        enemy_position_y = ram[0x00CF:0x00D3 + 1]
        enemy_position_x = (enemy_horizon_position * 256 + enemy_screen_position_x) % 512

        enemy_tile_position_x = (enemy_position_x + 8) // 16
        enemy_tile_position_y = (enemy_position_y - 8) // 16 - 1

        for i in range(5):
            if enemy_drawn[i] == 1:
                ey = enemy_tile_position_y[i]
                ex = enemy_tile_position_x[i]
                if 0 <= ex < full_screen_tiles.shape[1] and 0 <= ey < full_screen_tiles.shape[0]:
                    full_screen_tiles[ey][ex] = -1

        painter.setPen(QPen(Qt.black))

        current_screen_page = ram[0x071A]
        screen_position = ram[0x071C]
        screen_offset = (256 * current_screen_page + screen_position) % 512
        screen_tile_offset = screen_offset // 16

        screen_tiles = np.concatenate((full_screen_tiles, full_screen_tiles), axis=1)[:, screen_tile_offset:screen_tile_offset + 16]

        for i in range(screen_tiles.shape[0]):
            for j in range(screen_tiles.shape[1]):
                if screen_tiles[i][j] > 0:
                    screen_tiles[i][j] = 1
                    painter.setBrush(QBrush(Qt.cyan))
                elif screen_tiles[i][j] == -1:
                    screen_tiles[i][j] = 2
                    painter.setBrush(QBrush(Qt.red))
                else:
                    painter.setBrush(QBrush(Qt.gray))
                painter.drawRect(self.screen_width + 16 * j, 16 * i, 16, 16)

        player_position_x = ram[0x03AD]
        player_position_y = ram[0x03B8]

        player_tile_position_x = (player_position_x + 8) // 16
        player_tile_position_y = (player_position_y + 8) // 16 - 1

        painter.setBrush(QBrush(Qt.blue))
        painter.drawRect(self.screen_width + 16 * player_tile_position_x, 16 * player_tile_position_y, 16, 16)

        frame_x = player_tile_position_x
        frame_y = 2
        painter.setPen(QPen(Qt.magenta, 4, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.screen_width + 16 * frame_x, 16 * frame_y, 16 * 8, 16 * 10)

        input_data = screen_tiles[frame_y:frame_y+10, frame_x:frame_x+8]
        if 2 <= player_tile_position_y <= 11:
            input_data[player_tile_position_y - 2][0] = 2

        input_data = input_data.flatten()

        current_chromosome = self.ga.chromosomes[self.ga.current_chromosome_index]
        current_chromosome.frames += 1

        player_horizon_position = ram[0x006D]
        player_screen_position_x = ram[0x0086]
        current_chromosome.distance = 256 * player_horizon_position + player_screen_position_x

        if current_chromosome.max_distance < current_chromosome.distance:
            current_chromosome.max_distance = current_chromosome.distance
            current_chromosome.stop_frames = 0
        else:
            current_chromosome.stop_frames += 1

        player_float_state = ram[0x001D]
        player_state = ram[0x000E]
        player_vertical_screen_position = ram[0x00B5]

        if player_float_state == 0x03 or player_state in (0x06, 0x0B) or player_vertical_screen_position >= 2 \
            or current_chromosome.stop_frames > 180:
            if player_float_state == 0x03:
                current_chromosome.win = 1

            print(f'{self.ga.current_chromosome_index + 1}번 마리오: {current_chromosome.fitness()}')

            self.ga.current_chromosome_index += 1

            if self.ga.current_chromosome_index == 10:
                self.ga.next_generation()
                print(f'== {self.ga.generation} 세대 ==')

            self.env.reset()
        else:
            predict = current_chromosome.predict(input_data)
            press_buttons = np.array([predict[5], 0, 0, 0, predict[0], predict[1], predict[2], predict[3], predict[4]])
            self.env.step(press_buttons)

            for i in range(6):
                if predict[i] == 1:
                    painter.setBrush(QBrush(Qt.magenta))
                else:
                    painter.setBrush(QBrush(Qt.gray))
                painter.drawEllipse(self.screen_width + i * 40, 450, 10 * 2, 10 * 2)
                text = ('U', 'D', 'L', 'R', 'A', 'B')[i]
                painter.drawText(self.screen_width + i * 40, 480, text)

        painter.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Mario()
    sys.exit(app.exec_())
