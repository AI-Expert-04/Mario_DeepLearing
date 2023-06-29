import copy, os, random, time, cv2, numpy as np
from PIL import Image


class ImageDrawingEvolutionary:
    def __init__(self, image_array):
        self.original_image_np_array = image_array
        self.original_shape = self.original_image_np_array.shape
        self.copy_gene_array = self.generate_random_rgb(self.original_shape[0], self.original_shape[1])
        self.fitness = self.calculate_fitness()

    def generate_random_rgb(self, dim1, dim2):
        img = np.zeros((dim1, dim2, 3), np.uint8)

        img[:, :] = np.array([0, 0, 0])
        # if not os.path.exists('monaliza'):
        #     os.mkdir('monaliza')
        # img = cv2.imread("monaliza/im35000.JPG")
        return img

    @classmethod
    def generate_np_from_img(self, image_name):
        img = cv2.imread(image_name)
        array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return array

    def calculate_fitness(self):
        fitness = np.sum(np.absolute(self.original_image_np_array - self.copy_gene_array))
        return fitness

    def draw_image(self, save=False, filename=None):
        cv2.destroyAllWindows()
        cv2.imshow("image", self.copy_gene_array)
        if save:
            cv2.imwrite(f"monaliza/{filename}", self.copy_gene_array)
        cv2.waitKey(4)



    @classmethod
    def select_best(cls, population, parent_count):
        population = sorted(population, key=lambda x: x.fitness)
        return population[:parent_count]

    @classmethod
    def select_random(cls, population, parent_count):
        pop_random = copy.deepcopy(population)
        random.shuffle(pop_random)
        return pop_random[:parent_count]

    @classmethod
    def crossover(cls, parents):
        child_1 = copy.deepcopy(parents[0])
        child_2 = copy.deepcopy(parents[1])

        if random.choice(['r', 'c']) == 'r':
            random_row_index_1 = random.randint(0, parents[0].original_shape[0]-1)
            random_row_index_2 = random.randint(0, parents[0].original_shape[0]-1)
            if random_row_index_1 > random_row_index_2:
                random_row_index_1, random_row_index_2 = random_row_index_2, random_row_index_1

            # random_col_index_1 = random.randint(0,parents[0].original_shape[1]-1)
            # random_col_index_2 = random.randint(0,parents[0].original_shape[1]-1)
            # if random_col_index_1>random_col_index_2:
            #     random_col_index_1,random_col_index_2=random_col_index_2,random_col_index_1
            # child_1.copy_gene_array[random_row_index_1][random_col_index_1:random_col_index_2] = parents[1].copy_gene_array[random_row_index_1][random_col_index_1:random_col_index_2]
            # child_2.copy_gene_array[random_row_index_1][random_col_index_1:random_col_index_2] = parents[0].copy_gene_array[random_row_index_1][random_col_index_1:random_col_index_2]

            child_1.copy_gene_array[random_row_index_1:random_row_index_2] = parents[1].copy_gene_array[random_row_index_1:random_row_index_2]
            child_2.copy_gene_array[random_row_index_1:random_row_index_2] = parents[0].copy_gene_array[random_row_index_1:random_row_index_2]

            child_1.fitness = child_1.calculate_fitness()
            child_2.fitness = child_2.calculate_fitness()
            return child_1, child_2
        else:
            random_col_index_1 = random.randint(0, parents[0].original_shape[1] - 1)
            random_col_index_2 = random.randint(0, parents[0].original_shape[1] - 1)
            if random_col_index_1 > random_col_index_2:
                random_col_index_1, random_col_index_2 = random_col_index_2, random_col_index_1

            # random_col_index_1 = random.randint(0,parents[0].original_shape[1]-1)
            # random_col_index_2 = random.randint(0,parents[0].original_shape[1]-1)
            # if random_col_index_1>random_col_index_2:
            #     random_col_index_1,random_col_index_2=random_col_index_2,random_col_index_1
            # child_1.copy_gene_array[random_col_index_1][random_col_index_1:random_col_index_2] = parents[1].copy_gene_array[random_col_index_1][random_col_index_1:random_col_index_2]
            # child_2.copy_gene_array[random_col_index_1][random_col_index_1:random_col_index_2] = parents[0].copy_gene_array[random_col_index_1][random_col_index_1:random_col_index_2]

            child_1.copy_gene_array[:][random_col_index_1:random_col_index_2] = parents[1].copy_gene_array[:][
                                                                             random_col_index_1:random_col_index_2]
            child_2.copy_gene_array[:][random_col_index_1:random_col_index_2] = parents[0].copy_gene_array[:][
                                                                             random_col_index_1:random_col_index_2]

            child_1.fitness = child_1.calculate_fitness()
            child_2.fitness = child_2.calculate_fitness()
            return child_1, child_2

    @classmethod
    def mutation(cls, chromosome):

        if random.random() < 0.5:
            np_arr = copy.deepcopy(chromosome.copy_gene_array)

            rand_ind1 = random.randint(0, len(np_arr)-1)
            rand_ind2 = random.randint(0, len(np_arr)-1)
            rand_ind3 = random.randint(0, len(np_arr[0])-1)
            rand_ind4 = random.randint(0, len(np_arr[0])-1)
            if rand_ind1 > rand_ind2:
                rand_ind1, rand_ind2 = rand_ind2, rand_ind1
            if rand_ind3 > rand_ind4:
                rand_ind3, rand_ind4 = rand_ind4, rand_ind3

            # adding solid colors
            # choice = np.array([random.randint(1,255),random.randint(1,255),random.randint(1,255)])
            # for i in range(rand_ind1,rand_ind2):
            #     for j in range(rand_ind3,rand_ind4):
            #         np_arr[i][j]=choice

            # adding changes in colors

            rgb = random.choice(['r', 'g', 'b'])
            if rgb == 'r':
                if random.choice([0, 1]) == 0:
                    for i in range(rand_ind1, rand_ind2):
                        for j in range(rand_ind3, rand_ind4):
                            if np_arr[i][j][0] + 15 < 256:
                                np_arr[i][j][0] += 15

                else:
                    for i in range(rand_ind1, rand_ind2):
                        for j in range(rand_ind3, rand_ind4):
                            if np_arr[i][j][0] - 15 > 256:
                                np_arr[i][j][0] -= 15
            elif rgb == 'g':
                if random.choice([0, 1]) == 0:
                    for i in range(rand_ind1, rand_ind2):
                        for j in range(rand_ind3, rand_ind4):
                            if np_arr[i][j][1] + 15 < 256:
                                np_arr[i][j][1] += 15

                else:
                    for i in range(rand_ind1, rand_ind2):
                        for j in range(rand_ind3, rand_ind4):
                            if np_arr[i][j][1] - 15 > 256:
                                np_arr[i][j][1] -= 15
            else:
                if random.choice([0, 1]) == 0:
                    for i in range(rand_ind1, rand_ind2):
                        for j in range(rand_ind3, rand_ind4):
                            if np_arr[i][j][2] + 15 < 256:
                                np_arr[i][j][2] += 15

                else:
                    for i in range(rand_ind1, rand_ind2):
                        for j in range(rand_ind3, rand_ind4):
                            if np_arr[i][j][2] - 15 > 256:
                                np_arr[i][j][2] -= 15

            chromosome.copy_gene_array = np_arr
            chromosome.fitness = chromosome.calculate_fitness()

        else:
            np_arr = copy.deepcopy(chromosome.copy_gene_array)
            rand_ind1 = random.randint(0, len(np_arr)-1)
            rand_ind4 = random.randint(0, len(np_arr[0])-1)
            np_arr[rand_ind1, rand_ind4] = np.array([random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)])
            chromosome.copy_gene_array = np_arr
            chromosome.fitness = chromosome.calculate_fitness()

        return chromosome


if __name__ == '__main__':
    image_name = "IMG_9059.JPG"

    image_np_array = ImageDrawingEvolutionary.generate_np_from_img(image_name)
    image_np_array = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2RGB)

    population = []
    generation = 0

    for i in range(10):
        population.append(ImageDrawingEvolutionary(image_np_array))

    print("-------------After Select----------------")

    best = ImageDrawingEvolutionary.select_best(population, 1)[0]
    best.draw_image(True, "im"+str(generation)+".JPG")

    print(best.fitness)
    # generation = 35001

    while best.fitness > 100:
        new_population = []

        for i in range(0, len(population), 2):
            # a= time.time()
            random_parents = ImageDrawingEvolutionary.select_random(population, 2)
            # b=time.time()
            # print(b-a)
            # random_parents = ImageDrawingEvolutionary.select_binary_tour(population, 3, 2)

            # a= time.time()
            child_1, child_2 = ImageDrawingEvolutionary.crossover(random_parents)
            # b=time.time()
            # print(b-a)
            # a= time.time()
            child_1, child_2 = ImageDrawingEvolutionary.mutation(child_1), ImageDrawingEvolutionary.mutation(child_2)
            # b=time.time()
            # print(b-a)

            new_population.append(child_1)
            new_population.append(child_2)

        population = ImageDrawingEvolutionary.select_best(population+new_population, len(population))

        print("-------------After Select----------------")
        print(f"Generation : {generation}")
        for each in population:
            print(each.fitness)

        print("Best:-")

        best = ImageDrawingEvolutionary.select_best(population, 1)[0]
        if generation % 100 == 0:
            best.draw_image(True, "im"+str(generation)+".JPG")

        else:
            best.draw_image()
        # 2131890257
        print(best.fitness)
        generation += 1
